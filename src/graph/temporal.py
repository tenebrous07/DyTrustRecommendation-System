"""Temporal interest modelling (Module 13).

Stated plainly, because the project name invites the opposite reading: this is
**not** a temporal graph neural network. There is no TGN, no TGAT, no temporal
attention. It is exponential recency weighting on edges plus periodic
preference snapshots, and it is described that way in the README, the
architecture document and the viva notes.

That choice is defensible on this data rather than merely convenient: Phase 2
measured a mean per-user category drift of 0.433 between the first and second
half of each user's history, and the category-share plot shows platform mix
turning over with real console launches. Recency weighting addresses a drift
that demonstrably exists here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

__all__ = ["exponential_decay", "preference_snapshots", "interest_drift"]

logger = get_logger(__name__)

_SECONDS_PER_YEAR = 365.25 * 24 * 3600


def exponential_decay(
    timestamps: pd.Series | np.ndarray,
    reference_timestamp: int,
    lambda_per_year: float = 0.5,
    min_weight: float = 0.01,
) -> np.ndarray:
    """``exp(-lambda * age_years)``, floored, clamped at 1.0 for future rows.

    Args:
        timestamps: Milliseconds since epoch.
        reference_timestamp: The "present" instant. Must be the maximum
            *training* timestamp; wall-clock now would place the reference
            inside the evaluation period and make results depend on the date
            the script was run.
    """
    ages = (reference_timestamp - np.asarray(timestamps, dtype=np.float64)) / 1000.0
    age_years = np.clip(ages / _SECONDS_PER_YEAR, 0.0, None)
    return np.clip(np.exp(-lambda_per_year * age_years), min_weight, 1.0).astype(np.float32)


def preference_snapshots(
    interactions: pd.DataFrame,
    item_to_category: dict[str, str],
    granularity: str = "YE",
    top_k: int = 3,
) -> pd.DataFrame:
    """Each user's dominant categories per period, for the dashboard timeline.

    Returns a long frame of ``user_id, period, category, share, rank``.
    """
    frame = interactions[["user_id", "item_id", "event_time"]].copy()
    frame["category"] = frame["item_id"].astype(str).map(item_to_category)
    frame = frame.dropna(subset=["category"])
    if frame.empty:
        return pd.DataFrame(columns=["user_id", "period", "category", "share", "rank"])

    frame["period"] = frame["event_time"].dt.to_period(
        {"YE": "Y", "ME": "M", "QE": "Q"}.get(granularity, "Y")
    )
    counts = frame.groupby(["user_id", "period", "category"], observed=True).size()
    counts.name = "count"
    counts = counts.reset_index()
    totals = counts.groupby(["user_id", "period"], observed=True)["count"].transform("sum")
    counts["share"] = counts["count"] / totals
    counts["rank"] = counts.groupby(["user_id", "period"], observed=True)["count"].rank(
        ascending=False, method="first"
    ).astype(int)
    snapshots = counts[counts["rank"] <= top_k].sort_values(["user_id", "period", "rank"])
    logger.info("Preference snapshots: %s rows over %s users",
                f"{len(snapshots):,}", f"{snapshots['user_id'].nunique():,}")
    return snapshots[["user_id", "period", "category", "share", "rank"]].reset_index(drop=True)


def interest_drift(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Total-variation distance between each user's consecutive period mixes.

    0 means the category mix did not move; 1 means it turned over completely.
    """
    records = []
    for user_id, group in snapshots.groupby("user_id", observed=True):
        periods = sorted(group["period"].unique())
        for earlier, later in zip(periods, periods[1:]):
            a = group[group["period"] == earlier].set_index("category")["share"]
            b = group[group["period"] == later].set_index("category")["share"]
            categories = a.index.union(b.index)
            distance = 0.5 * float(
                (a.reindex(categories, fill_value=0.0)
                 - b.reindex(categories, fill_value=0.0)).abs().sum()
            )
            records.append({"user_id": user_id, "from_period": str(earlier),
                            "to_period": str(later), "drift": distance})
    return pd.DataFrame(records)
