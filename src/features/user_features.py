"""Per-user behavioural profiles (Module 5, aggregation half).

Built in Phase 2 rather than Phase 3 because both the EDA (Module 3) and the
outlier detector (Module 4) need exactly these aggregates, and computing them
twice in two modules would be the kind of duplication that drifts apart.

What lives here: **pure aggregation**. A DataFrame of interactions goes in, a
DataFrame indexed by ``user_id`` comes out. There is no fitting, no scaling and
no encoding, so there is nothing here that can leak - every column is derived
from a single user's own rows.

What does *not* live here: the leakage-sensitive transform layer (scalers,
target encodings, train-only statistics). Phase 3 adds that on top, with an
explicit fit/transform split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.logging import get_logger, log_stage

__all__ = ["build_user_profiles", "USER_PROFILE_COLUMNS"]

logger = get_logger(__name__)

USER_PROFILE_COLUMNS: tuple[str, ...] = (
    "n_interactions", "n_distinct_items",
    "mean_rating", "rating_std", "median_rating", "rating_entropy",
    "frac_five_star", "frac_one_star",
    "first_seen", "last_seen", "span_days", "active_days", "reviews_per_active_day",
    "mean_review_length", "median_review_length",
    "total_helpful_votes", "mean_helpful_votes",
    "frac_verified", "frac_flagged", "frac_duplicate_text",
    "median_gap_hours", "min_gap_hours",
)

_SECONDS_PER_HOUR = 3600.0


def _rating_entropy(series: pd.Series) -> float:
    """Shannon entropy (bits) of a user's rating distribution.

    Separates two behaviours that a standard deviation conflates: a user who
    always gives 5 stars (entropy 0) and a user who alternates 1 and 5
    (entropy 1.0 but a large standard deviation). Only the first is the
    signature the trust component is interested in.
    """
    counts = series.value_counts(normalize=True)
    if len(counts) <= 1:
        return 0.0
    return float(-(counts * np.log2(counts)).sum())


def build_user_profiles(interactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate interactions into one row per user.

    Args:
        interactions: Cleaned interactions with at least ``user_id``,
            ``item_id``, ``rating`` and ``event_time``. Optional columns
            (``review_text``, ``helpful_votes``, ``verified_purchase``,
            ``outlier_flag``) are used when present and skipped when absent.

    Returns:
        DataFrame indexed by ``user_id``. Columns that could not be computed
        because their source column was missing are simply absent - they are
        never filled with a placeholder that would read as a real measurement.
    """
    if interactions.empty:
        raise ValueError("Cannot build user profiles from an empty frame.")

    with log_stage(logger, "Build user profiles") as stage:
        frame = interactions.sort_values(["user_id", "event_time"], kind="stable")
        grouped = frame.groupby("user_id", observed=True)

        profiles = pd.DataFrame({
            "n_interactions": grouped.size(),
            "n_distinct_items": grouped["item_id"].nunique(),
        })

        # -- rating behaviour ------------------------------------------------
        if "rating" in frame.columns:
            ratings = grouped["rating"]
            profiles["mean_rating"] = ratings.mean().astype("float64")
            profiles["rating_std"] = ratings.std().fillna(0.0).astype("float64")
            profiles["median_rating"] = ratings.median().astype("float64")
            profiles["rating_entropy"] = ratings.apply(_rating_entropy)
            profiles["frac_five_star"] = grouped["rating"].apply(lambda s: float((s == 5).mean()))
            profiles["frac_one_star"] = grouped["rating"].apply(lambda s: float((s == 1).mean()))

        # -- temporal behaviour ----------------------------------------------
        if "event_time" in frame.columns:
            times = grouped["event_time"]
            profiles["first_seen"] = times.min()
            profiles["last_seen"] = times.max()
            span = (profiles["last_seen"] - profiles["first_seen"]).dt.total_seconds() / 86400.0
            profiles["span_days"] = span.astype("float64")
            profiles["active_days"] = grouped["event_time"].apply(
                lambda s: float(s.dt.normalize().nunique())
            )
            # Rate over days the user was actually active, not over the whole
            # span: a user with two reviews eight years apart is not "inactive",
            # they are two-thirds of a burst with a long gap, and dividing by the
            # span would report a misleadingly tiny rate.
            profiles["reviews_per_active_day"] = (
                profiles["n_interactions"] / profiles["active_days"].clip(lower=1.0)
            )

            gaps = frame.groupby("user_id", observed=True)["event_time"].diff()
            gap_hours = gaps.dt.total_seconds() / _SECONDS_PER_HOUR
            gap_frame = pd.DataFrame({"user_id": frame["user_id"].to_numpy(),
                                      "gap_hours": gap_hours.to_numpy()})
            gap_stats = gap_frame.dropna().groupby("user_id", observed=True)["gap_hours"]
            profiles["median_gap_hours"] = gap_stats.median()
            profiles["min_gap_hours"] = gap_stats.min()

        # -- text behaviour --------------------------------------------------
        if "review_text" in frame.columns:
            lengths = frame["review_text"].fillna("").str.len()
            length_frame = pd.DataFrame({"user_id": frame["user_id"].to_numpy(),
                                         "length": lengths.to_numpy()})
            by_user = length_frame.groupby("user_id", observed=True)["length"]
            profiles["mean_review_length"] = by_user.mean().astype("float64")
            profiles["median_review_length"] = by_user.median().astype("float64")

            # Share of a user's own reviews whose text they reused elsewhere.
            # Computed directly rather than read off the cleaner's outlier flag,
            # so this module does not silently depend on cleaning policy.
            text = frame["review_text"].fillna("")
            substantive = text.str.len() >= 20
            dup_frame = pd.DataFrame({
                "user_id": frame["user_id"].to_numpy(),
                "text": text.where(substantive, other=pd.NA).to_numpy(),
            })
            def _dup_share(group: pd.Series) -> float:
                present = group.dropna()
                if present.empty:
                    return 0.0
                repeated = present.duplicated(keep=False).sum()
                return float(repeated / len(group))
            profiles["frac_duplicate_text"] = dup_frame.groupby(
                "user_id", observed=True
            )["text"].apply(_dup_share)

        # -- engagement and quality signals -----------------------------------
        if "helpful_votes" in frame.columns:
            votes = grouped["helpful_votes"]
            profiles["total_helpful_votes"] = votes.sum().astype("float64")
            profiles["mean_helpful_votes"] = votes.mean().astype("float64")

        if "verified_purchase" in frame.columns:
            profiles["frac_verified"] = grouped["verified_purchase"].apply(
                lambda s: float(s.fillna(False).mean())
            )

        if "outlier_flag" in frame.columns:
            profiles["frac_flagged"] = grouped["outlier_flag"].apply(
                lambda s: float(s.fillna(False).mean())
            )

        stage["users"] = f"{len(profiles):,}"
        stage["features"] = len(profiles.columns)

    missing = [c for c in USER_PROFILE_COLUMNS if c not in profiles.columns]
    if missing:
        logger.info("User profile columns unavailable in this dataset: %s", missing)
    return profiles
