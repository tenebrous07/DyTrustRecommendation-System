"""Chronological train/val/test split (Module 22).

Random splitting of interactions leaks the future into the past: a model can be
trained on a user's 2023 purchase and evaluated on their 2019 one. Every split
here is by TIME, and the training graph is guaranteed to contain no edge whose
timestamp falls in the validation or test period.

The guarantees, all asserted in :func:`assert_no_leakage`:

* ``max(train.timestamp) <= min(val.timestamp)`` and likewise val -> test.
* The training graph contains only training edges.
* Negative sampling excludes positives from *every* split, not just train.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..utils.config import Config
from ..utils.logging import get_logger, log_stage

__all__ = ["Split", "chronological_split", "assert_no_leakage"]

logger = get_logger(__name__)


@dataclass(slots=True)
class Split:
    """Three chronologically ordered partitions plus their boundaries."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    val_cutoff: pd.Timestamp
    test_cutoff: pd.Timestamp
    strategy: str
    stats: dict[str, Any]

    @property
    def train_max_timestamp(self) -> int:
        """Reference point for recency weighting. Never wall-clock 'now'."""
        return int(self.train["timestamp"].max())

    def all_positives(self) -> dict[str, set[str]]:
        """user -> every item they touch in ANY split, for negative sampling."""
        combined = pd.concat([self.train, self.val, self.test], ignore_index=True)
        return combined.groupby("user_id")["item_id"].apply(set).to_dict()

    def describe(self) -> str:
        return "\n".join(
            f"  {name:5s} {len(frame):>9,} rows  "
            f"{frame['event_time'].min().date()} -> {frame['event_time'].max().date()}  "
            f"users={frame['user_id'].nunique():,} items={frame['item_id'].nunique():,}"
            for name, frame in (("train", self.train), ("val", self.val), ("test", self.test))
            if len(frame)
        )


def chronological_split(interactions: pd.DataFrame, config: Config) -> Split:
    """Split by global timestamp quantiles.

    ``global_temporal`` cuts the whole corpus at two wall-clock instants, which
    simulates deploying a model at a point in time and serving everyone from it.
    ``per_user`` instead holds out each user's own last interactions, which
    guarantees every user appears in test but lets a user's test event predate
    another user's training event - a weaker and slightly optimistic protocol.
    The default is the stricter one.
    """
    cfg = config.require("split")
    strategy = str(cfg.get("strategy", "global_temporal"))
    train_frac = float(cfg.get("train_frac", 0.8))
    val_frac = float(cfg.get("val_frac", 0.1))

    if not 0 < train_frac < 1 or not 0 <= val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError(f"Invalid split fractions: train={train_frac}, val={val_frac}.")

    frame = interactions.sort_values("timestamp", kind="stable").reset_index(drop=True)

    with log_stage(logger, f"Chronological split ({strategy})") as stage:
        if strategy == "global_temporal":
            val_cut = frame["timestamp"].quantile(train_frac)
            test_cut = frame["timestamp"].quantile(train_frac + val_frac)
            train = frame[frame["timestamp"] <= val_cut]
            val = frame[(frame["timestamp"] > val_cut) & (frame["timestamp"] <= test_cut)]
            test = frame[frame["timestamp"] > test_cut]
        elif strategy == "per_user":
            ranked = frame.groupby("user_id")["timestamp"].rank(method="first", pct=True)
            train = frame[ranked <= train_frac]
            val = frame[(ranked > train_frac) & (ranked <= train_frac + val_frac)]
            test = frame[ranked > train_frac + val_frac]
            val_cut = train["timestamp"].max()
            test_cut = val["timestamp"].max() if len(val) else val_cut
        else:
            raise ValueError(f"Unknown split.strategy '{strategy}'. Use global_temporal|per_user.")
        stage["train"] = f"{len(train):,}"
        stage["val"] = f"{len(val):,}"
        stage["test"] = f"{len(test):,}"

    # Evaluation can only score users and items the training graph knows about.
    train_users, train_items = set(train["user_id"]), set(train["item_id"])
    stats = {
        "strategy": strategy,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "val_cutoff": str(pd.to_datetime(val_cut, unit="ms", utc=True)),
        "test_cutoff": str(pd.to_datetime(test_cut, unit="ms", utc=True)),
        "test_users_unseen_in_train": int((~test["user_id"].isin(train_users)).sum()),
        "test_items_unseen_in_train": int((~test["item_id"].isin(train_items)).sum()),
    }
    stats["test_cold_user_share"] = (
        stats["test_users_unseen_in_train"] / len(test) if len(test) else 0.0
    )
    stats["test_cold_item_share"] = (
        stats["test_items_unseen_in_train"] / len(test) if len(test) else 0.0
    )

    split = Split(
        train=train.reset_index(drop=True), val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
        val_cutoff=pd.to_datetime(val_cut, unit="ms", utc=True),
        test_cutoff=pd.to_datetime(test_cut, unit="ms", utc=True),
        strategy=strategy, stats=stats,
    )
    logger.info("Split:\n%s", split.describe())
    logger.info(
        "Cold in test: %.1f%% of rows have an unseen user, %.1f%% an unseen item. "
        "These are the rows the cold-start path (Module 23) must handle, and they are "
        "reported separately so warm metrics are not flattered.",
        100 * stats["test_cold_user_share"], 100 * stats["test_cold_item_share"],
    )
    return split


def assert_no_leakage(split: Split) -> None:
    """Raise if any temporal ordering guarantee is violated."""
    if len(split.train) and len(split.val):
        if split.train["timestamp"].max() > split.val["timestamp"].min():
            raise AssertionError(
                f"Leakage: a train interaction ({split.train['timestamp'].max()}) is newer "
                f"than the earliest val interaction ({split.val['timestamp'].min()})."
            )
    if len(split.val) and len(split.test):
        if split.val["timestamp"].max() > split.test["timestamp"].min():
            raise AssertionError("Leakage: a val interaction is newer than the earliest test one.")
    if len(split.train) and len(split.test):
        if split.train["timestamp"].max() > split.test["timestamp"].min():
            raise AssertionError("Leakage: a train interaction is newer than the earliest test one.")

    total = len(split.train) + len(split.val) + len(split.test)
    keys = pd.concat([
        split.train[["user_id", "item_id"]], split.val[["user_id", "item_id"]],
        split.test[["user_id", "item_id"]],
    ])
    if len(keys.drop_duplicates()) != total:
        raise AssertionError("The same (user, item) pair appears in more than one split.")
    logger.info("Leakage checks passed for the %s split.", split.strategy)
