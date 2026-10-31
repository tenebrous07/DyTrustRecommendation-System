"""Per-review structured features (Module 5, review half).

These are the non-semantic inputs to the fusion layer (Module 8): the numbers
that describe *how* a review was written and received, alongside the SBERT
vector that describes what it says.

Two of them are leakage-sensitive and therefore live behind a fit/transform
split rather than being computed inline:

* ``rating_centred`` subtracts the item's mean rating. Computed over all data,
  that mean is informed by future reviews of the same item.
* ``recency_weight`` needs a reference instant. Using wall-clock "now" would
  make the weighting depend on when the script happens to run, and would place
  the reference inside the test period. The reference is
  ``max(train.timestamp)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..utils.logging import get_logger, log_stage

__all__ = ["REVIEW_FEATURE_COLUMNS", "ReviewFeatureBuilder"]

logger = get_logger(__name__)

REVIEW_FEATURE_COLUMNS = (
    "review_length_log", "word_count_log", "helpful_votes_log",
    "verified_purchase_f", "sentiment_compound", "rating_centred", "recency_weight",
)

_SECONDS_PER_YEAR = 365.25 * 24 * 3600


@dataclass
class ReviewFeatureBuilder:
    """Builds review-level features with train-only statistics.

    Args:
        lambda_per_year: Exponential decay rate. ``recency_weight =
            exp(-lambda * age_years)``, so at lambda 0.5 a two-year-old review
            carries e^-1 ~ 0.37 of the weight of a brand-new one.
        min_weight: Floor, so an ancient interaction contributes a little rather
            than exactly nothing. A weight of 0 would silently delete an edge.
    """

    lambda_per_year: float = 0.5
    min_weight: float = 0.01

    item_means: dict[str, float] = field(default_factory=dict)
    global_mean: float = 0.0
    reference_timestamp: int = 0
    fitted_scope: str = "not fitted"
    _fitted: bool = False

    def fit(self, train: pd.DataFrame, scope: str = "TRAIN SPLIT ONLY") -> "ReviewFeatureBuilder":
        """Learn item means and the recency reference from training rows only."""
        if "rating" in train.columns:
            means = train.groupby("item_id")["rating"].mean()
            self.item_means = {str(k): float(v) for k, v in means.items()}
            self.global_mean = float(train["rating"].mean())
        self.reference_timestamp = int(train["timestamp"].max())
        self.fitted_scope = scope
        self._fitted = True
        logger.info(
            "Review features fitted on %s: %s item means, reference instant %s",
            scope, f"{len(self.item_means):,}",
            pd.to_datetime(self.reference_timestamp, unit="ms", utc=True).date(),
        )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Compute review features for any partition using the fitted statistics."""
        if not self._fitted:
            raise RuntimeError("ReviewFeatureBuilder.transform called before fit().")

        with log_stage(logger, "Review features") as stage:
            out = pd.DataFrame(index=frame.index)

            if "review_text" in frame.columns:
                text = frame["review_text"].fillna("").astype(str)
                out["review_length_log"] = np.log1p(text.str.len()).astype("float32")
                out["word_count_log"] = np.log1p(text.str.count(r"\S+")).astype("float32")
            else:
                out["review_length_log"] = np.float32(0.0)
                out["word_count_log"] = np.float32(0.0)

            if "helpful_votes" in frame.columns:
                votes = pd.to_numeric(frame["helpful_votes"], errors="coerce").fillna(0)
                out["helpful_votes_log"] = np.log1p(votes.clip(lower=0)).astype("float32")
            else:
                out["helpful_votes_log"] = np.float32(0.0)

            if "verified_purchase" in frame.columns:
                out["verified_purchase_f"] = (
                    frame["verified_purchase"].fillna(False).astype("int8").astype("float32")
                )
            else:
                out["verified_purchase_f"] = np.float32(0.0)

            out["sentiment_compound"] = (
                frame["sentiment_compound"].astype("float32")
                if "sentiment_compound" in frame.columns else np.float32(0.0)
            )

            # Items unseen in training fall back to the global training mean,
            # never to the item's own (future) mean.
            if "rating" in frame.columns:
                item_mean = frame["item_id"].astype(str).map(self.item_means)
                out["rating_centred"] = (
                    frame["rating"].astype("float64") - item_mean.fillna(self.global_mean)
                ).astype("float32")
            else:
                out["rating_centred"] = np.float32(0.0)

            out["recency_weight"] = self.recency_weight(frame["timestamp"]).astype("float32")
            stage["rows"] = f"{len(out):,}"
            stage["features"] = len(out.columns)
        return out

    def recency_weight(self, timestamps: pd.Series) -> pd.Series:
        """exp(-lambda * age_years), floored, relative to the training reference.

        Interactions after the reference (validation and test rows) would give a
        negative age and therefore a weight above 1. They are clamped to age 0,
        so no row is ever *up*-weighted for being in the future.
        """
        age_seconds = (self.reference_timestamp - pd.to_numeric(timestamps)) / 1000.0
        age_years = (age_seconds / _SECONDS_PER_YEAR).clip(lower=0.0)
        weights = np.exp(-self.lambda_per_year * age_years)
        return weights.clip(lower=self.min_weight, upper=1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fitted_scope": self.fitted_scope,
            "lambda_per_year": self.lambda_per_year,
            "min_weight": self.min_weight,
            "n_item_means": len(self.item_means),
            "global_mean_rating": self.global_mean,
            "reference_timestamp": self.reference_timestamp,
        }
