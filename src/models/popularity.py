"""Popularity baseline (Module 28, baseline 1).

Not a strawman on this data. Phase 2 measured a popularity Gini of 0.521 with
the top 20% of items holding 59.5% of interactions, so recommending the head is
a genuinely strong strategy here. A personalised model that cannot beat it is
not learning personalisation, which is precisely what this baseline is for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .base import BaseRecommender

__all__ = ["PopularityRecommender"]

logger = get_logger(__name__)


class PopularityRecommender(BaseRecommender):
    """Ranks every user identically, by global training popularity."""

    name = "popularity"

    def fit(self, train: pd.DataFrame, **kwargs) -> "PopularityRecommender":
        counts = np.zeros(self.n_items, dtype=np.float32)
        observed = train["item_idx"].value_counts()
        counts[observed.index.to_numpy()] = observed.to_numpy(dtype=np.float32)
        self.scores = counts
        self.is_fitted = True
        logger.info("Popularity fitted: %s items with at least one interaction",
                    f"{int((counts > 0).sum()):,}")
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("fit() must be called before score_users().")
        return np.tile(self.scores, (len(user_indices), 1))
