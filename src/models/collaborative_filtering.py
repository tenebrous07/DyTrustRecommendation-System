"""Item-based collaborative filtering (Module 28, baseline 2).

Item-kNN over cosine similarity of the implicit user-item matrix. Item-based
rather than user-based because this corpus has 15,704 training users against
6,619 items, so the item-item similarity matrix is both smaller and estimated
from more observations per cell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from ..utils.logging import get_logger, log_stage
from .base import BaseRecommender

__all__ = ["ItemKNNRecommender"]

logger = get_logger(__name__)


class ItemKNNRecommender(BaseRecommender):
    """score(u, i) = sum over the k nearest j of sim(i, j) * 1[u touched j]."""

    name = "item_knn_cf"

    def __init__(self, n_users: int, n_items: int, k: int = 50, **kwargs) -> None:
        super().__init__(n_users, n_items, k=k, **kwargs)
        self.k = k

    def fit(self, train: pd.DataFrame, **kwargs) -> "ItemKNNRecommender":
        with log_stage(logger, f"Fit item-kNN (k={self.k})"):
            self.matrix = sparse.csr_matrix(
                (np.ones(len(train), dtype=np.float32),
                 (train["user_idx"].to_numpy(), train["item_idx"].to_numpy())),
                shape=(self.n_users, self.n_items),
            )
            counts = np.asarray(self.matrix.sum(axis=0)).ravel()
            norms = np.sqrt(counts)
            norms[norms == 0] = 1.0

            scaled = self.matrix @ sparse.diags(1.0 / norms)
            similarity = np.asarray((scaled.T @ scaled).todense(), dtype=np.float32)
            np.fill_diagonal(similarity, 0.0)   # an item must not recommend itself

            # Keep the top-k neighbours per item. The dense remainder is mostly
            # noise and drags the scores back toward raw popularity.
            if 0 < self.k < self.n_items:
                cut = np.partition(similarity, -self.k, axis=1)[:, -self.k][:, None]
                similarity[similarity < cut] = 0.0
            self.similarity = similarity
        self.is_fitted = True
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("fit() must be called before score_users().")
        profiles = np.asarray(self.matrix[user_indices].todense(), dtype=np.float32)
        return profiles @ self.similarity
