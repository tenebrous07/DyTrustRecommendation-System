"""Seed ensemble of independently trained embedding recommenders.

A single HGT run is noisy: across seeds the test NDCG@10 moves by roughly 10%
of its value, because training is one full-batch optimiser step per epoch from a
random start. Averaging the *scores* of several runs removes much of that
variance at no cost at training time beyond the extra runs.

Scores are averaged, not embeddings. Two independently trained models place
users and items in unrelated coordinate systems, so their embeddings cannot be
combined, but their user-item scores are directly comparable once each member's
scores are standardised per user.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseRecommender

__all__ = ["EnsembleRecommender"]


class EnsembleRecommender(BaseRecommender):
    """Mean of per-user z-scored dot-product scores from several members."""

    name = "ensemble"

    def __init__(self, n_users: int, n_items: int,
                 members: list[tuple[np.ndarray, np.ndarray]], **kwargs) -> None:
        if not members:
            raise ValueError("An ensemble needs at least one member.")
        super().__init__(n_users, n_items, members=len(members), **kwargs)
        self.members = members
        self.is_fitted = True

    def fit(self, train: pd.DataFrame, **kwargs) -> "EnsembleRecommender":
        return self   # members arrive already trained

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        users = np.asarray(user_indices)
        total = np.zeros((len(users), self.n_items), dtype=np.float32)
        for user_emb, item_emb in self.members:
            scores = user_emb[users] @ item_emb.T
            spread = scores.std(axis=1, keepdims=True)
            total += (scores - scores.mean(axis=1, keepdims=True)) / np.where(spread > 0, spread, 1.0)
        return total / len(self.members)
