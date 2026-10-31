"""Cold-start handling (Module 23).

No claim is made that cold start is *solved*. These are documented fallbacks
whose quality is reported separately from warm-user metrics, so the averages
are never flattered by mixing the two.

Phase 2 found the training graph has a single connected component, so a cold
user here is genuinely a *new* user rather than a disconnected one - which is
why the fallbacks below are content- and popularity-based rather than
graph-repair strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

__all__ = ["ColdStartHandler"]

logger = get_logger(__name__)


@dataclass
class ColdStartHandler:
    """Popularity and content fallbacks for unseen users and items."""

    vocabulary: Any
    item_popularity: np.ndarray
    item_categories: np.ndarray | None = None
    item_embeddings: np.ndarray | None = None
    min_interactions_for_personalised: int = 3

    def is_cold_user(self, user_id: str, n_interactions: int = 0) -> bool:
        return (str(user_id) not in self.vocabulary.users
                or n_interactions < self.min_interactions_for_personalised)

    def recommend_new_user(
        self, k: int = 10, preferred_category: str | None = None,
        category_lookup: dict[str, int] | None = None,
    ) -> list[tuple[str, float, str]]:
        """Popularity, optionally restricted to a stated category preference.

        Returns ``(item_id, score, reason)`` so the dashboard can be explicit
        that this is a fallback rather than a personalised prediction.
        """
        scores = self.item_popularity.astype(np.float64).copy()
        reason = "no interaction history; ranked by training popularity"

        if (preferred_category is not None and self.item_categories is not None
                and category_lookup is not None):
            code = category_lookup.get(preferred_category)
            if code is not None:
                mask = self.item_categories == code
                if mask.any():
                    scores = np.where(mask, scores, -np.inf)
                    reason = (f"no interaction history; most popular within the stated "
                              f"preference '{preferred_category}'")

        k = min(k, int(np.isfinite(scores).sum()))
        top = np.argsort(-scores)[:k]
        return [(self.vocabulary.item_ids[int(i)], float(scores[i]), reason) for i in top]

    def recommend_for_new_item(self, item_embedding: np.ndarray, k: int = 10
                               ) -> list[tuple[str, float, str]]:
        """Nearest catalogue items by content similarity (SBERT).

        Lets a product with no interactions inherit an audience from items whose
        descriptions it resembles.
        """
        if self.item_embeddings is None:
            raise RuntimeError(
                "No item content embeddings available; new-item cold start needs SBERT "
                "vectors over product titles and descriptions."
            )
        query = item_embedding / (np.linalg.norm(item_embedding) + 1e-12)
        matrix = self.item_embeddings
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
        similarity = (matrix / norms) @ query
        top = np.argsort(-similarity)[:k]
        return [(self.vocabulary.item_ids[int(i)], float(similarity[i]),
                 "content similarity to the new item's description") for i in top]

    def report(self, cold_users: int, warm_users: int) -> dict[str, Any]:
        total = cold_users + warm_users
        return {
            "cold_users": cold_users, "warm_users": warm_users,
            "cold_share": cold_users / total if total else 0.0,
            "note": ("Cold-start results are reported separately from warm-user metrics. "
                     "Cold start is NOT claimed to be solved; these are documented "
                     "fallbacks whose quality is measured on its own terms."),
        }
