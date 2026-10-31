"""Common recommender interface.

Every model - baselines and DyTrustRec alike - is evaluated through this one
interface, so the comparison in Module 28 is genuinely like-for-like rather
than each model being scored by whichever protocol suits it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["BaseRecommender"]


class BaseRecommender(ABC):
    """Scores every item for a batch of users.

    Implementations must be deterministic given a seed, and must never consult
    validation or test interactions during :meth:`fit`.
    """

    name: str = "base"

    def __init__(self, n_users: int, n_items: int, **kwargs: Any) -> None:
        self.n_users = n_users
        self.n_items = n_items
        self.params = kwargs
        self.is_fitted = False

    @abstractmethod
    def fit(self, train: pd.DataFrame, **kwargs: Any) -> "BaseRecommender":
        """Fit on training interactions carrying integer ``user_idx``/``item_idx``."""

    @abstractmethod
    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        """Return a dense ``(len(user_indices), n_items)`` score matrix."""

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name, "n_users": self.n_users, "n_items": self.n_items,
            **{k: v for k, v in self.params.items()
               if isinstance(v, (int, float, str, bool))},
        }
