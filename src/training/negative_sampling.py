"""Negative sampling (Module 21).

The leakage rule that matters here: a sampled "negative" must not be an item
the user interacts with in **any** split. Excluding only training positives
would place validation and test items in the negative pool, teaching the model
to rank exactly the items it is about to be evaluated on *below* the noise -
which inflates the reported metrics without improving the model.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ..utils.logging import get_logger

__all__ = ["NegativeSampler"]

logger = get_logger(__name__)


class NegativeSampler:
    """Uniform or popularity-weighted sampling with all-split exclusion."""

    def __init__(
        self,
        n_items: int,
        positives_by_user: dict[int, set[int]],
        strategy: str = "uniform",
        item_popularity: np.ndarray | None = None,
        max_attempts: int = 20,
        seed: int = 42,
    ) -> None:
        if strategy not in {"uniform", "popularity"}:
            raise ValueError(f"strategy must be uniform|popularity, got '{strategy}'.")
        self.n_items = n_items
        self.strategy = strategy
        self.max_attempts = max_attempts
        self.rng = np.random.default_rng(seed)

        rows, cols = [], []
        for user, items in positives_by_user.items():
            rows.extend([user] * len(items))
            cols.extend(items)
        n_users = (max(positives_by_user) + 1) if positives_by_user else 1
        self.positives = sparse.csr_matrix(
            (np.ones(len(rows), dtype=bool), (rows, cols)),
            shape=(n_users, n_items),
        )
        self.positives_lil = self.positives.tolil()

        if strategy == "popularity":
            weights = np.clip(np.asarray(item_popularity, dtype=np.float64), 1e-9, None)
            self.probabilities = weights / weights.sum()
        else:
            self.probabilities = None

        total = int(self.positives.sum())
        logger.info(
            "Negative sampler: %s items, %s excluded positives across ALL splits "
            "(%s strategy)", f"{n_items:,}", f"{total:,}", strategy,
        )

    def sample(self, users: np.ndarray, n_negatives: int = 1) -> np.ndarray:
        """Draw negatives for each user.

        Returns:
            ``(len(users), n_negatives)`` item indices.
        """
        size = (len(users), n_negatives)
        if self.probabilities is None:
            candidates = self.rng.integers(0, self.n_items, size=size)
        else:
            candidates = self.rng.choice(self.n_items, size=size, p=self.probabilities)

        # Rejection-resample only the collisions, so the common case costs one
        # vectorised draw rather than a Python loop over every pair.
        for _ in range(self.max_attempts):
            collisions = self._is_positive(users, candidates)
            if not collisions.any():
                break
            n_bad = int(collisions.sum())
            if self.probabilities is None:
                replacement = self.rng.integers(0, self.n_items, size=n_bad)
            else:
                replacement = self.rng.choice(self.n_items, size=n_bad, p=self.probabilities)
            candidates[collisions] = replacement
        else:
            remaining = int(self._is_positive(users, candidates).sum())
            if remaining:
                # Never silently ship a false negative; say how many survived.
                logger.warning(
                    "%s negative(s) still collide with a known positive after %d attempts. "
                    "This happens for users who have touched most of the catalogue.",
                    f"{remaining:,}", self.max_attempts,
                )
        return candidates

    def _is_positive(self, users: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        repeated = np.repeat(users[:, None], candidates.shape[1], axis=1)
        flags = np.asarray(
            self.positives[repeated.ravel(), candidates.ravel()]
        ).ravel()
        return flags.reshape(candidates.shape)
