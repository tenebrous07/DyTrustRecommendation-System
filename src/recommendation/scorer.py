"""Scoring and ranking (Module 15)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..utils.logging import get_logger

__all__ = ["Recommendation", "RecommendationScorer"]

logger = get_logger(__name__)


@dataclass
class Recommendation:
    """One ranked item, with everything the dashboard needs to justify it."""

    item_id: str
    item_index: int
    rank: int
    score: float
    confidence: float | None = None
    explanation: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    trust: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"item_id": self.item_id, "rank": self.rank, "score": round(self.score, 6),
                "confidence": None if self.confidence is None else round(self.confidence, 4),
                "trust": None if self.trust is None else round(self.trust, 4),
                "explanation": self.explanation, "evidence": self.evidence}


class RecommendationScorer:
    """Turns a model's score matrix into ranked, id-keyed recommendations."""

    def __init__(self, model, vocabulary, train_positives: dict[int, np.ndarray]) -> None:
        self.model = model
        self.vocab = vocabulary
        self.train_positives = train_positives

    def raw_scores(self, user_id: str) -> np.ndarray:
        """Full-catalogue scores for one user, with already-seen items masked."""
        index = self.vocab.users.get(str(user_id))
        if index is None:
            raise KeyError(
                f"User '{user_id}' is not in the training vocabulary. This is a "
                f"cold-start user; route to src.recommendation.cold_start instead."
            )
        scores = np.asarray(self.model.score_users(np.array([index])), dtype=np.float64)[0]
        seen = self.train_positives.get(index)
        if seen is not None and len(seen):
            scores[seen] = -np.inf
        return scores

    def top_k(self, user_id: str, k: int = 10) -> list[Recommendation]:
        scores = self.raw_scores(user_id)
        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            Recommendation(item_id=self.vocab.item_ids[int(i)], item_index=int(i),
                           rank=rank, score=float(scores[i]))
            for rank, i in enumerate(top, start=1)
        ]

    def rank_of(self, user_id: str, item_id: str) -> tuple[int, float]:
        """Where a specific item sits in this user's full ranking.

        Needed by the counterfactual module, which has to know how far an item
        is from the Top-K boundary before it can say what would move it there.
        """
        scores = self.raw_scores(user_id)
        item_index = self.vocab.items.get(str(item_id))
        if item_index is None:
            raise KeyError(f"Item '{item_id}' is not in the training vocabulary.")
        score = float(scores[item_index])
        if not np.isfinite(score):
            raise ValueError(
                f"Item '{item_id}' is masked for user '{user_id}' - they already "
                f"interacted with it in training, so it has no rank."
            )
        rank = int((scores > score).sum()) + 1
        return rank, score
