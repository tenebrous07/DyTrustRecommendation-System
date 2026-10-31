"""Unified evaluation harness (Module 19).

Every model is scored here, on the same split, with the same exclusions and the
same metric code. That is the whole point: a comparison where each model brings
its own protocol is not a comparison.

Protocol, fixed for all models:

* Rank the **full catalogue** for each user, not a sampled subset of negatives.
  Sampled-negative evaluation is common in the literature and is known to
  reorder model rankings, so it is avoided here.
* Mask every item the user already touched in **training**. Recommending an
  item back to the user who just bought it is not a hit.
* Evaluate only users with at least one held-out relevant item. Users with none
  are counted and reported, not silently scored 0.
* Report warm and cold users **separately**, so a cold-start-heavy split cannot
  quietly drag the headline number down (or up).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.config import Config, resolve_path
from ..utils.logging import get_logger, log_stage
from .metrics import (
    average_precision_at_k,
    coverage,
    hit_rate_at_k,
    intra_list_diversity,
    ndcg_at_k,
    novelty,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = ["Vocabulary", "EvaluationResult", "Evaluator"]

logger = get_logger(__name__)


@dataclass
class Vocabulary:
    """Contiguous index space derived from the TRAINING split only."""

    users: dict[str, int] = field(default_factory=dict)
    items: dict[str, int] = field(default_factory=dict)
    user_ids: list[str] = field(default_factory=list)
    item_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_train(cls, train: pd.DataFrame) -> "Vocabulary":
        user_ids = list(dict.fromkeys(train["user_id"].astype(str)))
        item_ids = list(dict.fromkeys(train["item_id"].astype(str)))
        return cls(
            users={u: i for i, u in enumerate(user_ids)},
            items={p: i for i, p in enumerate(item_ids)},
            user_ids=user_ids, item_ids=item_ids,
        )

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def index(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach ``user_idx`` / ``item_idx``; unseen ids become -1."""
        out = frame.copy()
        out["user_idx"] = out["user_id"].astype(str).map(self.users).fillna(-1).astype(int)
        out["item_idx"] = out["item_id"].astype(str).map(self.items).fillna(-1).astype(int)
        return out


@dataclass
class EvaluationResult:
    """Metrics for one model on one split."""

    model: str
    split: str
    metrics: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "split": self.split, "metrics": self.metrics,
                "diagnostics": self.diagnostics, "notes": self.notes}

    def headline(self, primary: str = "ndcg@10") -> str:
        return (f"{self.model:26s} {primary}={self.metrics.get(primary, float('nan')):.5f}  "
                f"recall@10={self.metrics.get('recall@10', float('nan')):.5f}  "
                f"coverage@10={self.metrics.get('coverage@10', float('nan')):.4f}")


class Evaluator:
    """Scores recommenders against a held-out split."""

    def __init__(
        self,
        train: pd.DataFrame,
        evaluation_split: pd.DataFrame,
        vocabulary: Vocabulary,
        k_values: list[int] | None = None,
        item_categories: np.ndarray | None = None,
        batch_size: int = 512,
        split_name: str = "test",
    ) -> None:
        self.vocab = vocabulary
        self.k_values = sorted(k_values or [5, 10, 20])
        self.max_k = max(self.k_values)
        self.batch_size = batch_size
        self.split_name = split_name

        indexed_train = vocabulary.index(train)
        self.train_positives: dict[int, np.ndarray] = {
            int(u): group["item_idx"].to_numpy()
            for u, group in indexed_train[indexed_train["user_idx"] >= 0].groupby("user_idx")
        }
        self.item_popularity = np.bincount(
            indexed_train.loc[indexed_train["item_idx"] >= 0, "item_idx"],
            minlength=vocabulary.n_items,
        ).astype(np.float64)

        indexed_eval = vocabulary.index(evaluation_split)
        self.n_eval_rows = len(indexed_eval)
        # A held-out row is only scoreable if BOTH its user and item were seen
        # in training. Cold rows are counted, then reported separately.
        warm = indexed_eval[(indexed_eval["user_idx"] >= 0) & (indexed_eval["item_idx"] >= 0)]
        self.cold_user_rows = int((indexed_eval["user_idx"] < 0).sum())
        self.cold_item_rows = int((indexed_eval["item_idx"] < 0).sum())

        self.ground_truth: dict[int, set[int]] = {
            int(u): set(group["item_idx"].tolist())
            for u, group in warm.groupby("user_idx")
        }
        self.eval_users = np.array(sorted(self.ground_truth), dtype=np.int64)

        self.item_categories = (
            item_categories if item_categories is not None
            else np.zeros(vocabulary.n_items, dtype=np.int64)
        )
        logger.info(
            "Evaluator ready on '%s': %s scoreable users, %s of %s rows unusable "
            "(%s cold user, %s cold item)",
            split_name, f"{len(self.eval_users):,}",
            f"{self.cold_user_rows + self.cold_item_rows:,}", f"{self.n_eval_rows:,}",
            f"{self.cold_user_rows:,}", f"{self.cold_item_rows:,}",
        )

    def _ranked_batches(self, model, depth: int, model_name: str = "model"):
        """Yield ``(users, ranked)`` where ``ranked[r]`` is user r's top-``depth`` items."""
        for start in range(0, len(self.eval_users), self.batch_size):
            users = self.eval_users[start:start + self.batch_size]
            scores = np.asarray(model.score_users(users), dtype=np.float32)
            if scores.shape != (len(users), self.vocab.n_items):
                raise ValueError(
                    f"{model_name}.score_users returned {scores.shape}, expected "
                    f"{(len(users), self.vocab.n_items)}."
                )

            # Mask training positives so a model cannot score by recall of
            # what the user already has.
            for row, user in enumerate(users):
                seen = self.train_positives.get(int(user))
                if seen is not None and len(seen):
                    scores[row, seen] = -np.inf

            top = np.argpartition(-scores, depth - 1, axis=1)[:, :depth]
            ordering = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
            yield users, np.take_along_axis(top, ordering, axis=1)

    def per_user_ndcg(self, model, k: int = 10) -> np.ndarray:
        """NDCG@k for every scoreable user, ordered like ``eval_users``.

        The per-user array is what paired significance tests need; a mean on its
        own cannot say whether two models differ by more than sampling noise.
        """
        values = []
        for users, ranked_batch in self._ranked_batches(model, k):
            for row, user in enumerate(users):
                values.append(ndcg_at_k(ranked_batch[row], self.ground_truth[int(user)], k))
        return np.asarray(values, dtype=np.float64)

    def top_lists(self, model, k: int = 10) -> list[np.ndarray]:
        """Top-k item indices for every scoreable user, ordered like ``eval_users``."""
        lists: list[np.ndarray] = []
        for _, ranked_batch in self._ranked_batches(model, k):
            lists.extend(ranked_batch)
        return lists

    def ndcg(self, model, k: int = 10) -> float:
        """Mean NDCG@k only: cheap enough to call every few epochs for early stopping."""
        return float(self.per_user_ndcg(model, k).mean())

    def evaluate(self, model, name: str | None = None) -> EvaluationResult:
        """Rank the catalogue for every scoreable user and aggregate metrics."""
        model_name = name or getattr(model, "name", model.__class__.__name__)
        accumulators: dict[str, list[float]] = {}
        top_lists: list[np.ndarray] = []

        def add(metric: str, value: float) -> None:
            accumulators.setdefault(metric, []).append(value)

        with log_stage(logger, f"Evaluate {model_name} on {self.split_name}") as stage:
            for users, ranked_batch in self._ranked_batches(model, self.max_k, model_name):
                for row, user in enumerate(users):
                    ranked = ranked_batch[row]
                    relevant = self.ground_truth[int(user)]
                    top_lists.append(ranked)
                    for k in self.k_values:
                        add(f"precision@{k}", precision_at_k(ranked, relevant, k))
                        add(f"recall@{k}", recall_at_k(ranked, relevant, k))
                        add(f"ndcg@{k}", ndcg_at_k(ranked, relevant, k))
                        add(f"map@{k}", average_precision_at_k(ranked, relevant, k))
                        add(f"hit_rate@{k}", hit_rate_at_k(ranked, relevant, k))
                    add("mrr", reciprocal_rank(ranked, relevant))
            stage["users"] = f"{len(self.eval_users):,}"

        metrics = {name_: float(np.mean(values)) for name_, values in accumulators.items()}
        for k in self.k_values:
            metrics[f"coverage@{k}"] = coverage(top_lists, self.vocab.n_items, k)
            metrics[f"diversity@{k}"] = intra_list_diversity(top_lists, self.item_categories, k)
            metrics[f"novelty@{k}"] = novelty(top_lists, self.item_popularity, k)

        result = EvaluationResult(
            model=model_name, split=self.split_name, metrics=metrics,
            diagnostics={
                "n_users_evaluated": int(len(self.eval_users)),
                "n_eval_rows": self.n_eval_rows,
                "cold_user_rows_excluded": self.cold_user_rows,
                "cold_item_rows_excluded": self.cold_item_rows,
                "catalogue_size": self.vocab.n_items,
                **(model.describe() if hasattr(model, "describe") else {}),
            },
            notes=[
                "Full-catalogue ranking; no sampled-negative shortcut.",
                "Training positives are masked before ranking.",
                f"{self.cold_user_rows:,} rows with an unseen user and "
                f"{self.cold_item_rows:,} with an unseen item were excluded from these "
                f"metrics and are the cold-start path's responsibility (Module 23).",
                "Coverage and novelty are relative to the post-k-core catalogue, "
                "not the full Amazon catalogue.",
            ],
        )
        logger.info("  %s", result.headline())
        return result


def save_results(results: list[EvaluationResult], path: str | Path) -> Path:
    """Write a list of results as JSON plus a comparison table."""
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([r.to_dict() for r in results], indent=2), encoding="utf-8")
    return out
