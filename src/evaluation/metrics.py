"""Ranking and beyond-accuracy metrics (Module 19).

All accuracy metrics take a ranked list of item indices per user and the set of
that user's held-out relevant items.

Conventions fixed once, so every model is scored on identical terms:

* K is a cut-off; lists shorter than K are not padded.
* A user with no held-out relevant item is excluded from the average rather
  than scored 0. Including them would penalise every model in proportion to how
  many such users the split produced, which is a property of the split, not of
  the model.
* NDCG uses binary relevance, with the ideal DCG computed over
  ``min(K, |relevant|)``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "precision_at_k", "recall_at_k", "ndcg_at_k", "average_precision_at_k",
    "reciprocal_rank", "hit_rate_at_k", "coverage", "intra_list_diversity", "novelty",
]


def precision_at_k(ranked: np.ndarray, relevant: set[int], k: int) -> float:
    """Share of the top-K that is relevant."""
    if k <= 0:
        return 0.0
    return float(sum(1 for item in ranked[:k] if item in relevant) / k)


def recall_at_k(ranked: np.ndarray, relevant: set[int], k: int) -> float:
    """Share of relevant items recovered in the top-K."""
    if not relevant:
        return 0.0
    return float(sum(1 for item in ranked[:k] if item in relevant) / len(relevant))


def ndcg_at_k(ranked: np.ndarray, relevant: set[int], k: int) -> float:
    """Normalised discounted cumulative gain, binary relevance."""
    if not relevant:
        return 0.0
    top = ranked[:k]
    gains = np.array([1.0 if item in relevant else 0.0 for item in top])
    discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
    dcg = float((gains * discounts).sum())
    ideal_n = min(k, len(relevant))
    idcg = float((1.0 / np.log2(np.arange(2, ideal_n + 2))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(ranked: np.ndarray, relevant: set[int], k: int) -> float:
    """Mean of precision@i over positions i<=K holding a relevant item."""
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for position, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            hits += 1
            total += hits / position
    return float(total / min(len(relevant), k))


def reciprocal_rank(ranked: np.ndarray, relevant: set[int]) -> float:
    """1 / rank of the first relevant item, or 0 if none appears."""
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def hit_rate_at_k(ranked: np.ndarray, relevant: set[int], k: int) -> float:
    """1 if any relevant item appears in the top-K."""
    return float(any(item in relevant for item in ranked[:k]))


def coverage(recommended: list[np.ndarray], n_items: int, k: int) -> float:
    """Share of the catalogue appearing in at least one user's top-K.

    Computed over the post-k-core catalogue, not the full Amazon catalogue.
    """
    if n_items <= 0:
        return 0.0
    seen: set[int] = set()
    for ranked in recommended:
        seen.update(int(i) for i in ranked[:k])
    return float(len(seen) / n_items)


def intra_list_diversity(
    recommended: list[np.ndarray], item_categories: np.ndarray, k: int
) -> float:
    """Mean share of distinct categories within each user's top-K.

    An interpretable proxy for intra-list dissimilarity that needs no embedding
    space, so it stays comparable across models whose embeddings are not
    mutually comparable.
    """
    scores = []
    for ranked in recommended:
        top = ranked[:k]
        if len(top) == 0:
            continue
        scores.append(len(set(item_categories[top].tolist())) / len(top))
    return float(np.mean(scores)) if scores else 0.0


def novelty(recommended: list[np.ndarray], item_popularity: np.ndarray, k: int) -> float:
    """Mean self-information -log2 p(item) over recommended items.

    Reported alongside accuracy because a model can buy NDCG by recommending
    only blockbusters, and this is what exposes that trade.
    """
    total = float(item_popularity.sum())
    if total <= 0 or not recommended:
        return 0.0
    information = -np.log2(np.clip(item_popularity / total, 1e-12, None))
    scores = [float(information[ranked[:k]].mean()) for ranked in recommended if len(ranked)]
    return float(np.mean(scores)) if scores else 0.0
