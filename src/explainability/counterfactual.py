"""Counterfactual "why not?" analysis (Module 17).

The question is not "why is this item ranked 8th?" but "what would have to
change for it to reach the Top-K?" - and the answer must be *computed against
the model*, not written as a plausible sentence.

Method: perturb the user's representation toward a named factor's direction by
a scalar alpha, re-score, and bisect on alpha for the smallest value that lifts
the item across the Top-K boundary. If no alpha within bounds achieves it, the
result is reported as **infeasible** rather than dressed up.

The perturbation directions are chosen so the answer is interpretable:

* ``category_affinity`` - move toward the centroid of the item's category.
* ``brand_affinity``    - move toward the centroid of the item's brand.
* ``semantic_similarity`` - move toward the item's own embedding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..utils.logging import get_logger

__all__ = ["CounterfactualResult", "CounterfactualAnalyzer"]

logger = get_logger(__name__)


@dataclass
class CounterfactualResult:
    """What would have to change, or an explicit statement that nothing would."""

    item_id: str
    current_rank: int
    target_k: int
    feasible: bool
    factor: str | None = None
    required_alpha: float | None = None
    achieved_rank: int | None = None
    text: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"item_id": self.item_id, "current_rank": self.current_rank,
                "target_k": self.target_k, "feasible": self.feasible,
                "factor": self.factor,
                "required_alpha": None if self.required_alpha is None
                else round(self.required_alpha, 4),
                "achieved_rank": self.achieved_rank, "text": self.text,
                "diagnostics": self.diagnostics}


class CounterfactualAnalyzer:
    """Bisection search over interpretable perturbations of the user vector."""

    def __init__(
        self,
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        vocabulary,
        item_categories: np.ndarray | None = None,
        item_brands: np.ndarray | None = None,
        masked_items: Callable[[int], np.ndarray] | None = None,
        max_iterations: int = 20,
        max_alpha: float = 5.0,
    ) -> None:
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        self.vocab = vocabulary
        self.item_categories = item_categories
        self.item_brands = item_brands
        self.masked_items = masked_items
        self.max_iterations = max_iterations
        self.max_alpha = max_alpha

    # -- directions --------------------------------------------------------
    def _direction(self, factor: str, item_index: int) -> np.ndarray | None:
        """A unit vector in embedding space corresponding to a named factor."""
        if factor == "semantic_similarity":
            target = self.item_embeddings[item_index]
        elif factor == "category_affinity" and self.item_categories is not None:
            peers = np.flatnonzero(self.item_categories == self.item_categories[item_index])
            if len(peers) == 0:
                return None
            target = self.item_embeddings[peers].mean(axis=0)
        elif factor == "brand_affinity" and self.item_brands is not None:
            peers = np.flatnonzero(self.item_brands == self.item_brands[item_index])
            if len(peers) == 0:
                return None
            target = self.item_embeddings[peers].mean(axis=0)
        else:
            return None
        norm = np.linalg.norm(target)
        return target / norm if norm > 1e-12 else None

    def _rank(self, user_vector: np.ndarray, item_index: int, user_index: int) -> int:
        scores = self.item_embeddings @ user_vector
        if self.masked_items is not None:
            masked = self.masked_items(user_index)
            if masked is not None and len(masked):
                scores[masked] = -np.inf
        return int((scores > scores[item_index]).sum()) + 1

    # -- analysis ----------------------------------------------------------
    def analyse(self, user_id: str, item_id: str, k: int = 5,
                factors: list[str] | None = None) -> CounterfactualResult:
        """Find the smallest interpretable change that lifts the item into Top-K."""
        user_index = self.vocab.users.get(str(user_id))
        item_index = self.vocab.items.get(str(item_id))
        if user_index is None or item_index is None:
            raise KeyError(f"Unknown user '{user_id}' or item '{item_id}'.")

        base_vector = self.user_embeddings[user_index].astype(np.float64)
        current_rank = self._rank(base_vector, item_index, user_index)

        if current_rank <= k:
            return CounterfactualResult(
                item_id=str(item_id), current_rank=current_rank, target_k=k, feasible=True,
                achieved_rank=current_rank,
                text=f"This item is already ranked {current_rank}, inside the Top-{k}.",
            )

        factors = factors or ["category_affinity", "brand_affinity", "semantic_similarity"]
        best: CounterfactualResult | None = None
        attempted: dict[str, Any] = {}

        for factor in factors:
            direction = self._direction(factor, item_index)
            if direction is None:
                attempted[factor] = "no direction available (missing category/brand data)"
                continue

            scale = np.linalg.norm(base_vector) or 1.0
            high = self.max_alpha
            rank_at_high = self._rank(base_vector + high * scale * direction,
                                      item_index, user_index)
            if rank_at_high > k:
                # Even the largest admissible push fails: say so, do not invent
                # a smaller number that would not work.
                attempted[factor] = (f"infeasible - even alpha={high} only reaches "
                                     f"rank {rank_at_high}")
                continue

            low = 0.0
            for _ in range(self.max_iterations):
                mid = (low + high) / 2
                if self._rank(base_vector + mid * scale * direction,
                              item_index, user_index) <= k:
                    high = mid
                else:
                    low = mid
            achieved = self._rank(base_vector + high * scale * direction,
                                  item_index, user_index)
            attempted[factor] = {"required_alpha": round(high, 4), "achieved_rank": achieved}

            if best is None or high < (best.required_alpha or np.inf):
                readable = factor.replace("_", " ")
                best = CounterfactualResult(
                    item_id=str(item_id), current_rank=current_rank, target_k=k,
                    feasible=True, factor=factor, required_alpha=high,
                    achieved_rank=achieved,
                    text=(f"This item is ranked {current_rank}. It would enter the Top-{k} "
                          f"if the user's {readable} toward this product increased by "
                          f"{high:.2f} standard user-vector norms - the smallest change "
                          f"among the factors tested."),
                    diagnostics={"factors_tested": attempted},
                )

        if best is None:
            return CounterfactualResult(
                item_id=str(item_id), current_rank=current_rank, target_k=k, feasible=False,
                text=(f"This item is ranked {current_rank} and no tested factor moves it into "
                      f"the Top-{k} within the search bounds. The gap is not attributable to "
                      f"a single interpretable factor."),
                diagnostics={"factors_tested": attempted},
            )
        best.diagnostics = {"factors_tested": attempted}
        return best
