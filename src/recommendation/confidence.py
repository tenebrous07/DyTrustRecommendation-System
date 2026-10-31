"""Confidence estimation (Module 18).

A ranking score is **not** a confidence. A model can be maximally confident and
wrong, and it can rank an item first on almost no evidence. These are separate
quantities and are computed separately here.

Confidence blends three components, weighted by config:

* **Predictive stability** - 1 minus the normalised spread of the score, either
  under Monte Carlo dropout or across the members of a seed ensemble. Answers
  "would this model, retrained or resampled, still rank it here?"
* **Evidence volume** - how much history the user actually has, saturating, so
  the hundredth interaction adds less than the third.
* **Evidence trust** - the mean trust of the interactions supporting the user.

The behaviour that matters and is asserted in tests: a cold-start user must
come out with LOW confidence even when the score is high.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..utils.logging import get_logger

__all__ = ["ConfidenceEstimate", "ConfidenceEstimator"]

logger = get_logger(__name__)


@dataclass
class ConfidenceEstimate:
    """A confidence in [0, 1] plus the components that produced it."""

    item_id: str
    confidence: float
    components: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"item_id": self.item_id, "confidence": round(self.confidence, 4),
                "components": {k: round(v, 4) for k, v in self.components.items()},
                "reason": self.reason}


class ConfidenceEstimator:
    """Monte Carlo dropout plus evidence-based components."""

    def __init__(
        self,
        config,
        user_interaction_counts: dict[str, int] | None = None,
        trust_map: dict[str, float] | None = None,
        saturation: float = 20.0,
    ) -> None:
        cfg = config.require("confidence")
        self.method = str(cfg.get("method", "mc_dropout"))
        self.mc_samples = int(cfg.get("mc_samples", 20))
        components = cfg.get("components", {})
        self.w_stability = float(components.get("predictive_stability", 0.5))
        self.w_volume = float(components.get("evidence_volume", 0.3))
        self.w_trust = float(components.get("evidence_trust", 0.2))
        self.cold_start_penalty = bool(cfg.get("cold_start_penalty", True))
        self.min_interactions = int(
            config.get("cold_start.min_interactions_for_personalised", 3)
        )
        self.user_counts = user_interaction_counts or {}
        self.trust_map = trust_map or {}
        self.saturation = saturation

    # -- components --------------------------------------------------------
    @torch.no_grad()
    def predictive_stability(
        self, model, user_index: int, item_indices: np.ndarray
    ) -> np.ndarray:
        """1 - normalised std of the score across MC dropout samples.

        Dropout is re-enabled at inference so each pass samples a different
        sub-network; a score that survives that is one the model is not relying
        on a single fragile pathway for.
        """
        inner = getattr(model, "model", None)
        if inner is None or not hasattr(model, "x_dict"):
            return np.full(len(item_indices), np.nan)

        inner.train()  # activate dropout
        samples = []
        try:
            for _ in range(self.mc_samples):
                out = inner(model.x_dict, model.edge_index_dict, model.sbert_dim)
                user_vec = out["user"][user_index]
                scores = (out["product"][item_indices] * user_vec).sum(dim=1)
                samples.append(scores.detach().cpu().numpy())
        finally:
            inner.eval()

        stacked = np.stack(samples)
        std = stacked.std(axis=0)
        spread = float(np.abs(stacked.mean(axis=0)).mean()) + 1e-9
        return np.clip(1.0 - (std / spread), 0.0, 1.0)

    @staticmethod
    def ensemble_stability(member_scores: np.ndarray) -> np.ndarray:
        """Stability from independently trained models, same normalisation as MC dropout.

        Args:
            member_scores: ``(members, items)`` scores of one user's candidate items.
        """
        member_scores = np.asarray(member_scores, dtype=np.float64)
        std = member_scores.std(axis=0)
        spread = float(np.abs(member_scores.mean(axis=0)).mean()) + 1e-9
        return np.clip(1.0 - std / spread, 0.0, 1.0)

    def evidence_volume(self, user_id: str) -> float:
        """Saturating function of interaction count."""
        count = self.user_counts.get(str(user_id), 0)
        return float(1.0 - np.exp(-count / self.saturation))

    def evidence_trust(self, user_id: str) -> float:
        return float(self.trust_map.get(str(user_id), 0.5))

    # -- combination -------------------------------------------------------
    def estimate(
        self, model, user_id: str, user_index: int,
        item_indices: np.ndarray, item_ids: list[str],
        stability: np.ndarray | None = None,
    ) -> list[ConfidenceEstimate]:
        """``stability`` may be supplied directly (for example from a seed ensemble)."""
        if stability is None:
            stability = (self.predictive_stability(model, user_index, item_indices)
                         if self.method == "mc_dropout"
                         else np.full(len(item_indices), np.nan))
        stability = np.asarray(stability, dtype=float)
        volume = self.evidence_volume(user_id)
        trust = self.evidence_trust(user_id)
        count = self.user_counts.get(str(user_id), 0)
        is_cold = count < self.min_interactions

        estimates = []
        for position, item_id in enumerate(item_ids):
            per_item_stability = stability[position]
            if np.isnan(per_item_stability):
                # No MC dropout available; renormalise over the components we
                # do have rather than silently substituting a made-up 0.5.
                total = self.w_volume + self.w_trust
                confidence = (self.w_volume * volume + self.w_trust * trust) / max(total, 1e-9)
                components = {"evidence_volume": volume, "evidence_trust": trust}
                reason_parts = ["MC dropout unavailable for this model"]
            else:
                confidence = (self.w_stability * float(per_item_stability)
                              + self.w_volume * volume + self.w_trust * trust)
                components = {"predictive_stability": float(per_item_stability),
                              "evidence_volume": volume, "evidence_trust": trust}
                reason_parts = []

            if self.cold_start_penalty and is_cold:
                # A high score on two interactions is a guess. Halving keeps the
                # ordering but makes the uncertainty visible.
                confidence *= 0.5
                reason_parts.append(
                    f"cold-start user with only {count} training interaction(s), "
                    f"so confidence is halved"
                )
            else:
                reason_parts.append(
                    f"{count} training interaction(s), supporting-evidence trust {trust:.2f}"
                )

            estimates.append(ConfidenceEstimate(
                item_id=str(item_id), confidence=float(np.clip(confidence, 0.0, 1.0)),
                components=components, reason="; ".join(reason_parts),
            ))
        return estimates
