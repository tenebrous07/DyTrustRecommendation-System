"""Anomaly score -> trust score (Module 12).

The step CoLA does not do. It emits an unbounded anomaly score; a trust weight
must be bounded, comparable across runs, and safe to multiply into a loss. How
that conversion is done is a modelling choice, so it is configurable and
ablated rather than hardcoded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..graph.weighting import anomaly_to_trust
from ..utils.config import Config, resolve_path
from ..utils.logging import get_logger

__all__ = ["TrustScores", "build_trust_scores"]

logger = get_logger(__name__)


@dataclass
class TrustScores:
    """Per-entity trust, plus the mapping used to get there."""

    entity_ids: list[str]
    trust: np.ndarray
    anomaly: np.ndarray
    entity_type: str = "user"
    metadata: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_map(self) -> dict[str, float]:
        return {str(i): float(t) for i, t in zip(self.entity_ids, self.trust)}

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "entity_id": self.entity_ids, "entity_type": self.entity_type,
            "anomaly_score": self.anomaly, "trust_score": self.trust,
        })

    def for_interactions(self, frame: pd.DataFrame, neutral: float = 0.5) -> np.ndarray:
        """Propagate entity trust onto each interaction row.

        Entities absent from the score set (for example a user with no
        co-purchasing peer, whom CoLA cannot inform) receive ``neutral`` rather
        than 0. A trust of 0 would delete the interaction from a trust-weighted
        loss, which is a strong claim to make about a node we simply know
        nothing about.
        """
        key = "user_id" if self.entity_type == "user" else "item_id"
        mapping = self.as_map()
        values = frame[key].astype(str).map(mapping)
        missing = int(values.isna().sum())
        if missing:
            logger.info("%s interaction(s) have no trust score; assigned neutral %.2f.",
                        f"{missing:,}", neutral)
        return values.fillna(neutral).to_numpy(dtype=np.float32)

    def save(self, path: str | Path) -> Path:
        out = resolve_path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(out, index=False)
        out.with_suffix(".meta.json").write_text(
            json.dumps({"metadata": self.metadata, "notes": self.notes}, indent=2),
            encoding="utf-8",
        )
        return out


def build_trust_scores(
    cola_result, config: Config, entity_type: str = "user"
) -> TrustScores:
    """Apply the configured normalisation and transform."""
    cfg = config.require("trust")
    normalisation = str(cfg.get("normalisation", "rank"))
    floor = float(cfg.get("floor", 0.05))
    ceiling = float(cfg.get("ceiling", 1.0))

    anomaly = np.asarray(cola_result.anomaly_scores, dtype=np.float64)
    trust = anomaly_to_trust(anomaly, normalisation=normalisation,
                             floor=floor, ceiling=ceiling)

    scores = TrustScores(
        entity_ids=list(cola_result.node_ids), trust=trust, anomaly=anomaly,
        entity_type=entity_type,
        metadata={
            "normalisation": normalisation, "transform": "trust = 1 - normalised(anomaly)",
            "floor": floor, "ceiling": ceiling,
            "trust_mean": float(trust.mean()), "trust_std": float(trust.std()),
            "trust_min": float(trust.min()), "trust_max": float(trust.max()),
            **cola_result.metadata,
        },
        notes=[
            *cola_result.notes,
            f"Trust = 1 - {normalisation}-normalised anomaly, clamped to "
            f"[{floor}, {ceiling}].",
            f"The floor of {floor} is not cosmetic: a trust of exactly 0 in a "
            f"trust-weighted loss removes the interaction from training entirely and "
            f"changes the effective dataset size, which would confound the very "
            f"ablation the weighting exists to inform.",
            "Rank normalisation is the default because it is invariant to the score "
            "distribution's shape; min-max is dominated by a single extreme value and "
            "a z-score sigmoid assumes an approximate normality these scores lack.",
            "These trust values are UNVALIDATED against ground truth. No fraud labels "
            "exist in this dataset.",
        ],
    )
    logger.info("Trust scores: mean %.4f, std %.4f, min %.4f, max %.4f (%s normalisation)",
                trust.mean(), trust.std(), trust.min(), trust.max(), normalisation)
    return scores
