"""Trust- and time-weighted edge weights (Module 12).

    w_e = base * trust_e^alpha * temporal_e^beta

Both factors default to 1.0 when their component is disabled, so switching
trust or temporal weighting off in an ablation leaves the plain graph exactly
as it was rather than subtly rescaling every edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.config import Config
from ..utils.logging import get_logger

__all__ = ["compute_edge_weights", "anomaly_to_trust"]

logger = get_logger(__name__)


def anomaly_to_trust(
    anomaly_scores: np.ndarray,
    normalisation: str = "rank",
    floor: float = 0.05,
    ceiling: float = 1.0,
) -> np.ndarray:
    """Convert unbounded anomaly scores into a bounded trust value.

    CoLA emits an unbounded score that is neither a probability nor a trust
    value; this mapping is a modelling choice, which is why the normalisation is
    a config option and is ablated rather than fixed.

    ``rank`` is the default because it is invariant to the score distribution's
    shape. min-max is dominated by a single extreme score, and a z-score sigmoid
    assumes an approximate normality the scores do not have.

    The floor matters more than it looks: a trust of exactly 0 in a
    trust-weighted loss silently deletes an interaction from training and
    changes the effective dataset size, which would confound the ablation it is
    supposed to inform.
    """
    scores = np.asarray(anomaly_scores, dtype=np.float64)
    if scores.size == 0:
        return scores.astype(np.float32)

    if normalisation == "rank":
        order = scores.argsort().argsort()
        normalised = order / max(1, len(scores) - 1)
    elif normalisation == "minmax":
        low, high = float(scores.min()), float(scores.max())
        normalised = np.zeros_like(scores) if high - low < 1e-12 else (scores - low) / (high - low)
    elif normalisation == "zscore_sigmoid":
        std = float(scores.std()) or 1.0
        normalised = 1.0 / (1.0 + np.exp(-(scores - float(scores.mean())) / std))
    else:
        raise ValueError(
            f"Unknown trust.normalisation '{normalisation}'. "
            f"Use rank|minmax|zscore_sigmoid."
        )

    trust = 1.0 - normalised
    return np.clip(trust, floor, ceiling).astype(np.float32)


def compute_edge_weights(
    frame: pd.DataFrame,
    config: Config,
    trust: np.ndarray | pd.Series | None = None,
    temporal: np.ndarray | pd.Series | None = None,
) -> np.ndarray:
    """Combine base, trust and temporal factors into one weight per edge."""
    cfg = config.require("edge_weighting")
    base = float(cfg.get("base", 1.0))
    alpha = float(cfg.get("trust_exponent", 1.0))
    beta = float(cfg.get("temporal_exponent", 1.0))
    use_trust = bool(cfg.get("use_trust", True))
    use_temporal = bool(cfg.get("use_temporal", True))

    n = len(frame)
    weights = np.full(n, base, dtype=np.float64)

    if use_trust and trust is not None:
        values = np.asarray(trust, dtype=np.float64)
        if values.shape[0] != n:
            raise ValueError(f"trust has {values.shape[0]} values for {n} edges.")
        weights *= np.power(np.clip(values, 1e-6, None), alpha)
    elif use_trust:
        logger.warning("edge_weighting.use_trust is true but no trust supplied; "
                       "trust factor left at 1.0.")

    if use_temporal and temporal is not None:
        values = np.asarray(temporal, dtype=np.float64)
        if values.shape[0] != n:
            raise ValueError(f"temporal has {values.shape[0]} values for {n} edges.")
        weights *= np.power(np.clip(values, 1e-6, None), beta)
    elif use_temporal:
        logger.warning("edge_weighting.use_temporal is true but no temporal weights "
                       "supplied; temporal factor left at 1.0.")

    if not np.isfinite(weights).all():
        raise ValueError("Edge weight computation produced non-finite values.")
    logger.info("Edge weights: mean %.4f, min %.4f, max %.4f (trust=%s, temporal=%s)",
                weights.mean(), weights.min(), weights.max(), use_trust, use_temporal)
    return weights.astype(np.float32)
