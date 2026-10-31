"""Behavioural trust detector.

CoLA scores a user by how well their attributes can be predicted from their
graph neighbourhood. That is the wrong question for review manipulation: a
coordinated campaign is consistent *by construction* (shared templates, shared
targets, shared timing), so each shill is easy to predict from the others and is
rated as more trustworthy than a genuine user. On the synthetic benchmark CoLA
reaches AUC 0.295, below chance, where one-line rules reach 0.92-0.99.

This detector asks the question the shilling-detection literature asks instead:
does the account *behave* like a campaign account? It uses three indicators that
need no labels and no graph:

* ``shared_text``  - share of the user's reviews whose text also appears under
  other accounts (campaign templates).
* ``own_repeat``   - share of the user's reviews that repeat the user's own text.
* ``burstiness``   - the most reviews the user posted inside any 24-hour window,
  as a share of their history.

The RDMA rating-gap indicator (distance from the item's mean rating) was tried
and dropped: on the synthetic benchmark it scored AUC 0.19-0.34, i.e. genuine
users deviate from consensus more than shills do, so it pointed the wrong way.

Each indicator is rank-normalised to [0, 1] and averaged, so no indicator
dominates by scale and no threshold has to be tuned. Users with too little
history to judge receive the median score, which is "no evidence", not "clean".

Everything is computed from the frame it is given, which must be the training
split only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

__all__ = ["BehaviouralResult", "BehaviouralTrustDetector", "INDICATORS", "INDICATORS_V2"]

logger = get_logger(__name__)

INDICATORS = ("shared_text", "own_repeat", "burstiness")
# Version 2 adds two indicators aimed at an attacker who knows the first three exist:
# paraphrased reviews (compared by meaning, not by exact text) and reviews spread over
# weeks (a seven-day window instead of one day).
INDICATORS_V2 = INDICATORS + ("burst_7d", "semantic_shared")
_DAY_MS = 24 * 3600 * 1000


@dataclass
class BehaviouralResult:
    """Same surface as the CoLA result, so ``build_trust_scores`` accepts either."""

    node_ids: list[str]
    anomaly_scores: np.ndarray
    indicators: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _rank01(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(values).rank(method="average").to_numpy()
    return (ranks - 1.0) / max(1, len(ranks) - 1)


def _max_in_window(times: np.ndarray, window: int) -> int:
    """Largest number of timestamps inside any ``window``-wide span."""
    times = np.sort(times)
    reach = np.searchsorted(times, times + window, side="right")
    return int((reach - np.arange(len(times))).max())


class BehaviouralTrustDetector:
    """Label-free, graph-free manipulation indicators per user."""

    def __init__(self, min_interactions: int = 3, min_text_length: int = 20,
                 similarity: float = 0.85, min_matches: int = 3, combine: str = "mean") -> None:
        if combine not in ("mean", "max"):
            raise ValueError(f"combine must be mean|max, got '{combine}'.")
        self.combine = combine
        self.min_interactions = min_interactions
        self.min_text_length = min_text_length
        self.similarity = similarity
        self.min_matches = min_matches

    def semantic_shared(self, interactions: pd.DataFrame, embeddings: np.ndarray,
                        block: int = 1536) -> pd.Series:
        """Share of each user's reviews that read like at least ``min_matches`` reviews by OTHER accounts.

        Compares sentence embeddings, so a campaign that rewrites every review is still
        found, where exact-text matching sees only unrelated strings.
        """
        import torch

        users = interactions["user_id"].astype(str).to_numpy()
        codes = torch.as_tensor(pd.factorize(users)[0])
        text = (interactions["review_text"].fillna("").astype(str).str.len()
                >= self.min_text_length).to_numpy()
        matrix = torch.as_tensor(np.asarray(embeddings, dtype=np.float32))
        flagged = np.zeros(len(users), dtype=bool)
        with torch.no_grad():
            for start in range(0, len(users), block):
                sims = matrix[start:start + block] @ matrix.T
                sims[codes[start:start + block, None] == codes[None, :]] = -1.0
                flagged[start:start + block] = ((sims >= self.similarity).sum(1) >= self.min_matches).numpy()
        flagged &= text
        return pd.Series(flagged.astype(float)).groupby(users).mean()

    def indicators(self, interactions: pd.DataFrame,
                   embeddings: np.ndarray | None = None) -> pd.DataFrame:
        frame = interactions[["user_id", "timestamp"]].copy()
        frame["user_id"] = frame["user_id"].astype(str)
        text = (interactions["review_text"].fillna("").astype(str).str.strip().str.lower()
                if "review_text" in interactions.columns
                else pd.Series("", index=interactions.index))
        frame["text"] = text.where(text.str.len() >= self.min_text_length, other=pd.NA)
        by_user = frame.groupby("user_id", sort=False)

        out = pd.DataFrame({"n_interactions": by_user.size()})

        # Text that other accounts also posted. Counted over distinct users so a
        # single person repeating themselves is not mistaken for a campaign.
        users_per_text = frame.dropna(subset=["text"]).groupby("text")["user_id"].nunique()
        shared = frame["text"].map(users_per_text).fillna(0) >= 2
        out["shared_text"] = shared.groupby(frame["user_id"], sort=False).mean()

        repeats = frame.dropna(subset=["text"]).groupby(["user_id", "text"]).size()
        repeated_rows = repeats[repeats > 1].groupby("user_id").sum()
        out["own_repeat"] = (repeated_rows.reindex(out.index).fillna(0) / out["n_interactions"])

        burst = by_user["timestamp"].apply(
            lambda s: _max_in_window(s.to_numpy(dtype=np.int64), _DAY_MS))
        out["burstiness"] = burst / out["n_interactions"]
        if embeddings is not None:
            week = 7 * _DAY_MS
            out["burst_7d"] = frame.groupby("user_id", sort=False)["timestamp"].apply(
                lambda t: _max_in_window(t.to_numpy(dtype=np.int64), week)) / out["n_interactions"]
            out["semantic_shared"] = self.semantic_shared(interactions, embeddings).reindex(out.index)
        return out

    def fit_score(self, interactions: pd.DataFrame, user_ids: list[str],
                  embeddings: np.ndarray | None = None) -> BehaviouralResult:
        """Anomaly score per user, ordered like ``user_ids`` (higher = more suspicious).

        With ``embeddings`` (one row per interaction) the two version-2 indicators are added.
        """
        used = INDICATORS_V2 if embeddings is not None else INDICATORS
        table = self.indicators(interactions, embeddings).reindex(user_ids)
        thin = table["n_interactions"].fillna(0) < self.min_interactions

        ranked = pd.DataFrame(index=table.index)
        for name in used:
            ranked[name] = _rank01(table[name].fillna(0.0).to_numpy())
        # "max" takes the strongest evidence: one convincing indicator is enough, and an
        # indicator that points the wrong way against some attacker cannot dilute it.
        score = (ranked.max(axis=1) if self.combine == "max" else ranked.mean(axis=1)).to_numpy().copy()
        # No evidence is not good evidence: thin histories sit at the median.
        score[thin.to_numpy()] = float(np.median(score[~thin.to_numpy()])
                                       if (~thin).any() else 0.5)

        logger.info("Behavioural detector: %s users scored, %s with thin history set to the median.",
                    f"{len(user_ids):,}", f"{int(thin.sum()):,}")
        return BehaviouralResult(
            node_ids=list(user_ids), anomaly_scores=score.astype(np.float64),
            indicators=table,
            metadata={"detector": "behavioural", "indicators": list(used),
                      "min_interactions": self.min_interactions,
                      "n_thin_history": int(thin.sum())},
            notes=["Label-free: three rank-normalised behavioural indicators, averaged with "
                   "equal weight. No threshold and no weight was tuned.",
                   "Thin-history users receive the median anomaly score (no evidence)."],
        )
