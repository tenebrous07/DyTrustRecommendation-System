"""CoLA: contrastive self-supervised graph anomaly detection (Module 11).

CoLA is Liu et al., IEEE TNNLS 2021 - **existing work**. It is used here, not
invented here.

Its published formulation operates on a homogeneous attributed graph, so it
cannot be pointed at the heterogeneous graph as-is. The adaptation is explicit
and is the thing that needs defending, not the algorithm:

1. Project the heterogeneous graph down to a **user-product bipartite graph**.
2. For each target node, sample a local subgraph.
3. Train a contrastive discriminator to tell "this subgraph contains this node"
   from "this subgraph belongs to somebody else".
4. A node whose neighbourhood *fails* to predict it is anomalous.

Two approximations, both deliberate and both recorded in the output:

* The published sampler is random-walk-with-restart. This uses uniform
  neighbour sampling over 2 hops, which is far cheaper and, on a bipartite
  graph where every 2-hop neighbour is a co-purchasing peer, samples from
  essentially the same neighbourhood.
* The score is averaged over R independent rounds because a single round is
  extremely noisy - this follows the original paper.

What comes out is an **unbounded anomaly score**, not a probability and not a
trust value. Converting it is :mod:`src.anomaly.trust`'s job, and it is a
modelling choice rather than a fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

from ..utils.logging import get_logger, log_stage

__all__ = ["CoLADetector", "CoLAResult"]

logger = get_logger(__name__)


@dataclass
class CoLAResult:
    """Per-node anomaly scores plus the provenance needed to read them."""

    node_ids: list[str]
    anomaly_scores: np.ndarray
    entity_type: str = "user"
    training_loss: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "entity_id": self.node_ids,
            "entity_type": self.entity_type,
            "anomaly_score": self.anomaly_scores,
            "anomaly_rank": pd.Series(self.anomaly_scores).rank(
                ascending=False, method="min").astype(int),
        })


class _Encoder(nn.Module):
    """One GCN-style layer plus a bilinear discriminator.

    Deliberately shallow, exactly as in the paper: the contrastive task is
    about local agreement, and depth would let the encoder smooth away the
    local irregularity the method exists to detect.
    """

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.transform = nn.Linear(in_dim, hidden_dim)
        self.target = nn.Linear(in_dim, hidden_dim)
        self.discriminator = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target_x: torch.Tensor, context_x: torch.Tensor,
                context_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target_x:    (B, F)      the node being scored
            context_x:   (B, C, F)   its sampled local subgraph
            context_mask:(B, C)      1 for real neighbours, 0 for padding

        Returns:
            (B,) discriminator logits.
        """
        hidden = self.dropout(F.relu(self.transform(context_x)))
        weights = context_mask.unsqueeze(-1)
        denominator = weights.sum(dim=1).clamp(min=1.0)
        readout = (hidden * weights).sum(dim=1) / denominator     # masked mean
        query = self.dropout(F.relu(self.target(target_x)))
        return self.discriminator(query, readout).squeeze(-1)


class CoLADetector:
    """Trains the contrastive task and emits per-node anomaly scores."""

    def __init__(self, config, device: str = "cpu", seed: int = 42) -> None:
        self.cfg = config.require("cola")
        self.device = torch.device(device)
        self.seed = seed
        self.hidden_dim = int(self.cfg.get("hidden_dimension", 64))
        self.subgraph_size = int(self.cfg.get("subgraph_size", 4))
        self.epochs = int(self.cfg.get("epochs", 100))
        self.learning_rate = float(self.cfg.get("learning_rate", 1e-3))
        self.weight_decay = float(self.cfg.get("weight_decay", 0.0))
        self.dropout = float(self.cfg.get("dropout", 0.1))
        self.eval_rounds = int(self.cfg.get("eval_rounds", 32))
        self.rng = np.random.default_rng(seed)

    # -- projection --------------------------------------------------------
    @staticmethod
    def build_projection(train: pd.DataFrame, n_users: int, n_items: int) -> sparse.csr_matrix:
        """Step 1: the user-product bipartite projection CoLA runs on."""
        return sparse.csr_matrix(
            (np.ones(len(train), dtype=np.float32),
             (train["user_idx"].to_numpy(), train["item_idx"].to_numpy())),
            shape=(n_users, n_items),
        )

    def _sample_contexts(self, adjacency: sparse.csr_matrix,
                         co_occurrence: sparse.csr_matrix,
                         targets: np.ndarray) -> np.ndarray:
        """Uniformly sample up to ``subgraph_size`` 2-hop peers per target.

        On a user-product bipartite graph a 2-hop neighbour of a user is another
        user who touched a shared item, so the sampled context is exactly the
        set of co-purchasing peers whose agreement CoLA tests.
        """
        contexts = np.full((len(targets), self.subgraph_size), -1, dtype=np.int64)
        for row, node in enumerate(targets):
            start, end = co_occurrence.indptr[node], co_occurrence.indptr[node + 1]
            peers = co_occurrence.indices[start:end]
            peers = peers[peers != node]
            if len(peers) == 0:
                continue
            size = min(self.subgraph_size, len(peers))
            chosen = self.rng.choice(len(peers), size=size, replace=False)
            contexts[row, :size] = peers[chosen]
        return contexts

    # -- training ----------------------------------------------------------
    def fit_score(
        self,
        features: np.ndarray,
        adjacency: sparse.csr_matrix,
        node_ids: list[str],
        entity_type: str = "user",
    ) -> CoLAResult:
        """Train the contrastive task and score every node.

        Args:
            features: ``(n_nodes, F)`` node attributes.
            adjacency: ``(n_users, n_items)`` bipartite incidence matrix.
            node_ids: Original ids, positionally aligned with ``features``.
        """
        torch.manual_seed(self.seed)
        n_nodes = features.shape[0]

        with log_stage(logger, "CoLA: build co-occurrence graph"):
            # A @ A.T links users who share at least one item.
            co_occurrence = (adjacency @ adjacency.T).tocsr()
            co_occurrence.setdiag(0)
            co_occurrence.eliminate_zeros()

        x = torch.as_tensor(np.nan_to_num(features), dtype=torch.float32, device=self.device)
        model = _Encoder(x.shape[1], self.hidden_dim, self.dropout).to(self.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        losses: list[float] = []
        all_nodes = np.arange(n_nodes)

        with log_stage(logger, f"CoLA: contrastive training ({self.epochs} epochs)"):
            for epoch in range(1, self.epochs + 1):
                model.train()
                contexts = self._sample_contexts(adjacency, co_occurrence, all_nodes)
                mask = torch.as_tensor((contexts >= 0).astype(np.float32), device=self.device)
                safe = np.clip(contexts, 0, n_nodes - 1)
                context_x = x[torch.as_tensor(safe, device=self.device)]

                positive = model(x, context_x, mask)
                # Negative pair: the same context matched against a shuffled
                # target, which is the paper's corruption strategy.
                shuffled = torch.as_tensor(self.rng.permutation(n_nodes), device=self.device)
                negative = model(x[shuffled], context_x, mask)

                loss = (F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive))
                        + F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative)))
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                losses.append(float(loss))
                if epoch % 25 == 0 or epoch == 1:
                    logger.info("  CoLA epoch %3d/%d  loss %.5f", epoch, self.epochs, float(loss))

        with log_stage(logger, f"CoLA: score over {self.eval_rounds} rounds"):
            model.eval()
            accumulated = np.zeros(n_nodes, dtype=np.float64)
            with torch.no_grad():
                for _ in range(self.eval_rounds):
                    contexts = self._sample_contexts(adjacency, co_occurrence, all_nodes)
                    mask = torch.as_tensor((contexts >= 0).astype(np.float32),
                                           device=self.device)
                    context_x = x[torch.as_tensor(np.clip(contexts, 0, n_nodes - 1),
                                                  device=self.device)]
                    positive = torch.sigmoid(model(x, context_x, mask))
                    shuffled = torch.as_tensor(self.rng.permutation(n_nodes), device=self.device)
                    negative = torch.sigmoid(model(x[shuffled], context_x, mask))
                    # Paper's score: a normal node's own context predicts it far
                    # better than a random node's, so the gap is small.
                    accumulated += (negative - positive).cpu().numpy()
            scores = accumulated / self.eval_rounds

        isolated = int((co_occurrence.getnnz(axis=1) == 0).sum())
        result = CoLAResult(
            node_ids=list(node_ids), anomaly_scores=scores.astype(np.float32),
            entity_type=entity_type, training_loss=losses,
            metadata={
                "hidden_dimension": self.hidden_dim, "subgraph_size": self.subgraph_size,
                "epochs": self.epochs, "eval_rounds": self.eval_rounds,
                "n_nodes": n_nodes, "n_isolated_nodes": isolated,
                "final_loss": losses[-1] if losses else None,
            },
            notes=[
                "CoLA (Liu et al., TNNLS 2021) is existing work; this project uses it, "
                "it does not claim it.",
                "Adapted to a user-product bipartite projection because the published "
                "formulation is for homogeneous attributed graphs.",
                "Neighbour sampling is uniform over 2 hops rather than random-walk-with-"
                "restart. Cheaper, and on a bipartite graph the 2-hop neighbourhood is "
                "exactly the co-purchasing peer set the original sampler targets.",
                "Output is an UNBOUNDED anomaly score, not a probability and not trust.",
                f"{isolated:,} nodes have no co-occurring peer and therefore receive an "
                f"uninformative score; they are given neutral trust downstream.",
                "This dataset has no fraud labels, so these scores CANNOT be validated "
                "against ground truth. Only the synthetic-injection protocol can.",
            ],
        )
        logger.info("CoLA scores: mean %.5f, std %.5f, range [%.5f, %.5f]",
                    scores.mean(), scores.std(), scores.min(), scores.max())
        return result
