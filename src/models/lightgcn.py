"""LightGCN (Module 28, baseline 4).

He et al., SIGIR 2020. Existing work, used here as the strongest graph baseline
that DyTrustRec must beat to justify its extra machinery.

LightGCN deliberately strips the non-linearity and feature transformation out
of GCN, keeping only neighbourhood aggregation over the normalised bipartite
adjacency, and averages the embeddings from every propagation layer. That makes
it the right control for this project: it isolates *what heterogeneity,
semantics and trust add*, because LightGCN already captures the pure
collaborative graph signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..training.early_stopping import ValidationMonitor
from ..training.losses import bpr_loss
from ..training.negative_sampling import NegativeSampler
from ..utils.logging import get_logger, log_stage
from .base import BaseRecommender

__all__ = ["LightGCNRecommender"]

logger = get_logger(__name__)


class LightGCNRecommender(BaseRecommender):
    """Layer-averaged propagation over the symmetric-normalised bipartite graph."""

    name = "lightgcn"

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 num_layers: int = 3, epochs: int = 100, learning_rate: float = 0.05,
                 weight_decay: float = 0.0, batch_size: int = 8192,
                 device: str = "cpu", seed: int = 42, eval_every: int = 5,
                 patience: int = 8, **kwargs) -> None:
        super().__init__(n_users, n_items, embedding_dim=embedding_dim,
                         num_layers=num_layers, epochs=epochs,
                         learning_rate=learning_rate, **kwargs)
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.seed = seed
        self.eval_every = eval_every
        self.patience = patience
        self.best_val: float | None = None
        self.best_epoch: int | None = None

    def _build_adjacency(self, train: pd.DataFrame) -> torch.Tensor:
        """Symmetric-normalised sparse adjacency A_hat = D^-1/2 A D^-1/2."""
        users = train["user_idx"].to_numpy()
        items = train["item_idx"].to_numpy() + self.n_users
        total = self.n_users + self.n_items

        src = np.concatenate([users, items])
        dst = np.concatenate([items, users])
        degree = np.bincount(src, minlength=total).astype(np.float32)
        degree[degree == 0] = 1.0
        values = (1.0 / np.sqrt(degree[src])) * (1.0 / np.sqrt(degree[dst]))

        return torch.sparse_coo_tensor(
            torch.as_tensor(np.stack([src, dst]), dtype=torch.long),
            torch.as_tensor(values, dtype=torch.float32),
            size=(total, total),
        ).coalesce().to(self.device)

    def _propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        accumulated = embeddings
        current = embeddings
        for _ in range(self.num_layers):
            current = torch.sparse.mm(self.adjacency, current)
            accumulated = accumulated + current
        # Mean over the input layer plus every propagation layer.
        accumulated = accumulated / (self.num_layers + 1)
        return accumulated[:self.n_users], accumulated[self.n_users:]

    @torch.no_grad()
    def _refresh(self) -> None:
        self.final_user, self.final_item = self._propagate()
        self.is_fitted = True

    def fit(self, train: pd.DataFrame, sampler: NegativeSampler | None = None,
            validator=None, **kwargs) -> "LightGCNRecommender":
        """Train; with a ``validator`` (higher is better) stop early and keep the best epoch."""
        if sampler is None:
            raise ValueError("LightGCNRecommender.fit requires a NegativeSampler.")
        torch.manual_seed(self.seed)

        self.user_emb = nn.Embedding(self.n_users, self.embedding_dim).to(self.device)
        self.item_emb = nn.Embedding(self.n_items, self.embedding_dim).to(self.device)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        self.adjacency = self._build_adjacency(train)

        # Scale note, found by diagnosis rather than assumed: averaging over
        # L+1 layers shrinks embedding norms, so at std=0.1 init the dot
        # products are ~1e-3, sigmoid(s_pos - s_neg) sits at 0.5, and the BPR
        # loss pins at ln(2) = 0.69315 with vanishing gradients. Measured: at
        # lr=1e-3 the item-embedding norm decays to 0.04 and the model never
        # escapes; at lr=0.05 with no weight decay it reaches 4.27 and trains.
        # Weight decay is 0 because L2 on top of that decay is what kills it.

        optimiser = torch.optim.Adam(
            list(self.user_emb.parameters()) + list(self.item_emb.parameters()),
            lr=self.learning_rate, weight_decay=self.weight_decay,
        )

        users = train["user_idx"].to_numpy()
        items = train["item_idx"].to_numpy()
        monitor = (ValidationMonitor([self.user_emb.weight, self.item_emb.weight],
                                     patience=self.patience)
                   if validator is not None else None)
        rng = np.random.default_rng(self.seed)

        with log_stage(logger, f"Fit LightGCN ({self.num_layers} layers, {self.epochs} epochs)"):
            for epoch in range(1, self.epochs + 1):
                order = rng.permutation(len(users))
                total = 0.0
                for start in range(0, len(order), self.batch_size):
                    batch = order[start:start + self.batch_size]
                    u = torch.as_tensor(users[batch], device=self.device)
                    pos = torch.as_tensor(items[batch], device=self.device)
                    neg = torch.as_tensor(sampler.sample(users[batch], 1).ravel(),
                                          device=self.device)

                    user_out, item_out = self._propagate()
                    pu = user_out[u]
                    loss = bpr_loss((pu * item_out[pos]).sum(1), (pu * item_out[neg]).sum(1))

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                    total += float(loss.detach()) * len(batch)
                if epoch % 10 == 0 or epoch == 1:
                    logger.info("  LightGCN epoch %3d/%d  loss %.5f", epoch, self.epochs,
                                total / len(order))
                if monitor is not None and (epoch % self.eval_every == 0
                                            or epoch == self.epochs):
                    self._refresh()
                    if monitor.update(float(validator(self)), epoch):
                        break

        if monitor is not None:
            monitor.restore()
            self.best_val, self.best_epoch = monitor.best_val, monitor.best_epoch
        self._refresh()
        return self

    @torch.no_grad()
    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("fit() must be called before score_users().")
        u = torch.as_tensor(np.asarray(user_indices), device=self.device)
        return (self.final_user[u] @ self.final_item.T).cpu().numpy().astype(np.float32)
