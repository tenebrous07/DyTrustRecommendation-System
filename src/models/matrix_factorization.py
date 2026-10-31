"""BPR matrix factorisation (Module 28, baseline 3)."""

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

__all__ = ["MatrixFactorizationRecommender"]

logger = get_logger(__name__)


class MatrixFactorizationRecommender(BaseRecommender):
    """score(u, i) = p_u . q_i + b_i, trained with BPR.

    The item bias term is included because this catalogue is highly skewed
    (Gini 0.521); without it the latent factors spend capacity re-learning
    popularity instead of taste.
    """

    name = "matrix_factorization"

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 epochs: int = 50, learning_rate: float = 0.01,
                 weight_decay: float = 1e-5, batch_size: int = 8192,
                 device: str = "cpu", seed: int = 42, eval_every: int = 5,
                 patience: int = 8, **kwargs) -> None:
        super().__init__(n_users, n_items, embedding_dim=embedding_dim, epochs=epochs,
                         learning_rate=learning_rate, weight_decay=weight_decay, **kwargs)
        self.embedding_dim = embedding_dim
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

    def fit(self, train: pd.DataFrame, sampler: NegativeSampler | None = None,
            validator=None, **kwargs) -> "MatrixFactorizationRecommender":
        """Train; with a ``validator`` (higher is better) stop early and keep the best epoch."""
        torch.manual_seed(self.seed)
        self.user_emb = nn.Embedding(self.n_users, self.embedding_dim).to(self.device)
        self.item_emb = nn.Embedding(self.n_items, self.embedding_dim).to(self.device)
        self.item_bias = nn.Embedding(self.n_items, 1).to(self.device)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        nn.init.zeros_(self.item_bias.weight)

        parameters = list(self.user_emb.parameters()) + list(self.item_emb.parameters()) \
            + list(self.item_bias.parameters())
        optimiser = torch.optim.Adam(parameters, lr=self.learning_rate,
                                     weight_decay=self.weight_decay)

        users = train["user_idx"].to_numpy()
        items = train["item_idx"].to_numpy()
        if sampler is None:
            raise ValueError("MatrixFactorizationRecommender.fit requires a NegativeSampler.")

        monitor = (ValidationMonitor(parameters, patience=self.patience)
                   if validator is not None else None)
        rng = np.random.default_rng(self.seed)
        with log_stage(logger, f"Fit MF (dim={self.embedding_dim}, {self.epochs} epochs)"):
            for epoch in range(1, self.epochs + 1):
                order = rng.permutation(len(users))
                total = 0.0
                for start in range(0, len(order), self.batch_size):
                    batch = order[start:start + self.batch_size]
                    u = users[batch]
                    pos = items[batch]
                    neg = sampler.sample(u, 1).ravel()

                    ut = torch.as_tensor(u, device=self.device)
                    pt = torch.as_tensor(pos, device=self.device)
                    nt = torch.as_tensor(neg, device=self.device)

                    pu = self.user_emb(ut)
                    pos_score = (pu * self.item_emb(pt)).sum(1) + self.item_bias(pt).squeeze(1)
                    neg_score = (pu * self.item_emb(nt)).sum(1) + self.item_bias(nt).squeeze(1)
                    loss = bpr_loss(pos_score, neg_score)

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                    total += float(loss.detach()) * len(batch)
                if epoch % 10 == 0 or epoch == 1:
                    logger.info("  MF epoch %3d/%d  loss %.5f", epoch, self.epochs,
                                total / len(order))
                if monitor is not None and (epoch % self.eval_every == 0
                                            or epoch == self.epochs):
                    self.is_fitted = True
                    if monitor.update(float(validator(self)), epoch):
                        break
        if monitor is not None:
            monitor.restore()
            self.best_val, self.best_epoch = monitor.best_val, monitor.best_epoch
        self.is_fitted = True
        return self

    @torch.no_grad()
    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("fit() must be called before score_users().")
        u = torch.as_tensor(np.asarray(user_indices), device=self.device)
        scores = self.user_emb(u) @ self.item_emb.weight.T + self.item_bias.weight.T
        return scores.cpu().numpy().astype(np.float32)
