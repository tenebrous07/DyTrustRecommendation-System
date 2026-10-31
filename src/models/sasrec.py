"""SASRec (Kang & McAuley, ICDM 2018): the standard sequential-recommendation baseline.

Included because it is the model graph recommenders are usually measured against on
Amazon data, and because a recommender that only beats matrix factorisation has not
yet shown much. It reads each user's interactions in time order and predicts the
next item with causal self-attention; it knows nothing about reviews, trust or the
graph. It is trained and scored by exactly the same code path as every other model
here: same chronological split, same negatives rule, same validation early stopping,
same full-catalogue evaluator.

At scoring time a user is represented by the last ``max_len`` items of their
*training* sequence; the user's held-out items are never seen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..training.early_stopping import ValidationMonitor
from ..training.negative_sampling import NegativeSampler
from ..utils.logging import get_logger, log_stage
from .base import BaseRecommender

__all__ = ["SASRecRecommender"]

logger = get_logger(__name__)


class _Encoder(nn.Module):
    """Item + position embeddings followed by causal self-attention blocks."""

    def __init__(self, n_items: int, max_len: int, dim: int, heads: int, layers: int,
                 dropout: float) -> None:
        super().__init__()
        # Index 0 is padding; real items are shifted up by one.
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.dropout = nn.Dropout(dropout)
        block = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.max_len = max_len
        nn.init.normal_(self.item_emb.weight, std=0.05)
        nn.init.normal_(self.pos_emb.weight, std=0.05)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(seq.shape[1], device=seq.device)
        x = self.item_emb(seq) * (self.item_emb.embedding_dim ** 0.5) + self.pos_emb(positions)
        x = self.dropout(x) * (seq != 0).unsqueeze(-1)
        causal = torch.triu(torch.ones(seq.shape[1], seq.shape[1], dtype=torch.bool,
                                       device=seq.device), diagonal=1)
        # Only the causal mask is passed. Masking padded keys as well would leave the
        # left-padded positions with no key to attend to, and a softmax over nothing
        # is NaN, which would poison the gradients. Padded positions are zeroed instead.
        keep = (seq != 0).unsqueeze(-1)
        x = self.blocks(x, mask=causal) * keep
        return self.norm(x) * keep


class SASRecRecommender(BaseRecommender):
    """score(u, i) = h_u . e_i, with h_u the encoder state after u's last training item."""

    name = "sasrec"

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64, num_layers: int = 2,
                 num_heads: int = 2, max_len: int = 50, dropout: float = 0.2,
                 learning_rate: float = 1e-3, weight_decay: float = 0.0, batch_size: int = 256,
                 epochs: int = 200, eval_every: int = 5, patience: int = 8,
                 device: str = "cpu", seed: int = 42, **kwargs) -> None:
        super().__init__(n_users, n_items, embedding_dim=embedding_dim, num_layers=num_layers,
                         num_heads=num_heads, max_len=max_len, dropout=dropout,
                         learning_rate=learning_rate, **kwargs)
        self.embedding_dim, self.num_layers, self.num_heads = embedding_dim, num_layers, num_heads
        self.max_len, self.dropout = max_len, dropout
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.batch_size, self.epochs = batch_size, epochs
        self.eval_every, self.patience = eval_every, patience
        self.device = torch.device(device)
        self.seed = seed
        self.best_val: float | None = None
        self.best_epoch: int | None = None

    # -- data --------------------------------------------------------------
    def _sequences(self, train: pd.DataFrame) -> dict[int, np.ndarray]:
        ordered = train.sort_values(["user_idx", "timestamp"], kind="stable")
        return {int(u): g["item_idx"].to_numpy() for u, g in ordered.groupby("user_idx")}

    def _pad(self, items: np.ndarray) -> np.ndarray:
        window = items[-self.max_len:] + 1
        out = np.zeros(self.max_len, dtype=np.int64)
        out[self.max_len - len(window):] = window
        return out

    # -- training ----------------------------------------------------------
    def fit(self, train: pd.DataFrame, sampler: NegativeSampler | None = None,
            validator=None, **kwargs) -> "SASRecRecommender":
        if sampler is None:
            raise ValueError("SASRecRecommender.fit requires a NegativeSampler.")
        torch.manual_seed(self.seed)
        self.sequences = self._sequences(train)
        trainable = [u for u, s in self.sequences.items() if len(s) >= 2]

        self.encoder = _Encoder(self.n_items, self.max_len, self.embedding_dim, self.num_heads,
                                self.num_layers, self.dropout).to(self.device)
        optimiser = torch.optim.Adam(self.encoder.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        monitor = (ValidationMonitor(list(self.encoder.parameters()), patience=self.patience)
                   if validator is not None else None)
        rng = np.random.default_rng(self.seed)

        with log_stage(logger, f"Fit SASRec (dim={self.embedding_dim}, up to {self.epochs} epochs)"):
            for epoch in range(1, self.epochs + 1):
                self.encoder.train()
                order = rng.permutation(len(trainable))
                total, count = 0.0, 0
                for start in range(0, len(order), self.batch_size):
                    users = np.array([trainable[i] for i in order[start:start + self.batch_size]])
                    inputs = np.stack([self._pad(self.sequences[u][:-1]) for u in users])
                    targets = np.stack([self._pad(self.sequences[u][1:]) for u in users])
                    # One negative per target position, never a positive of that user in any split.
                    negatives = sampler.sample(np.repeat(users, self.max_len), 1).reshape(
                        len(users), self.max_len) + 1
                    inp = torch.as_tensor(inputs, device=self.device)
                    pos = torch.as_tensor(targets, device=self.device)
                    neg = torch.as_tensor(negatives, device=self.device)
                    mask = pos != 0

                    hidden = self.encoder(inp)
                    table = self.encoder.item_emb
                    pos_score = (hidden * table(pos)).sum(-1)
                    neg_score = (hidden * table(neg)).sum(-1)
                    loss = (F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score),
                                                               reduction="none")
                            + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score),
                                                                 reduction="none"))[mask].mean()
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                    total += float(loss.detach()) * int(mask.sum())
                    count += int(mask.sum())
                if epoch % 10 == 0 or epoch == 1:
                    logger.info("  SASRec epoch %3d  loss %.5f", epoch, total / max(1, count))
                if monitor is not None and (epoch % self.eval_every == 0 or epoch == self.epochs):
                    self.is_fitted = True
                    if monitor.update(float(validator(self)), epoch):
                        break
        if monitor is not None:
            monitor.restore()
            self.best_val, self.best_epoch = monitor.best_val, monitor.best_epoch
        self.is_fitted = True
        return self

    # -- scoring -----------------------------------------------------------
    @torch.no_grad()
    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("fit() must be called before score_users().")
        self.encoder.eval()
        batch = np.stack([self._pad(self.sequences.get(int(u), np.array([], dtype=np.int64)))
                          for u in user_indices])
        hidden = self.encoder(torch.as_tensor(batch, device=self.device))[:, -1]
        scores = hidden @ self.encoder.item_emb.weight[1:].T
        return scores.cpu().numpy().astype(np.float32)

    def describe(self) -> dict:
        return {**super().describe(), "epochs_budget": self.epochs, "best_epoch": self.best_epoch,
                "best_val_ndcg@10": self.best_val}
