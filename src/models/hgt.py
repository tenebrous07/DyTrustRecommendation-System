"""Heterogeneous Graph Transformer recommender (Modules 14, 15, 31).

HGT is Hu et al., WWW 2020 - existing work, not a contribution of this project.
What is assembled here is the recommendation head around it: typed input
projections, the learned fusion of SBERT semantics with behavioural features,
dot-product scoring, and the training procedure.

Two training paths, selected by ``training.full_batch``:

* **full_batch: true** - one forward pass and one optimiser step per epoch over
  the whole graph.
* **full_batch: false** - ``LinkNeighborLoader`` samples a subgraph per batch of
  interactions (the route to the larger tiers).

What the full-batch path does, and why (all of it measured, see the paper):

* **Edge-masked training.** The interaction being scored is also a
  message-passing edge, so an unmasked model can read the answer off the graph:
  training loss collapses (0.30 -> 0.04) while held-out NDCG stalls and the
  recommendations spread over 62% of the catalogue, which is memorisation, not
  taste. Each epoch a random share of interaction edges is hidden from message
  passing and those hidden rows are the supervision targets. Evaluation uses
  the full graph. All edge types that encode the same training row (purchased,
  wrote, about and their reverses) share one mask.
* **Validation early stopping.** Held-out NDCG@10 is measured every
  ``training.eval_every`` epochs, the best weights are restored, and training
  stops when it has not improved for ``early_stopping.patience`` evaluations.
* **Weighted objective.** Per-interaction weights (trust and/or recency) scale
  each BPR term. The weights act on the loss; ``HGTConv`` itself cannot consume
  an edge weight, so storing one on the graph does nothing.
* **Trust gate (optional).** Rows flagged by the caller are removed from message
  passing and supervision altogether, at training and at inference.

Shapes:
    user.x     (n_users, d_u)      product.x  (n_items, d_p)
    review.x   (n_reviews, 384+6)  -> fusion -> (n_reviews, 128)
    brand / category                -> learned embeddings
    output     (n_users, 128), (n_items, 128)
    score(u,i) = e_u . e_i
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_geometric.backend
from torch_geometric.nn import HGTConv

from ..features.fusion import SemanticBehaviouralFusion
from ..training.early_stopping import EarlyStopping
from ..training.losses import bpr_loss, normalise_weights
from ..training.negative_sampling import NegativeSampler
from ..utils.logging import get_logger, log_stage
from .base import BaseRecommender

__all__ = ["HGTModel", "HGTRecommender", "configure_backend_for_device"]

# Relations whose edges are one-per-training-row and so encode the very
# interaction being predicted. Reverse relations carry the same rows.
_INTERACTION_RELATIONS = frozenset({"purchased", "wrote", "about"})

logger = get_logger(__name__)


def configure_backend_for_device(device: torch.device) -> None:
    """Disable pyg-lib's fused ``segment_matmul`` when running on MPS.

    Installing ``pyg-lib`` unlocked ``NeighborLoader``, but it also routes
    ``HGTConv``'s typed linear layers through ``pyg::segment_matmul``, which has
    **no MPS kernel**: the forward pass dies with ``NotImplementedError``.
    ``PYTORCH_ENABLE_MPS_FALLBACK`` does not rescue it either, because that
    covers aten operators and this is a custom pyg operator.

    Setting this flag falls back to the loop-based typed-linear path, which is
    slightly slower but runs natively on MPS. The flag is global to PyG, so it
    is set once here rather than per layer.
    """
    if device.type == "mps" and torch_geometric.backend.use_segment_matmul is not False:
        torch_geometric.backend.use_segment_matmul = False
        logger.info(
            "MPS device: disabled pyg-lib segment_matmul (no MPS kernel). "
            "NeighborLoader remains available; only the fused typed-linear path is off."
        )


class HGTModel(nn.Module):
    """Typed projections -> optional fusion -> stacked HGTConv."""

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        feature_dims: dict[str, int],
        featureless_counts: dict[str, int],
        hidden_channels: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        fusion: SemanticBehaviouralFusion | None = None,
        fusion_node_type: str = "review",
    ) -> None:
        super().__init__()
        self.fusion = fusion
        self.fusion_node_type = fusion_node_type
        self.dropout = nn.Dropout(dropout)

        # Featureless types (brand, category) get learned embeddings rather than
        # a hand-made numeric feature that would encode an arbitrary assumption.
        self.embeddings = nn.ModuleDict({
            node_type: nn.Embedding(count, hidden_channels)
            for node_type, count in featureless_counts.items()
        })

        self.projections = nn.ModuleDict()
        for node_type, dim in feature_dims.items():
            width = (fusion.output_dim
                     if fusion is not None and node_type == fusion_node_type else dim)
            self.projections[node_type] = nn.Linear(width, hidden_channels)

        self.convs = nn.ModuleList([
            HGTConv(hidden_channels, hidden_channels, metadata, heads=num_heads)
            for _ in range(num_layers)
        ])
        self.hidden_channels = hidden_channels

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        sbert_dim: int = 0,
        n_id_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            n_id_dict: Global node ids for each type, supplied by a sampled
                mini-batch. Featureless types (brand, category) own one
                embedding row per *global* node, so under neighbour sampling
                they must be gathered by global id; using the full weight matrix
                would silently misalign every brand and category in the batch.
                ``None`` means full-graph mode, where local == global.
        """
        projected: dict[str, torch.Tensor] = {}

        for node_type, features in x_dict.items():
            if (self.fusion is not None and node_type == self.fusion_node_type
                    and sbert_dim > 0):
                features = self.fusion(features[:, :sbert_dim], features[:, sbert_dim:])
            projected[node_type] = self.dropout(
                torch.relu(self.projections[node_type](features))
            )

        for node_type, embedding in self.embeddings.items():
            if n_id_dict is not None and node_type in n_id_dict:
                projected[node_type] = embedding(n_id_dict[node_type])
            else:
                projected[node_type] = embedding.weight

        for conv in self.convs:
            updated = conv(projected, edge_index_dict)
            # HGTConv can return None for a type that received no messages;
            # carry the previous representation forward rather than dropping it.
            projected = {
                node_type: (updated[node_type] if updated.get(node_type) is not None
                            else projected[node_type])
                for node_type in projected
            }
        return projected


class HGTRecommender(BaseRecommender):
    """BaseRecommender wrapper: builds, trains and scores the HGT model."""

    name = "hgt"

    def __init__(
        self,
        n_users: int,
        n_items: int,
        bundle,
        model_config,
        training_config,
        device: str = "cpu",
        use_fusion: bool = True,
        use_weighted_loss: bool = False,
        epochs: int | None = None,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(n_users, n_items, **kwargs)
        self.bundle = bundle
        self.model_config = model_config
        self.training_config = training_config
        self.device = torch.device(device)
        self.use_fusion = use_fusion
        self.use_weighted_loss = use_weighted_loss
        self.seed = seed

        hgt_cfg = model_config.require("hgt")
        self.hidden_channels = int(hgt_cfg.get("hidden_channels", 128))
        self.num_heads = int(hgt_cfg.get("num_heads", 4))
        self.num_layers = int(hgt_cfg.get("num_layers", 2))
        self.dropout = float(hgt_cfg.get("dropout", 0.2))
        self.epochs = int(epochs if epochs is not None
                          else training_config.get("training.max_epochs", 100))
        self.learning_rate = float(training_config.get("optimizer.learning_rate", 1e-3))
        self.weight_decay = float(training_config.get("optimizer.weight_decay", 1e-4))
        self.clip_norm = float(training_config.get("training.gradient_clip_norm", 1.0))
        self.num_negatives = int(training_config.get("negative_sampling.num_negatives", 1))

        self.mask_enabled = bool(training_config.get("training.edge_mask.enabled", True))
        self.keep_prob = float(training_config.get("training.edge_mask.keep_prob", 0.8))
        if not 0.0 < self.keep_prob < 1.0:
            raise ValueError(f"edge_mask.keep_prob must be in (0, 1), got {self.keep_prob}.")

        self.scheduler_name = str(training_config.get("scheduler.name", "none"))
        self.scheduler_floor = float(training_config.get("scheduler.min_lr_factor", 0.02))
        if self.scheduler_name not in ("none", "cosine"):
            raise ValueError(f"scheduler.name must be none|cosine, got '{self.scheduler_name}'.")

        self.eval_every = int(training_config.get("training.eval_every", 5))
        self.stop_enabled = bool(training_config.get("early_stopping.enabled", True))
        self.patience = int(training_config.get("early_stopping.patience", 8))
        self.min_delta = float(training_config.get("early_stopping.min_delta", 1e-4))

        self.full_batch = bool(training_config.get("training.full_batch", False))
        self.batch_size = int(training_config.get("training.batch_size", 4096))
        self.num_neighbors = list(training_config.get("training.num_neighbors", [15, 10]))
        if not self.full_batch and len(self.num_neighbors) < self.num_layers:
            # One fan-out value per propagation hop, or the sampler silently
            # gathers a shallower neighbourhood than the model consumes.
            raise ValueError(
                f"training.num_neighbors has {len(self.num_neighbors)} hops but hgt."
                f"num_layers is {self.num_layers}. Give one fan-out per layer."
            )
        self.history: list[dict[str, Any]] = []
        self.best_epoch: int | None = None
        self.best_val: float | None = None

    # -- construction ------------------------------------------------------
    def _build(self) -> HGTModel:
        data = self.bundle.data
        feature_dims, featureless = {}, {}
        for node_type in data.node_types:
            if "x" in data[node_type]:
                feature_dims[node_type] = int(data[node_type].x.shape[1])
            else:
                featureless[node_type] = int(data[node_type].num_nodes)

        self.sbert_dim = int(self.bundle.metadata.get("review_sbert_dim", 0))
        structured_dim = int(self.bundle.metadata.get("review_structured_dim", 0))

        fusion = None
        if self.use_fusion and self.sbert_dim > 0 and "review" in feature_dims:
            fcfg = self.model_config.require("fusion")
            fusion = SemanticBehaviouralFusion(
                sbert_dim=self.sbert_dim, structured_dim=structured_dim,
                output_dim=int(fcfg.get("output_dimension", 128)),
                hidden_dim=int(fcfg.get("hidden_dimension", 256)),
                dropout=float(fcfg.get("dropout", 0.2)),
                activation=str(fcfg.get("activation", "gelu")),
                layer_norm=bool(fcfg.get("layer_norm", True)),
                mode=str(fcfg.get("mode", "mlp")),
            )
            logger.info("Fusion enabled: %s", fusion.extra_repr())
        elif self.sbert_dim == 0:
            logger.info("No SBERT features in the graph; fusion disabled (ablation A).")

        return HGTModel(
            metadata=data.metadata(), feature_dims=feature_dims,
            featureless_counts=featureless, hidden_channels=self.hidden_channels,
            num_heads=self.num_heads, num_layers=self.num_layers,
            dropout=self.dropout, fusion=fusion,
        ).to(self.device)

    # -- training ----------------------------------------------------------
    def fit(self, train: pd.DataFrame, sampler: NegativeSampler | None = None,
            weights: np.ndarray | None = None, validator=None,
            drop_rows: np.ndarray | None = None, **kwargs) -> "HGTRecommender":
        """Train on ``train``.

        Args:
            weights: Per-interaction loss weights (trust and/or recency), one per
                row of ``train``. Rescaled to mean 1 so that weighting changes
                *where* the gradient goes and not how large it is.
            validator: ``callable(recommender) -> float``, a validation metric
                where higher is better. Enables early stopping and restores the
                best weights. Without it the model trains for ``epochs`` steps.
            drop_rows: Boolean array over ``train``; flagged rows are removed
                from message passing and from supervision (the trust gate).
        """
        if sampler is None:
            raise ValueError("HGTRecommender.fit requires a NegativeSampler.")
        if self.use_weighted_loss and weights is None:
            raise ValueError(
                "use_weighted_loss=True but no weights were supplied. Refusing to "
                "train a 'weighted' model with uniform weights, which would "
                "silently reduce to the plain objective."
            )
        torch.manual_seed(self.seed)
        configure_backend_for_device(self.device)

        # Neighbour sampling builds CSR indices, and
        # aten::_convert_indices_from_coo_to_csr has no MPS kernel (it is an
        # aten op, but one PYTORCH_ENABLE_MPS_FALLBACK does not cover here).
        if not self.full_batch and self.device.type == "mps":
            logger.warning(
                "Mini-batch training requested on MPS, which cannot build the sampler's "
                "CSR indices. Falling back to CPU for this model."
            )
            self.device = torch.device("cpu")

        self.model = self._build()
        data = self.bundle.data.to(self.device)
        self.x_dict = {t: data[t].x for t in data.node_types if "x" in data[t]}
        self._full_edges = {e: data[e].edge_index for e in data.edge_types}

        n_rows = len(train)
        self._interaction_edges = [
            e for e in self._full_edges
            if e[1].replace("rev_", "") in _INTERACTION_RELATIONS
        ]
        for edge_type in self._interaction_edges:
            if self._full_edges[edge_type].shape[1] != n_rows:
                raise ValueError(
                    f"{edge_type} has {self._full_edges[edge_type].shape[1]} edges for "
                    f"{n_rows} training rows; edge masking needs them row-aligned."
                )

        active = torch.ones(n_rows, dtype=torch.bool, device=self.device)
        if drop_rows is not None:
            drop = np.asarray(drop_rows, dtype=bool)
            if drop.shape[0] != n_rows:
                raise ValueError(f"drop_rows has {drop.shape[0]} entries for {n_rows} rows.")
            active = torch.as_tensor(~drop, device=self.device)
            logger.info("Trust gate: %s of %s interactions removed from the graph.",
                        f"{int(drop.sum()):,}", f"{n_rows:,}")
        self._active = active
        self.edge_index_dict = self._select_edges(active)

        optimiser = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
        # Full-batch training takes one optimiser step per epoch, so the schedule
        # spans exactly ``epochs`` steps. Decaying the rate lets the last epoch
        # settle instead of bouncing between the good and bad states a constant
        # rate visits, which matters whenever no validation set picks an epoch.
        self._scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=max(1, self.epochs), eta_min=self.learning_rate * self.scheduler_floor)
            if self.scheduler_name == "cosine" else None)
        users = train["user_idx"].to_numpy(copy=True)
        items = train["item_idx"].to_numpy(copy=True)
        user_t = torch.as_tensor(users, device=self.device)
        item_t = torch.as_tensor(items, device=self.device)

        weight_t = None
        if self.use_weighted_loss:
            weight_t = torch.as_tensor(np.asarray(weights, dtype=np.float32),
                                       device=self.device)
            logger.info("Weighted BPR active (mean raw weight %.4f).", float(weight_t.mean()))

        label = (f"HGT[fusion={self.use_fusion}, weighted={self.use_weighted_loss}, "
                 f"mask={self.keep_prob if self.mask_enabled else 'off'}]")
        if not self.full_batch:
            self._fit_minibatch(data, users, items, sampler, weight_t, optimiser, label)
        else:
            self._fit_full_batch(users, items, user_t, item_t, sampler, weight_t,
                                 optimiser, label, validator)

        self._refresh_embeddings()
        return self

    def _select_edges(self, row_mask: torch.Tensor) -> dict:
        """Edge dict keeping only ``row_mask`` rows of every interaction relation."""
        return {
            edge_type: (index[:, row_mask] if edge_type in self._interaction_edges else index)
            for edge_type, index in self._full_edges.items()
        }

    @torch.no_grad()
    def _refresh_embeddings(self) -> None:
        """Embeddings from the full (unmasked) graph, in eval mode."""
        self.model.eval()
        out = self.model(self.x_dict, self.edge_index_dict, self.sbert_dim)
        self.final_user = out["user"].detach()
        self.final_item = out["product"].detach()
        self.is_fitted = True

    def _fit_full_batch(self, users, items, user_t, item_t, sampler, weights,
                        optimiser, label, validator=None) -> None:
        """One optimiser step per epoch, with edge masking and validation stopping."""
        n_rows = len(users)
        stopper = (EarlyStopping(patience=self.patience, min_delta=self.min_delta)
                   if validator is not None and self.stop_enabled else None)
        best_state = None

        with log_stage(logger, f"Train {label} full-batch, up to {self.epochs} epochs") as stage:
            for epoch in range(1, self.epochs + 1):
                self.model.train()

                if self.mask_enabled:
                    keep = (torch.rand(n_rows, device=self.device) < self.keep_prob) & self._active
                    edges = self._select_edges(keep)
                    supervised = self._active & ~keep
                else:
                    edges = self.edge_index_dict
                    supervised = self._active

                out = self.model(self.x_dict, edges, self.sbert_dim)
                user_emb, item_emb = out["user"], out["product"]

                rows = supervised.nonzero(as_tuple=True)[0]
                u_rows, i_rows = user_t[rows], item_t[rows]
                negatives = torch.as_tensor(
                    sampler.sample(users[rows.cpu().numpy()], self.num_negatives),
                    device=self.device).reshape(len(rows), self.num_negatives)

                pu = user_emb[u_rows]
                pos = (pu * item_emb[i_rows]).sum(1)
                neg = (pu.unsqueeze(1) * item_emb[negatives]).sum(-1)
                row_weights = normalise_weights(weights[rows]) if weights is not None else None
                loss = bpr_loss(pos, neg if self.num_negatives > 1 else neg[:, 0],
                                weights=row_weights)

                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                optimiser.step()
                if self._scheduler is not None:
                    self._scheduler.step()

                record = {"epoch": epoch, "train_loss": float(loss.detach()), "steps": epoch}
                if validator is not None and (epoch % self.eval_every == 0
                                              or epoch == self.epochs):
                    self._refresh_embeddings()
                    value = float(validator(self))
                    record["val_ndcg@10"] = value
                    improved = (stopper.step(value, epoch) if stopper is not None
                                else self.best_val is None or value > self.best_val)
                    if improved:
                        self.best_val, self.best_epoch = value, epoch
                        best_state = {k: v.detach().clone()
                                      for k, v in self.model.state_dict().items()}
                self.history.append(record)

                if epoch % 10 == 0 or epoch == 1:
                    logger.info("  epoch %3d/%d  loss %.5f  val %s", epoch, self.epochs,
                                float(loss.detach()),
                                f"{record['val_ndcg@10']:.4f}" if "val_ndcg@10" in record else "-")
                if stopper is not None and stopper.should_stop:
                    break

            if best_state is not None:
                self.model.load_state_dict(best_state)
            stage["epochs_run"] = len(self.history)
            stage["best_epoch"] = self.best_epoch

    def _fit_minibatch(self, data, users, items, sampler, weights, optimiser,
                       label) -> None:
        """Neighbour-sampled training: many optimiser steps per epoch.

        Note on what is and is not leakage here: the sampled subgraph may
        contain the very edge being scored. That is true of full-batch training
        too, and of LightGCN and every graph recommender that treats the
        training interactions as the graph. It is not test leakage - validation
        and test edges are absent from the graph entirely - but it does mean the
        training loss is optimistic, so early stopping is driven by validation
        NDCG rather than by this loss.
        """
        from torch_geometric.loader import LinkNeighborLoader

        edge_type = ("user", "purchased", "product")
        rng = np.random.default_rng(self.seed)
        n_pairs = len(users)
        pairs_per_batch = max(1, self.batch_size // 2)
        weights_np = weights.detach().cpu().numpy() if weights is not None else None

        with log_stage(logger, f"Train {label} mini-batch, {self.epochs} epochs") as stage:
            total_steps = 0
            for epoch in range(1, self.epochs + 1):
                self.model.train()
                order = rng.permutation(n_pairs)
                negatives = sampler.sample(users, 1).ravel()

                # Interleave [pos_0, neg_0, pos_1, neg_1, ...] and keep the
                # loader unshuffled, so pairing survives batching intact.
                edge_label_index = np.empty((2, 2 * n_pairs), dtype=np.int64)
                edge_label_index[0, 0::2] = users[order]
                edge_label_index[0, 1::2] = users[order]
                edge_label_index[1, 0::2] = items[order]
                edge_label_index[1, 1::2] = negatives[order]

                # edge_label carries the per-pair trust weight (or 1.0), so it
                # travels with its pair through batching instead of being
                # re-derived from indices afterwards.
                pair_weight = (weights_np[order] if weights_np is not None
                               else np.ones(n_pairs, dtype=np.float32))
                edge_label = np.empty(2 * n_pairs, dtype=np.float32)
                edge_label[0::2] = pair_weight
                edge_label[1::2] = pair_weight

                loader = LinkNeighborLoader(
                    data,
                    num_neighbors=self.num_neighbors,
                    edge_label_index=(edge_type, torch.from_numpy(edge_label_index)),
                    edge_label=torch.from_numpy(edge_label),
                    batch_size=2 * pairs_per_batch,
                    shuffle=False,
                    num_workers=0,
                )

                epoch_loss, epoch_steps, seen = 0.0, 0, 0
                for batch in loader:
                    batch = batch.to(self.device)
                    store = batch[edge_type]
                    eli, lbl = store.edge_label_index, store.edge_label
                    n_even = (eli.shape[1] // 2) * 2
                    if n_even < 2:
                        continue
                    eli, lbl = eli[:, :n_even], lbl[:n_even]

                    x_dict = {t: batch[t].x for t in batch.node_types if "x" in batch[t]}
                    n_id_dict = {t: batch[t].n_id for t in batch.node_types
                                 if t in self.model.embeddings}
                    edge_index_dict = {e: batch[e].edge_index for e in batch.edge_types}
                    out = self.model(x_dict, edge_index_dict, self.sbert_dim,
                                     n_id_dict=n_id_dict)

                    user_emb, item_emb = out["user"], out["product"]
                    pos = (user_emb[eli[0, 0::2]] * item_emb[eli[1, 0::2]]).sum(1)
                    neg = (user_emb[eli[0, 1::2]] * item_emb[eli[1, 1::2]]).sum(1)
                    batch_weights = lbl[0::2] if weights_np is not None else None
                    loss = bpr_loss(pos, neg, weights=batch_weights)

                    optimiser.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                    optimiser.step()

                    epoch_loss += float(loss.detach()) * pos.shape[0]
                    seen += pos.shape[0]
                    epoch_steps += 1

                total_steps += epoch_steps
                mean_loss = epoch_loss / max(1, seen)
                self.history.append({"epoch": epoch, "train_loss": mean_loss,
                                     "steps": total_steps})
                if epoch % 5 == 0 or epoch == 1:
                    logger.info("  epoch %3d/%d  loss %.5f  (%d steps, %d cumulative)",
                                epoch, self.epochs, mean_loss, epoch_steps, total_steps)
            stage["optimiser_steps"] = total_steps

    @torch.no_grad()
    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("fit() must be called before score_users().")
        u = torch.as_tensor(np.asarray(user_indices), device=self.device)
        return (self.final_user[u] @ self.final_item.T).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        """Final user and item embeddings, for explanation and confidence."""
        return (self.final_user.cpu().numpy(), self.final_item.cpu().numpy())

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "hidden_channels": self.hidden_channels,
                "full_batch": self.full_batch, "batch_size": self.batch_size,
                "num_neighbors": self.num_neighbors,
                "num_heads": self.num_heads, "num_layers": self.num_layers,
                "epochs_budget": self.epochs, "epochs_run": len(self.history),
                "best_epoch": self.best_epoch, "best_val_ndcg@10": self.best_val,
                "edge_mask_keep_prob": self.keep_prob if self.mask_enabled else None,
                "num_negatives": self.num_negatives, "scheduler": self.scheduler_name,
                "fusion": self.use_fusion, "weighted_loss": self.use_weighted_loss}
