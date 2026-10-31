"""Heterogeneous graph builder (Module 9).

Turns cleaned interactions plus features into a PyG ``HeteroData``.

Two structural decisions, both forced by what Amazon Reviews 2023 actually
contains rather than by preference:

* **There is no Seller node type.** The dataset has no seller field, and the
  project rule is that a missing entity is reported, not invented.
* **Brand is a proxy.** It comes from ``store``, the storefront name, which
  usually but not always matches the manufacturer.

Leakage: the graph is built from ``split.train`` alone. Validation and test
edges never enter it, which is asserted in the tests.

Fusion is deliberately *not* applied here. It is a learned projection, so
baking it into a static graph would freeze random weights at build time. Review
nodes therefore carry the raw ``[sbert || structured]`` concatenation, and the
model applies fusion at forward time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from ..features.product_features import build_product_profiles, extract_category_levels
from ..features.transforms import LeakageSafeScaler
from ..features.user_features import build_user_profiles
from ..utils.config import Config
from ..utils.logging import get_logger, log_stage
from .hetero_graph import GraphBundle, NodeIndex
from .weighting import compute_edge_weights

__all__ = ["HeteroGraphBuilder"]

logger = get_logger(__name__)

# Edge types that exist given this dataset's fields. `sold_by` is absent by design.
EDGE_WROTE = ("user", "wrote", "review")
EDGE_ABOUT = ("review", "about", "product")
EDGE_PURCHASED = ("user", "purchased", "product")
EDGE_IN_CATEGORY = ("product", "belongs_to", "category")
EDGE_BY_BRAND = ("product", "manufactured_by", "brand")


class HeteroGraphBuilder:
    """Assembles the training graph from a chronological split."""

    def __init__(self, config: Config, model_config: Config) -> None:
        self.config = config
        self.model_config = model_config
        self.gcfg: Config = model_config.require("graph")
        self.node_types = list(self.gcfg.get("node_types", []))
        self.add_reverse = bool(self.gcfg.get("add_reverse_edges", True))
        self.include_reviews = bool(self.gcfg.get("include_review_nodes", True))
        self.user_scaler = LeakageSafeScaler()
        self.product_scaler = LeakageSafeScaler()

    def build(
        self,
        train: pd.DataFrame,
        products: pd.DataFrame | None,
        review_embeddings: np.ndarray | None = None,
        review_structured: pd.DataFrame | None = None,
        trust: np.ndarray | None = None,
        temporal: np.ndarray | None = None,
    ) -> GraphBundle:
        """Build the training graph.

        Args:
            train: Training interactions ONLY.
            products: Cleaned product metadata.
            review_embeddings: ``(len(train), 384)`` SBERT vectors, row-aligned
                with ``train``. ``None`` disables semantic features (ablation A).
            review_structured: ``(len(train), k)`` structured review features.
            trust: Per-interaction trust in [0, 1], or ``None``.
            temporal: Per-interaction recency weight, or ``None``.
        """
        data = HeteroData()
        indices: dict[str, NodeIndex] = {}
        metadata: dict[str, Any] = {
            "n_train_interactions": len(train),
            "seller_node": "ABSENT - no seller field exists in Amazon Reviews 2023",
            "brand_node": "PROXY - derived from the `store` (storefront) field",
        }

        with log_stage(logger, "Build heterogeneous graph") as stage:
            indices["user"] = NodeIndex.from_ids(train["user_id"])
            indices["product"] = NodeIndex.from_ids(train["item_id"])
            user_idx = indices["user"].encode(train["user_id"])
            item_idx = indices["product"].encode(train["item_id"])

            self._add_user_nodes(data, train, indices)
            products_resolved = self._add_product_nodes(data, train, products, indices)
            self._add_review_nodes(data, train, review_embeddings, review_structured,
                                   indices, metadata)
            self._add_interaction_edges(data, train, user_idx, item_idx, trust, temporal)
            self._add_review_edges(data, user_idx, item_idx, len(train))
            self._add_attribute_edges(data, products_resolved, indices, metadata)

            if self.add_reverse:
                self._add_reverse_edges(data)

            stage["nodes"] = f"{sum(data[t].num_nodes for t in data.node_types):,}"
            stage["edges"] = f"{sum(data[e].edge_index.shape[1] for e in data.edge_types):,}"

        metadata["node_counts"] = {t: int(data[t].num_nodes) for t in data.node_types}
        metadata["edge_counts"] = {
            "__".join(e): int(data[e].edge_index.shape[1]) for e in data.edge_types
        }
        metadata["user_scaler"] = self.user_scaler.to_dict()
        metadata["product_scaler"] = self.product_scaler.to_dict()

        bundle = GraphBundle(data=data, indices=indices, metadata=metadata)
        bundle.validate()
        logger.info("Graph built:\n%s", bundle.summary())
        return bundle

    # -- nodes -------------------------------------------------------------
    def _add_user_nodes(self, data, train, indices) -> None:
        profiles = build_user_profiles(train).reindex(indices["user"].ids)
        numeric = profiles.select_dtypes(include=[np.number])
        # Fitted on the training frame, which is the only frame this builder
        # ever sees, so the train-only guarantee holds by construction.
        self.user_scaler.fit(numeric, scope="TRAIN SPLIT ONLY (graph build)")
        features = self.user_scaler.transform(numeric).to_numpy(dtype=np.float32)
        data["user"].x = torch.from_numpy(np.nan_to_num(features))
        data["user"].num_nodes = len(indices["user"])

    def _add_product_nodes(self, data, train, products, indices) -> pd.DataFrame | None:
        resolved = None
        if products is not None and not products.empty:
            level = str(self.model_config.get("graph.category_level", "l2"))
            resolved = extract_category_levels(products, node_level=level)

        profiles = build_product_profiles(train, resolved).reindex(indices["product"].ids)
        numeric = profiles.select_dtypes(include=[np.number])
        self.product_scaler.fit(numeric, scope="TRAIN SPLIT ONLY (graph build)")
        features = self.product_scaler.transform(numeric).to_numpy(dtype=np.float32)
        data["product"].x = torch.from_numpy(np.nan_to_num(features))
        data["product"].num_nodes = len(indices["product"])
        return resolved

    def _add_review_nodes(self, data, train, embeddings, structured, indices, metadata) -> None:
        if not self.include_reviews:
            metadata["review_nodes"] = "disabled by config (ablation)"
            return

        n = len(train)
        indices["review"] = NodeIndex.from_ids([f"r{i}" for i in range(n)])
        blocks: list[np.ndarray] = []

        if embeddings is not None:
            if embeddings.shape[0] != n:
                raise ValueError(
                    f"review_embeddings has {embeddings.shape[0]} rows for {n} interactions."
                )
            blocks.append(embeddings.astype(np.float32))
            metadata["review_sbert_dim"] = int(embeddings.shape[1])
        else:
            metadata["review_sbert_dim"] = 0

        if structured is not None:
            if len(structured) != n:
                raise ValueError(
                    f"review_structured has {len(structured)} rows for {n} interactions."
                )
            blocks.append(structured.to_numpy(dtype=np.float32))
            metadata["review_structured_dim"] = int(structured.shape[1])
            metadata["review_structured_columns"] = list(structured.columns)
        else:
            metadata["review_structured_dim"] = 0

        if blocks:
            features = np.nan_to_num(np.concatenate(blocks, axis=1))
        else:
            # Ablation A: no semantics, no structured features. A single
            # constant column keeps the type present so the graph topology is
            # identical across arms and only the features differ.
            features = np.ones((n, 1), dtype=np.float32)
            metadata["review_features"] = "constant (ablation A: no SBERT, no structured)"

        data["review"].x = torch.from_numpy(features.astype(np.float32))
        data["review"].num_nodes = n

    # -- edges -------------------------------------------------------------
    def _add_interaction_edges(self, data, train, user_idx, item_idx, trust, temporal) -> None:
        edge_index = torch.from_numpy(np.stack([user_idx, item_idx]))
        data[EDGE_PURCHASED].edge_index = edge_index
        weights = compute_edge_weights(train, self.model_config, trust=trust, temporal=temporal)
        data[EDGE_PURCHASED].edge_weight = torch.from_numpy(weights)
        data[EDGE_PURCHASED].edge_time = torch.from_numpy(
            train["timestamp"].to_numpy(dtype=np.int64)
        )

    def _add_review_edges(self, data, user_idx, item_idx, n) -> None:
        if not self.include_reviews:
            return
        review_idx = np.arange(n, dtype=np.int64)
        data[EDGE_WROTE].edge_index = torch.from_numpy(np.stack([user_idx, review_idx]))
        data[EDGE_ABOUT].edge_index = torch.from_numpy(np.stack([review_idx, item_idx]))

    def _add_attribute_edges(self, data, products, indices, metadata) -> None:
        if products is None or products.empty:
            metadata["attribute_edges"] = "skipped - no product metadata available"
            return

        known = set(indices["product"].ids)
        subset = products[products["item_id"].astype(str).isin(known)]

        for column, node_type, edge_type in (
            ("category_node", "category", EDGE_IN_CATEGORY),
            ("brand", "brand", EDGE_BY_BRAND),
        ):
            if column not in subset.columns:
                continue
            pairs = subset[["item_id", column]].dropna()
            if pairs.empty:
                logger.warning("No usable values for '%s'; %s edges skipped.", column, node_type)
                continue
            indices[node_type] = NodeIndex.from_ids(pairs[column])
            src = indices["product"].encode(pairs["item_id"])
            dst = indices[node_type].encode(pairs[column])
            keep = (src >= 0) & (dst >= 0)
            data[node_type].num_nodes = len(indices[node_type])
            data[edge_type].edge_index = torch.from_numpy(np.stack([src[keep], dst[keep]]))
            logger.info("  %s: %s nodes, %s edges", node_type,
                        f"{len(indices[node_type]):,}", f"{int(keep.sum()):,}")

    @staticmethod
    def _add_reverse_edges(data) -> None:
        """Add the inverse of every relation.

        Not optional in practice: without a reverse relation, HGT can push
        information from users into products but never back, so user embeddings
        would never see the items they interacted with.
        """
        for src, rel, dst in list(data.edge_types):
            reverse = (dst, f"rev_{rel}", src)
            if reverse in data.edge_types:
                continue
            edge_index = data[(src, rel, dst)].edge_index
            data[reverse].edge_index = edge_index.flip(0)
            weight = data[(src, rel, dst)].get("edge_weight")
            if weight is not None:
                data[reverse].edge_weight = weight
