"""Heterogeneous graph container and index bookkeeping (Module 9).

A ``HeteroData`` object alone is not enough to serve recommendations: the model
works in contiguous integer indices while everything a user sees is keyed by
Amazon ids. :class:`GraphBundle` keeps the graph and those mappings together so
they can never drift apart or be saved separately.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import HeteroData

from ..utils.logging import get_logger

__all__ = ["NodeIndex", "GraphBundle"]

logger = get_logger(__name__)


@dataclass
class NodeIndex:
    """Bidirectional map between original ids and contiguous integer indices."""

    ids: list[str] = field(default_factory=list)
    lookup: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_ids(cls, ids) -> "NodeIndex":
        unique = list(dict.fromkeys(str(i) for i in ids))
        return cls(ids=unique, lookup={value: index for index, value in enumerate(unique)})

    def __len__(self) -> int:
        return len(self.ids)

    def encode(self, values) -> np.ndarray:
        """Map ids to indices. Unknown ids become -1 rather than raising."""
        return np.array([self.lookup.get(str(v), -1) for v in values], dtype=np.int64)

    def decode(self, indices) -> list[str]:
        return [self.ids[int(i)] for i in indices]

    def contains(self, value: str) -> bool:
        return str(value) in self.lookup


@dataclass
class GraphBundle:
    """A heterogeneous graph plus everything needed to interpret it."""

    data: HeteroData
    indices: dict[str, NodeIndex] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def n(self, node_type: str) -> int:
        return len(self.indices[node_type]) if node_type in self.indices else 0

    def summary(self) -> str:
        lines = ["Node types:"]
        for node_type in self.data.node_types:
            width = (tuple(self.data[node_type].x.shape[1:])
                     if "x" in self.data[node_type] else "learned embedding")
            lines.append(f"  {node_type:10s} {self.data[node_type].num_nodes:>8,}  features={width}")
        lines.append("Edge types:")
        for edge_type in self.data.edge_types:
            src, rel, dst = edge_type
            lines.append(f"  {src:>8s} --{rel:<16s}--> {dst:<10s} "
                         f"{self.data[edge_type].edge_index.shape[1]:>9,}")
        return "\n".join(lines)

    def validate(self) -> None:
        """Fail loudly on the errors that otherwise surface as silent bad training."""
        for node_type in self.data.node_types:
            store = self.data[node_type]
            if store.num_nodes in (None, 0):
                raise ValueError(f"Node type '{node_type}' has no nodes.")
            if "x" in store:
                if not torch.isfinite(store.x).all():
                    raise ValueError(f"Node type '{node_type}' has non-finite features.")
                if store.x.shape[0] != store.num_nodes:
                    raise ValueError(
                        f"'{node_type}': {store.x.shape[0]} feature rows for "
                        f"{store.num_nodes} nodes."
                    )

        for edge_type in self.data.edge_types:
            src, _, dst = edge_type
            edge_index = self.data[edge_type].edge_index
            if edge_index.numel() == 0:
                logger.warning("Edge type %s is empty.", edge_type)
                continue
            if int(edge_index.min()) < 0:
                raise ValueError(f"{edge_type} contains a negative node index.")
            if int(edge_index[0].max()) >= self.data[src].num_nodes:
                raise ValueError(f"{edge_type} source index exceeds |{src}|.")
            if int(edge_index[1].max()) >= self.data[dst].num_nodes:
                raise ValueError(f"{edge_type} destination index exceeds |{dst}|.")
            weight = self.data[edge_type].get("edge_weight")
            if weight is not None and not torch.isfinite(weight).all():
                raise ValueError(f"{edge_type} has non-finite edge weights.")
        logger.info("Graph validation passed.")

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as handle:
            pickle.dump({"data": self.data, "indices": self.indices,
                         "metadata": self.metadata}, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        out.with_suffix(".meta.json").write_text(
            json.dumps(self.metadata, indent=2, default=str), encoding="utf-8"
        )
        logger.info("Graph saved to %s", out)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "GraphBundle":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        return cls(data=payload["data"], indices=payload["indices"],
                   metadata=payload["metadata"])
