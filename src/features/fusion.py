"""Semantic-Behavioural Feature Fusion Layer (Module 8).

Projects a review's SBERT vector together with its structured behavioural
features into one fixed-width node representation.

Contribution note, kept deliberately narrow: feature fusion is not a new idea
and this module does not claim it is. What the project contributes is one
concrete parameterisation of it for trust-aware recommendation, and - more
importantly - the ablation that asks whether the learned projection beats plain
concatenation at all. ``mode="concat"`` exists precisely so that question can be
answered rather than assumed, and the answer is whatever the experiment says.

Shapes:
    sbert        (N, 384)
    structured   (N, k)      k = len(structured_features), 6 by default
    output       (N, 128)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.logging import get_logger

__all__ = ["SemanticBehaviouralFusion"]

logger = get_logger(__name__)

_ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU}


class SemanticBehaviouralFusion(nn.Module):
    """Two-layer MLP over ``[sbert ; structured]``, or a pass-through concat.

    Args:
        sbert_dim: Width of the semantic vector (384 for all-MiniLM-L6-v2).
        structured_dim: Number of structured behavioural features.
        output_dim: Width of the fused review representation.
        hidden_dim: Width of the hidden layer.
        dropout: Dropout applied after the activation.
        activation: One of gelu, relu, tanh, silu.
        layer_norm: Normalise the concatenated input. Matters here because the
            SBERT block is unit-norm while the structured block is standardised
            but not bounded; without it the 384 semantic dimensions dominate the
            6 behavioural ones purely by count.
        mode: ``mlp`` (learned projection) or ``concat`` (ablation baseline).
    """

    def __init__(
        self,
        sbert_dim: int = 384,
        structured_dim: int = 6,
        output_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        activation: str = "gelu",
        layer_norm: bool = True,
        mode: str = "mlp",
    ) -> None:
        super().__init__()
        if mode not in {"mlp", "concat"}:
            raise ValueError(f"mode must be 'mlp' or 'concat', got '{mode}'.")
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}.")

        self.sbert_dim = sbert_dim
        self.structured_dim = structured_dim
        self.mode = mode
        input_dim = sbert_dim + structured_dim

        self.input_norm = nn.LayerNorm(input_dim) if layer_norm else nn.Identity()

        if mode == "mlp":
            self.output_dim = output_dim
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                _ACTIVATIONS[activation](),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        else:
            # Ablation arm: no learned projection at all, so any difference is
            # attributable to the projection rather than to a width change.
            self.output_dim = input_dim
            self.net = nn.Identity()

    def forward(self, sbert: torch.Tensor, structured: torch.Tensor) -> torch.Tensor:
        """Fuse one batch.

        Args:
            sbert: ``(N, sbert_dim)``
            structured: ``(N, structured_dim)``

        Returns:
            ``(N, output_dim)``
        """
        if sbert.shape[0] != structured.shape[0]:
            raise ValueError(
                f"Batch mismatch: sbert has {sbert.shape[0]} rows, structured has "
                f"{structured.shape[0]}."
            )
        if sbert.shape[1] != self.sbert_dim:
            raise ValueError(f"Expected sbert width {self.sbert_dim}, got {sbert.shape[1]}.")
        if structured.shape[1] != self.structured_dim:
            raise ValueError(
                f"Expected structured width {self.structured_dim}, got {structured.shape[1]}."
            )
        combined = torch.cat([sbert, structured], dim=1)
        return self.net(self.input_norm(combined))

    def extra_repr(self) -> str:
        return (f"mode={self.mode}, in={self.sbert_dim}+{self.structured_dim}, "
                f"out={self.output_dim}")
