"""Training objectives (Modules 20 and 31).

BPR is the primary objective. The reasoning, recorded so the choice can be
defended rather than assumed: the feedback here is implicit (a review means the
user engaged, not that they graded their preference), BPR optimises exactly the
pairwise ordering the evaluation metrics measure, and it is the standard
objective for both LightGCN and MF, so every model is trained on like terms.

The trust-weighted variant is the project's concrete methodological extension:

    weighted_loss_e = trust_e * loss_e
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["bpr_loss", "sampled_bce_loss", "normalise_weights"]


def normalise_weights(weights: torch.Tensor) -> torch.Tensor:
    """Rescale weights to mean 1.

    Without this, trust weighting also shrinks the effective learning rate,
    because every gradient is multiplied by a number below 1. Any apparent gain
    from trust weighting would then be confounded with a tuning artefact, and
    the ablation it is meant to inform would be uninterpretable.
    """
    mean = weights.mean()
    return weights / mean if mean > 1e-12 else weights


def bpr_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    weights: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Bayesian Personalised Ranking loss.

        L = -log(sigmoid(s_pos - s_neg))

    Args:
        positive_scores: ``(N,)`` scores for observed interactions.
        negative_scores: ``(N,)`` or ``(N, M)`` scores for sampled negatives.
        weights: Optional ``(N,)`` per-interaction weights (trust).
        reduction: ``mean``, ``sum`` or ``none``.
    """
    if negative_scores.dim() == 2:
        positive_scores = positive_scores.unsqueeze(1).expand_as(negative_scores)
        per_pair = -F.logsigmoid(positive_scores - negative_scores)
        per_example = per_pair.mean(dim=1)
    else:
        per_example = -F.logsigmoid(positive_scores - negative_scores)

    if weights is not None:
        per_example = per_example * weights

    if reduction == "mean":
        return per_example.mean()
    if reduction == "sum":
        return per_example.sum()
    if reduction == "none":
        return per_example
    raise ValueError(f"reduction must be mean|sum|none, got '{reduction}'.")


def sampled_bce_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    weights: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary cross-entropy over sampled positives and negatives.

    Provided as the documented alternative to BPR. It optimises calibration of
    an absolute score rather than a pairwise ordering, which is a worse match
    for ranking metrics, so it is not the default.
    """
    if negative_scores.dim() == 1:
        negative_scores = negative_scores.unsqueeze(1)

    positive_loss = F.binary_cross_entropy_with_logits(
        positive_scores, torch.ones_like(positive_scores), reduction="none"
    )
    negative_loss = F.binary_cross_entropy_with_logits(
        negative_scores, torch.zeros_like(negative_scores), reduction="none"
    ).mean(dim=1)
    per_example = positive_loss + negative_loss

    if weights is not None:
        per_example = per_example * weights

    if reduction == "mean":
        return per_example.mean()
    if reduction == "sum":
        return per_example.sum()
    if reduction == "none":
        return per_example
    raise ValueError(f"reduction must be mean|sum|none, got '{reduction}'.")
