"""Compute-device selection.

Added beyond the base module list because this project targets Apple Silicon,
where device choice is not a one-liner: MPS is available and fast, but a small
number of PyTorch operators still fall back or fail on it. Centralising the
decision means a single config switch (``device: auto|mps|cuda|cpu``) can move
the whole pipeline to CPU when an operator misbehaves, instead of hunting
``.to("mps")`` calls through the codebase.
"""

from __future__ import annotations

import os
from typing import Literal

import torch

from .logging import get_logger

__all__ = ["resolve_device", "device_report", "DeviceSpec"]

logger = get_logger(__name__)

DeviceSpec = Literal["auto", "cpu", "cuda", "mps"]


def resolve_device(preference: DeviceSpec | str = "auto") -> torch.device:
    """Resolve a config string to a concrete :class:`torch.device`.

    ``auto`` prefers CUDA, then MPS, then CPU. An explicit request for an
    unavailable backend warns and falls back rather than crashing a long run.
    """
    pref = str(preference).lower()

    if pref == "auto":
        if torch.cuda.is_available():
            chosen = torch.device("cuda")
        elif torch.backends.mps.is_available():
            chosen = torch.device("mps")
        else:
            chosen = torch.device("cpu")
    elif pref == "cuda":
        if not torch.cuda.is_available():
            logger.warning("device='cuda' requested but CUDA is unavailable; using CPU.")
            chosen = torch.device("cpu")
        else:
            chosen = torch.device("cuda")
    elif pref == "mps":
        if not torch.backends.mps.is_available():
            logger.warning("device='mps' requested but MPS is unavailable; using CPU.")
            chosen = torch.device("cpu")
        else:
            chosen = torch.device("mps")
    elif pref == "cpu":
        chosen = torch.device("cpu")
    else:
        raise ValueError(f"Unknown device preference '{preference}'. Use auto|cpu|cuda|mps.")

    if chosen.type == "mps":
        # Lets an unimplemented MPS operator silently run on CPU instead of
        # aborting training. Costs a host<->device copy; far better than a crash
        # twenty minutes into an epoch.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    logger.info("Compute device: %s", chosen)
    return chosen


def device_report() -> dict[str, object]:
    """Backend availability, for the run manifest and the dashboard overview."""
    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_cpu_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
        "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
    }
