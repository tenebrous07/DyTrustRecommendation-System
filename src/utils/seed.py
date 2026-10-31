"""Reproducibility: seeding and run fingerprinting.

Two responsibilities, per project requirement 36:

1. :func:`set_seed` makes a run repeatable.
2. :func:`environment_fingerprint` records *what* was run, so a number in
   ``reports/`` can always be traced back to code, data and hardware.

Honesty note: seeding does not buy bit-exact reproducibility on every backend.
Apple MPS in particular does not expose deterministic-algorithm guarantees the
way CUDA does, and some reduction kernels are order-dependent. We therefore
seed everything we can, record the backend, and describe the result as
*reproducible up to backend non-determinism* rather than claiming bit-exactness.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .logging import get_logger

__all__ = ["set_seed", "environment_fingerprint", "write_run_manifest", "hash_file"]

logger = get_logger(__name__)

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, *, deterministic: bool = True) -> int:
    """Seed Python, NumPy and PyTorch (all devices).

    Args:
        seed: The seed value.
        deterministic: Request deterministic kernels where the backend supports
            it. Ignored with a warning on MPS, which has no such guarantee.

    Returns:
        The seed, so callers can log it in one line.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if deterministic and torch.backends.mps.is_available():
        logger.debug(
            "MPS backend active: deterministic kernels are not guaranteed. "
            "Runs are reproducible up to backend non-determinism."
        )
    logger.info("Seed set to %d (deterministic=%s)", seed, deterministic)
    return seed


def _package_versions() -> dict[str, str]:
    """Version of every library whose behaviour could change a result."""
    import importlib.metadata as md

    tracked = [
        "torch", "torch_geometric", "pyg-lib", "numpy", "pandas", "scipy",
        "scikit-learn", "networkx", "sentence-transformers", "transformers",
        "nltk", "pyarrow", "matplotlib", "plotly", "streamlit",
    ]
    versions: dict[str, str] = {}
    for name in tracked:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _git_revision() -> str:
    """Short commit hash, or a clear marker when not under version control."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=Path(__file__).parent,
        )
        if out.returncode != 0:
            return "not-a-git-repository"
        rev = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, cwd=Path(__file__).parent,
        ).stdout.strip()
        return f"{rev}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
    if sys.platform == "darwin":
        for key, sysctl in (("cpu_brand", "machdep.cpu.brand_string"),
                            ("memory_bytes", "hw.memsize")):
            try:
                res = subprocess.run(["sysctl", "-n", sysctl],
                                     capture_output=True, text=True, timeout=5)
                if res.returncode == 0:
                    value = res.stdout.strip()
                    info[key] = int(value) if value.isdigit() else value
            except (OSError, subprocess.SubprocessError):
                pass
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
    return info


def environment_fingerprint(seed: int | None = None) -> dict[str, Any]:
    """Everything needed to explain how a result was produced."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "python": sys.version.split()[0],
        "git_revision": _git_revision(),
        "hardware": _hardware(),
        "packages": _package_versions(),
    }


def hash_file(path: str | Path, *, chunk_size: int = 1 << 20, max_bytes: int | None = None) -> str:
    """SHA-256 of a file, used to pin the exact data a run consumed.

    Args:
        max_bytes: Hash only the first N bytes. For multi-GB raw dumps a prefix
            hash is enough to detect a changed file at negligible cost; the
            returned digest is suffixed so a partial hash is never mistaken
            for a full one.
    """
    digest = hashlib.sha256()
    read = 0
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            if max_bytes is not None and read + len(chunk) > max_bytes:
                digest.update(chunk[: max_bytes - read])
                return f"{digest.hexdigest()}-partial{max_bytes}"
            digest.update(chunk)
            read += len(chunk)
    return digest.hexdigest()


def write_run_manifest(
    path: str | Path,
    *,
    seed: int | None = None,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a JSON manifest beside a run's outputs."""
    manifest = {"environment": environment_fingerprint(seed)}
    if config is not None:
        manifest["config"] = config
    if extra is not None:
        manifest.update(extra)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    logger.info("Run manifest written to %s", out)
    return out
