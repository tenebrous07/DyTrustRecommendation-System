"""Configuration loading and path resolution.

Every experiment in DyTrustRec is driven by YAML under ``configs/``. Nothing in
``src/`` may hard-code a path, a threshold, or a hyperparameter. This module is
the single place that knows where the project root is and how to read config.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

__all__ = [
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "Config",
    "load_config",
    "resolve_path",
]


def _find_project_root() -> Path:
    """Locate the project root by walking up for the ``configs/`` marker.

    Honours ``DYTRUSTREC_ROOT`` so the package works when installed elsewhere
    or invoked from a notebook in a different working directory.
    """
    env = os.environ.get("DYTRUSTREC_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"DYTRUSTREC_ROOT does not exist: {root}")
        return root

    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate the DyTrustRec project root. Set DYTRUSTREC_ROOT."
    )


PROJECT_ROOT: Path = _find_project_root()
CONFIG_DIR: Path = PROJECT_ROOT / "configs"


class Config(Mapping[str, Any]):
    """Read-only nested mapping with attribute access and dotted lookup.

    Wrapping the raw dict buys three things over passing dicts around:
    a clear error when a key is missing, immutability so a downstream module
    cannot silently mutate shared config, and ``get("a.b.c", default)``.
    """

    __slots__ = ("_data", "_name")

    def __init__(self, data: Mapping[str, Any], name: str = "config") -> None:
        object.__setattr__(self, "_data", dict(data))
        object.__setattr__(self, "_name", name)

    # -- Mapping protocol ---------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        try:
            value = self._data[key]
        except KeyError:
            raise KeyError(
                f"'{key}' not found in {self._name}. "
                f"Available: {sorted(self._data)}"
            ) from None
        return Config(value, name=f"{self._name}.{key}") if isinstance(value, dict) else value

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # -- Ergonomics ---------------------------------------------------------
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(str(exc)) from None

    def __setattr__(self, key: str, value: Any) -> None:
        raise TypeError("Config is immutable; use with_overrides() instead.")

    def __repr__(self) -> str:
        return f"Config({self._name}, keys={sorted(self._data)})"

    def get(self, key: str, default: Any = None) -> Any:
        """Fetch by plain or dotted key, returning ``default`` if absent."""
        node: Any = self._data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return Config(node, name=key) if isinstance(node, dict) else node

    def require(self, key: str) -> Any:
        """Fetch by dotted key, raising if absent or None."""
        sentinel = object()
        value = self.get(key, sentinel)
        if value is sentinel or value is None:
            raise KeyError(f"Required config key '{key}' is missing from {self._name}.")
        return value

    def to_dict(self) -> dict[str, Any]:
        """Deep copy as a plain dict, for logging or serialising to a run manifest."""
        return copy.deepcopy(self._data)

    def with_overrides(self, overrides: Mapping[str, Any]) -> "Config":
        """Return a new Config with dotted-key overrides applied.

        Used by scripts to accept ``--set active_tier=smoke`` without editing YAML.
        """
        merged = copy.deepcopy(self._data)
        for dotted, value in overrides.items():
            parts = dotted.split(".")
            node = merged
            for part in parts[:-1]:
                node = node.setdefault(part, {})
                if not isinstance(node, dict):
                    raise TypeError(f"Cannot override '{dotted}': '{part}' is not a mapping.")
            node[parts[-1]] = value
        return Config(merged, name=self._name)


def load_config(name: str, overrides: Mapping[str, Any] | None = None) -> Config:
    """Load ``configs/<name>.yaml``.

    Args:
        name: Config stem, e.g. ``"data_config"``. A ``.yaml`` suffix is optional.
        overrides: Optional dotted-key overrides applied after loading.

    Returns:
        An immutable :class:`Config`.
    """
    stem = name[:-5] if name.endswith(".yaml") else name
    path = CONFIG_DIR / f"{stem}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"No config '{stem}' in {CONFIG_DIR}. Available: {available}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping at the top level, got {type(data).__name__}.")

    config = Config(data, name=stem)
    if stem == "data_config":
        config = config.with_overrides(dataset_profile_overrides())
    return config.with_overrides(overrides) if overrides else config


# Second-dataset profiles. ``DYTRUSTREC_DATASET=Musical_Instruments`` (any Amazon Reviews
# 2023 category) points every script at that category without a code change, using the
# same sampling and 5-core rules as the Video_Games dev tier. Each profile writes to its own
# directories, so the Video_Games artefacts are never touched.
#
# All_Beauty was tried first and rejected: 1,620 of 631,986 users have 5+ reviews, which
# leaves ~2,500 rows after k-core (~9,000 even at 3-core), far too few to compare methods on.
_BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"


def _category_profile(category: str) -> dict[str, Any]:
    slug = category.lower()
    return {
        "active_tier": "dev",
        "tiers.dev.category": category,
        "tiers.dev.reviews_url": f"{_BASE}/review_categories/{category}.jsonl",
        "tiers.dev.meta_url": f"{_BASE}/meta_categories/meta_{category}.jsonl",
        "paths.interim": f"data/interim_{slug}",
        "paths.processed": f"data/processed_{slug}",
        "paths.results": f"reports/results_{slug}",
        "paths.tables": f"reports/tables_{slug}",
        "paths.checkpoints": f"checkpoints_{slug}",
    }


def dataset_profile_overrides() -> dict[str, Any]:
    """Overrides for the profile named by ``DYTRUSTREC_DATASET`` (empty when unset)."""
    import os

    category = os.environ.get("DYTRUSTREC_DATASET", "").strip()
    if not category or category.lower() == "video_games":
        return {}
    if not category.replace("_", "").isalnum():
        raise ValueError(f"DYTRUSTREC_DATASET must be an Amazon 2023 category name, got '{category}'.")
    return _category_profile(category)


def active_dirs() -> dict[str, Path]:
    """Results, tables and checkpoint directories for the active dataset profile.

    Scripts use this instead of spelling out ``reports/results`` so that a second dataset
    can never read, skip over or overwrite the Video_Games runs.
    """
    paths = load_config("data_config")
    return {
        "results": resolve_path(paths.require("paths.results")),
        "tables": resolve_path(paths.require("paths.tables")),
        "checkpoints": resolve_path(paths.get("paths.checkpoints", "checkpoints")),
    }


def resolve_path(relative: str | Path, *, create_parent: bool = False) -> Path:
    """Resolve a config-declared path against the project root.

    Absolute paths pass through untouched so a user can point at an external
    data volume without editing code.
    """
    path = Path(relative).expanduser()
    resolved = path if path.is_absolute() else (PROJECT_ROOT / path)
    if create_parent:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
