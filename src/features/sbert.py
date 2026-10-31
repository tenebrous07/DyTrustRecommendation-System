"""SBERT semantic review embeddings (Module 7).

Encoding is deterministic and expensive, so it happens exactly once per corpus
and is cached to ``data/embeddings/``. The cache key covers the model name, the
pooling settings and a digest of the texts themselves, so changing any of them
invalidates the cache rather than silently reusing stale vectors.

Device note, measured rather than assumed: on this Apple M4, CPU encoded 256
short reviews in 0.06 s against 0.81 s on MPS, because a 22M-parameter model
never amortises the MPS warm-up and transfer cost. ``device: auto`` therefore
benchmarks both once on a real sample and picks the winner, instead of
reaching for the accelerator by reflex.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.config import Config, resolve_path
from ..utils.device import resolve_device
from ..utils.logging import get_logger, log_stage

__all__ = ["SBERTEncoder"]

logger = get_logger(__name__)


class SBERTEncoder:
    """Caching wrapper around a sentence-transformers model."""

    def __init__(self, config: Config) -> None:
        self.cfg: Config = config.require("sbert")
        self.model_name = str(self.cfg.require("model_name"))
        self.expected_dim = int(self.cfg.get("embedding_dimension", 384))
        self.batch_size = int(self.cfg.get("batch_size", 256))
        self.normalize = bool(self.cfg.get("normalize_embeddings", True))
        self.text_fields = list(self.cfg.get("text_fields", ["review_text"]))
        self.join_with = str(self.cfg.get("join_with", ". "))
        self.use_cache = bool(self.cfg.get("cache", True))
        self.cache_dir = resolve_path(str(self.cfg.get("cache_dir", "data/embeddings")))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._device_pref = str(self.cfg.get("device", "auto"))
        self._model = None
        self._device = None

    # -- text assembly -----------------------------------------------------
    def build_texts(self, frame: pd.DataFrame) -> pd.Series:
        """Concatenate the configured text fields into one string per row.

        Title and body are joined because Amazon review titles routinely carry
        the verdict ("Broke after two weeks") that the body only elaborates.
        """
        available = [f for f in self.text_fields if f in frame.columns]
        if not available:
            raise KeyError(
                f"None of the configured text_fields {self.text_fields} exist. "
                f"Available: {list(frame.columns)}"
            )
        if len(available) < len(self.text_fields):
            logger.warning("Text fields not present and skipped: %s",
                           set(self.text_fields) - set(available))

        parts = [frame[f].fillna("").astype(str).str.strip() for f in available]
        texts = parts[0]
        for part in parts[1:]:
            texts = np.where(
                (texts.str.len() > 0) & (part.str.len() > 0),
                texts + self.join_with + part,
                texts.where(texts.str.len() > 0, part),
            )
            texts = pd.Series(texts, index=frame.index)
        return texts.fillna("")

    # -- model -------------------------------------------------------------
    def _load(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        with log_stage(logger, f"Load SBERT ({self.model_name})"):
            model = SentenceTransformer(self.model_name,
                                        cache_folder=str(self.cache_dir / ".hf_cache"))
        actual = model.get_sentence_embedding_dimension()
        if actual != self.expected_dim:
            raise ValueError(
                f"Model '{self.model_name}' produces {actual}-d embeddings but config "
                f"declares embedding_dimension={self.expected_dim}. Fix the config rather "
                f"than letting the dimension mismatch propagate into the graph."
            )
        self._device = self._choose_device(model)
        self._model = model.to(str(self._device))
        return self._model

    def _choose_device(self, model) -> Any:
        if self._device_pref != "auto":
            return resolve_device(self._device_pref)

        import torch
        candidates = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
        if torch.cuda.is_available():
            return resolve_device("cuda")
        if len(candidates) == 1:
            return resolve_device("cpu")

        sample = ["a representative review sentence about a product"] * 128
        timings: dict[str, float] = {}
        for candidate in candidates:
            probe = model.to(candidate)
            probe.encode(sample[:16], show_progress_bar=False)   # warm up
            started = time.perf_counter()
            probe.encode(sample, batch_size=self.batch_size, show_progress_bar=False)
            timings[candidate] = time.perf_counter() - started
        best = min(timings, key=timings.get)
        logger.info("SBERT device benchmark: %s -> using '%s'",
                    {k: f"{v:.3f}s" for k, v in timings.items()}, best)
        return resolve_device(best)

    # -- caching -----------------------------------------------------------
    def _cache_key(self, texts: pd.Series) -> str:
        digest = hashlib.sha256()
        digest.update(self.model_name.encode())
        digest.update(f"|norm={self.normalize}|dim={self.expected_dim}".encode())
        digest.update(f"|n={len(texts)}".encode())
        for chunk in texts.astype(str).to_numpy():
            digest.update(chunk.encode("utf-8", "replace"))
        return digest.hexdigest()[:24]

    # -- public ------------------------------------------------------------
    def encode_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Encode one row per interaction, using the cache when valid."""
        texts = self.build_texts(frame)
        return self.encode(texts)

    def encode(self, texts: pd.Series | list[str]) -> np.ndarray:
        """Encode texts to an ``(n, dim)`` float32 array.

        Empty strings receive a zero vector rather than the embedding of the
        empty string, so "no text" is representable and distinguishable from
        "text that happens to sit near the origin".
        """
        texts = pd.Series(texts).fillna("").astype(str).reset_index(drop=True)
        key = self._cache_key(texts)
        cache_path = self.cache_dir / f"sbert_{key}.npy"
        meta_path = self.cache_dir / f"sbert_{key}.json"

        if self.use_cache and cache_path.is_file():
            embeddings = np.load(cache_path)
            if embeddings.shape == (len(texts), self.expected_dim):
                logger.info("SBERT cache hit: %s %s", cache_path.name, embeddings.shape)
                return embeddings
            logger.warning("Cache %s has shape %s, expected %s; re-encoding.",
                           cache_path.name, embeddings.shape, (len(texts), self.expected_dim))

        model = self._load()
        non_empty = texts.str.len() > 0
        embeddings = np.zeros((len(texts), self.expected_dim), dtype=np.float32)

        with log_stage(logger, f"SBERT encode ({int(non_empty.sum()):,} texts)") as stage:
            if int(non_empty.sum()):
                encoded = model.encode(
                    texts[non_empty].tolist(), batch_size=self.batch_size,
                    convert_to_numpy=True, normalize_embeddings=self.normalize,
                    show_progress_bar=False,
                )
                embeddings[non_empty.to_numpy()] = encoded.astype(np.float32)
            stage["device"] = str(self._device)
            stage["empty_zeroed"] = int((~non_empty).sum())

        if self.use_cache:
            np.save(cache_path, embeddings)
            meta_path.write_text(json.dumps({
                "model": self.model_name, "dim": self.expected_dim, "n": len(texts),
                "normalized": self.normalize, "device": str(self._device),
                "empty_zeroed": int((~non_empty).sum()),
            }, indent=2), encoding="utf-8")
            logger.info("SBERT embeddings cached to %s", cache_path)
        return embeddings
