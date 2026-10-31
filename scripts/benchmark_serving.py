#!/usr/bin/env python
"""How fast is serving, and where does brute force stop being enough?

Serving a trained embedding model is a dot product against the catalogue followed by a
top-k. That is cheap at this catalogue size and linear in it, so this measures the
real latency here and the same operation at larger synthetic catalogues, to show where an
approximate-nearest-neighbour index becomes necessary. Training cost per run comes from the
saved run records. Synthetic vectors are used only for the larger sizes and are labelled.

    python scripts/benchmark_serving.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import active_dirs  # noqa: E402

DIRS = active_dirs()

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def top_k_latency(users: np.ndarray, items: np.ndarray, k: int = 10, batch: int = 1) -> float:
    """Milliseconds per user for a dot-product top-k, batch size ``batch``."""
    started = time.perf_counter()
    done = 0
    for start in range(0, len(users), batch):
        scores = users[start:start + batch] @ items.T
        np.argpartition(-scores, k - 1, axis=1)[:, :k]
        done += scores.shape[0]
    return 1000 * (time.perf_counter() - started) / done


def main() -> int:
    saved = np.load(DIRS["checkpoints"] / "dytrustrec_embeddings.npz", allow_pickle=True)
    users, items = saved["user_emb"], saved["item_emb"]
    rng = np.random.default_rng(0)
    sample = users[rng.choice(len(users), size=500, replace=False)]

    rows = [{"setting": "this catalogue (real embeddings)", "items": len(items),
             "ms_per_user_single": top_k_latency(sample, items, batch=1),
             "ms_per_user_batch256": top_k_latency(sample, items, batch=256)}]
    for n in (100_000, 1_000_000):
        fake = rng.standard_normal((n, items.shape[1]), dtype=np.float32)
        probe = sample[:100 if n > 100_000 else 500]
        rows.append({"setting": "synthetic catalogue (random vectors)", "items": n,
                     "ms_per_user_single": top_k_latency(probe, fake, batch=1),
                     "ms_per_user_batch256": top_k_latency(probe, fake, batch=100)})

    runs = [json.loads(p.read_text()) for p in (DIRS["results"] / "runs").glob("*__seed*.json")]
    train = {}
    for r in runs:
        train.setdefault(r["model"], []).append(r["train_seconds"])
    result = {"serving": rows, "embedding_bytes_per_user": int(users.shape[1] * 4),
              "mean_train_seconds": {k: float(np.mean(v)) for k, v in sorted(train.items())},
              "hardware": "Apple M3 Pro, 12 cores, 18 GB, CPU numpy for serving"}
    (DIRS["results"] / "serving_benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = ["# Serving and training cost", "", f"Hardware: {result['hardware']}.", "",
             "| Catalogue | items | ms per user (one at a time) | ms per user (batched) |", "| --- | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['setting']} | {r['items']:,} | {r['ms_per_user_single']:.2f} | {r['ms_per_user_batch256']:.3f} |")
    lines += ["", f"One user's embedding is {result['embedding_bytes_per_user']} bytes, so 1 million users is "
                  f"{result['embedding_bytes_per_user'] * 1_000_000 / 1e9:.2f} GB.", "",
              "Mean training time per run, seconds (includes validation passes):", "",
              "| Model | seconds |", "| --- | ---: |"]
    for name, seconds in result["mean_train_seconds"].items():
        lines.append(f"| {name} | {seconds:.0f} |")
    (DIRS["tables"] / "serving_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
