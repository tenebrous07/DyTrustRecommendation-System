#!/usr/bin/env python
"""Export the embeddings the dashboard serves.

Every experiment run saves its final user and product embeddings next to its
result, so exporting is a copy, not a re-run: no model is rebuilt and no GPU is
needed. The arm that is served is ``serve_arm`` in ``configs/experiment_config.yaml``
(the best variant), at the configured seed.

    python scripts/export_model.py
    python scripts/export_model.py --arm I_dytrustrec_cosine --seed 43
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import active_dirs  # noqa: E402

DIRS = active_dirs()

import numpy as np  # noqa: E402

from src.experiments.context import ExperimentContext  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger("export_model")
ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export dashboard embeddings.")
    parser.add_argument("--arm", default=None, help="Run name, e.g. E_hgt_sbert_trust_recency.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    setup_logging("INFO")

    # Only the splits and the vocabulary are needed, so skip building a device context.
    context = ExperimentContext(device="cpu")
    arm = args.arm or str(context.experiment_config.get("serve_arm", "E_hgt_sbert_trust_recency"))
    seed = args.seed if args.seed is not None else int(context.experiment_config.get("seed", 42))

    source = DIRS["results"] / "runs" / f"{arm}__seed{seed}.emb.npz"
    if not source.is_file():
        raise SystemExit(f"{source.name} not found. Run scripts/run_experiments.py for arm {arm} first.")
    with np.load(source) as saved:
        user_emb = saved["user"].astype(np.float32)
        item_emb = saved["item"].astype(np.float32)
    if user_emb.shape[0] != context.vocab.n_users or item_emb.shape[0] != context.vocab.n_items:
        raise SystemExit("Saved embeddings do not match the training vocabulary.")

    output = DIRS["checkpoints"] / "dytrustrec_embeddings.npz"
    output.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        output, user_emb=user_emb, item_emb=item_emb,
        user_ids=np.array(context.vocab.user_ids, dtype=object),
        item_ids=np.array(context.vocab.item_ids, dtype=object),
        source=np.array(f"{arm} seed {seed}"),
    )
    logger.info("Exported %s from %s: user %s, item %s", output.name, source.name,
                user_emb.shape, item_emb.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
