#!/usr/bin/env python
"""Phases 3-5: split, features, SBERT embeddings, heterogeneous graph.

    python scripts/build_graph.py
    python scripts/build_graph.py --no-sbert        # ablation A features

Outputs:
    data/processed/split_{train,val,test}.parquet
    data/processed/review_features_train.parquet
    data/embeddings/sbert_*.npy
    data/processed/graph_train.pkl
    reports/results/build_graph_run.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.splitter import assert_no_leakage, chronological_split  # noqa: E402
from src.features.review_features import ReviewFeatureBuilder  # noqa: E402
from src.features.sbert import SBERTEncoder  # noqa: E402
from src.features.sentiment import add_sentiment_features  # noqa: E402
from src.graph.builder import HeteroGraphBuilder  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402
from src.utils.seed import set_seed, write_run_manifest  # noqa: E402

logger = get_logger("build_graph")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DyTrustRec Phases 3-5.")
    parser.add_argument("--no-sbert", action="store_true",
                        help="Skip semantic features (ablation A).")
    parser.add_argument("--no-sentiment", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    data_config = load_config("data_config")
    model_config = load_config("model_config")

    results = resolve_path(data_config.require("paths.results"))
    results.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level, log_file=results / "build_graph.log")
    seed = set_seed(int(model_config.get("seed", 42)))

    processed = resolve_path(data_config.require("paths.processed"))
    interactions = pd.read_parquet(processed / "interactions_clean.parquet")
    products_path = processed / "products_clean.parquet"
    products = pd.read_parquet(products_path) if products_path.is_file() else None
    logger.info("Loaded %s interactions", f"{len(interactions):,}")

    # ---- Phase 3a: split BEFORE any statistic is fitted --------------------
    split = chronological_split(interactions, data_config)
    assert_no_leakage(split)
    for name, frame in (("train", split.train), ("val", split.val), ("test", split.test)):
        frame.to_parquet(processed / f"split_{name}.parquet", index=False)

    # ---- Phase 3b: sentiment (frozen/inference-only, so split-agnostic) ----
    train = split.train
    if not args.no_sentiment:
        train = add_sentiment_features(train, model_config=model_config)

    # ---- Phase 3c: review features, fitted on train only -------------------
    temporal_cfg = model_config.require("temporal")
    review_builder = ReviewFeatureBuilder(
        lambda_per_year=float(temporal_cfg.get("lambda_per_year", 0.5)),
        min_weight=float(temporal_cfg.get("min_weight", 0.01)),
    ).fit(split.train)
    review_features = review_builder.transform(train)
    review_features.to_parquet(processed / "review_features_train.parquet", index=False)

    structured_columns = list(model_config.get("fusion.structured_features", []))
    available = [c for c in structured_columns if c in review_features.columns]
    missing = [c for c in structured_columns if c not in review_features.columns]
    if missing:
        # A silently dropped feature changes the model's input width without
        # changing anything visible, so this is fatal rather than a warning.
        raise SystemExit(
            f"fusion.structured_features names {missing}, which ReviewFeatureBuilder does "
            f"not produce. Available: {list(review_features.columns)}. Fix "
            f"configs/model_config.yaml rather than training on a narrower input than "
            f"the config claims."
        )
    structured = review_features[available]

    # ---- Phase 4: SBERT ----------------------------------------------------
    embeddings = None
    if not args.no_sbert:
        encoder = SBERTEncoder(model_config)
        embeddings = encoder.encode_frame(train)
        logger.info("SBERT embeddings: %s", embeddings.shape)

    # ---- Phase 5: graph ----------------------------------------------------
    builder = HeteroGraphBuilder(data_config, model_config)
    bundle = builder.build(
        train=train, products=products,
        review_embeddings=embeddings, review_structured=structured,
        trust=None,                                   # supplied in Phase 8/9
        temporal=review_features["recency_weight"].to_numpy(),
    )
    bundle.metadata["split_stats"] = split.stats
    bundle.metadata["review_feature_builder"] = review_builder.to_dict()
    bundle.metadata["sbert_enabled"] = embeddings is not None
    bundle.save(processed / "graph_train.pkl")

    write_run_manifest(results / "build_graph_run.json", seed=seed,
                       config={"data": data_config.to_dict(), "model": model_config.to_dict()},
                       extra={"phase": "3-5-features-and-graph",
                              "split_stats": split.stats,
                              "graph_metadata": bundle.metadata})
    logger.info("Phases 3-5 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
