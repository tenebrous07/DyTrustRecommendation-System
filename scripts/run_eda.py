#!/usr/bin/env python
"""Phase 2 entrypoint: exploratory analysis, outlier indicators, graph statistics.

Consumes the cleaned artefacts produced by ``scripts/prepare_data.py`` and
writes figures and tables under ``reports/``. Runs no model and trains nothing.

Examples:
    python scripts/run_eda.py
    python scripts/run_eda.py --skip-graph                  # fastest pass
    python scripts/run_eda.py --set graph.betweenness.k_samples=100
    python scripts/run_eda.py --sample-users 5000           # quick look

Outputs:
    reports/figures/*.png, *.html
    reports/tables/eda_summary.{json,md}, eda_*.csv
    reports/tables/outlier_report.{json,md}, outlier_scores.csv
    reports/tables/graph_stats.{json,md}, graph_top_*.csv
    reports/results/run_eda_run.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.analytics.eda import ExploratoryAnalysis                # noqa: E402
from src.analytics.graph_stats import GraphAnalyzer, build_bipartite_graph  # noqa: E402
from src.analytics.outliers import OutlierDetector               # noqa: E402
from src.features.user_features import build_user_profiles       # noqa: E402
from src.utils.config import load_config, resolve_path           # noqa: E402
from src.utils.logging import get_logger, setup_logging          # noqa: E402
from src.utils.seed import set_seed, write_run_manifest          # noqa: E402

logger = get_logger("run_eda")


def _parse_overrides(pairs: list[str]) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got '{pair}'")
        key, _, raw = pair.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        overrides[key.strip()] = value
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="DyTrustRec Phase 2 - EDA, outlier indicators, graph analytics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="analytics_config")
    parser.add_argument("--data-config", default="data_config")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="Dotted-key override. Repeatable.")
    parser.add_argument("--sample-users", type=int, default=None,
                        help="Analyse a random subset of users (for a quick look).")
    parser.add_argument("--skip-graph", action="store_true",
                        help="Skip graph analytics, which dominates runtime.")
    parser.add_argument("--skip-outliers", action="store_true")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    data_config = load_config(args.data_config)
    config = load_config(args.config, _parse_overrides(args.overrides))
    # `paths` is declared once, in data_config, and injected here so the
    # analytics modules never carry a second copy that could drift out of sync.
    config = config.with_overrides({"paths": data_config.require("paths").to_dict()})

    results_dir = resolve_path(config.get("paths.results", "reports/results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level, log_file=results_dir / "run_eda.log")

    seed = int(config.get("seed", 42))
    set_seed(seed)

    processed = resolve_path(config.require("paths.processed"))
    interactions_path = processed / "interactions_clean.parquet"
    products_path = processed / "products_clean.parquet"

    if not interactions_path.is_file():
        raise SystemExit(
            f"{interactions_path} not found. Run scripts/prepare_data.py first."
        )

    interactions = pd.read_parquet(interactions_path)
    products = pd.read_parquet(products_path) if products_path.is_file() else None
    logger.info("Loaded %s interactions, %s products",
                f"{len(interactions):,}", f"{0 if products is None else len(products):,}")

    if args.sample_users:
        chosen = (interactions["user_id"].drop_duplicates()
                  .sample(min(args.sample_users, interactions["user_id"].nunique()),
                          random_state=seed))
        interactions = interactions[interactions["user_id"].isin(set(chosen))]
        logger.warning(
            "--sample-users %d active: analysing %s interactions from %s users. "
            "Figures from this run are a preview, not the dataset's statistics.",
            args.sample_users, f"{len(interactions):,}", f"{interactions.user_id.nunique():,}",
        )

    manifest_extra: dict[str, object] = {"phase": "2-eda-outliers-graph"}

    # ---- EDA -------------------------------------------------------------
    eda = ExploratoryAnalysis(interactions, products, config)
    eda_result = eda.run()
    eda.save()
    manifest_extra["eda_summary"] = eda_result.summary

    # ---- Outlier indicators ----------------------------------------------
    if not args.skip_outliers:
        profiles = eda.user_profiles if not eda.user_profiles.empty \
            else build_user_profiles(interactions)
        detector = OutlierDetector(config)
        scope = "all cleaned interactions (exploratory; NOT train-only)"
        detector.fit(profiles, scope=scope)
        outlier_report = detector.score(profiles)
        detector.save(outlier_report)
        manifest_extra["outlier_summary"] = outlier_report.summary
    else:
        logger.info("--skip-outliers set.")

    # ---- Graph analytics --------------------------------------------------
    if not args.skip_graph and bool(config.get("graph.enabled", True)):
        graph = build_bipartite_graph(interactions)
        analyzer = GraphAnalyzer(graph, config, bipartite=True)
        graph_report = analyzer.run()
        analyzer.save()
        manifest_extra["graph_metrics"] = graph_report.metrics
        manifest_extra["graph_timings_seconds"] = graph_report.timings
    else:
        logger.info("--skip-graph set or graph.enabled=false.")

    write_run_manifest(results_dir / "run_eda_run.json", seed=seed,
                       config=config.to_dict(), extra=manifest_extra)

    logger.info("=" * 72)
    logger.info("Phase 2 complete.")
    logger.info("  figures -> %s", resolve_path(config.require("paths.figures")))
    logger.info("  tables  -> %s", resolve_path(config.require("paths.tables")))
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
