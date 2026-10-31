#!/usr/bin/env python
"""Phase 1 entrypoint: ingest and clean a tier of the Amazon Reviews 2023 corpus.

Examples:
    # Fast end-to-end check on the smallest tier
    python scripts/prepare_data.py --tier smoke

    # Default working tier
    python scripts/prepare_data.py --tier dev

    # Override any config key without editing YAML
    python scripts/prepare_data.py --tier dev --set sampling.strategy=reservoir

Outputs (raw inputs under data/raw/ are never modified):
    data/interim/interactions_raw.parquet
    data/interim/products_raw.parquet
    data/interim/load_manifest.json
    data/processed/interactions_clean.parquet
    data/processed/products_clean.parquet
    reports/tables/cleaning_report.{json,md}
    reports/results/prepare_data_run.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/prepare_data.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cleaner import DataCleaner                     # noqa: E402
from src.data.loader import DatasetLoader                    # noqa: E402
from src.utils.config import load_config, resolve_path       # noqa: E402
from src.utils.logging import get_logger, setup_logging      # noqa: E402
from src.utils.seed import set_seed, write_run_manifest      # noqa: E402

logger = get_logger("prepare_data")


def _parse_overrides(pairs: list[str]) -> dict[str, object]:
    """Turn ``--set a.b=value`` pairs into a dotted-key override mapping.

    Values are parsed as JSON when possible so ``--set tiers.dev.limits.max_reviews=1000``
    yields an int, and ``--set x=null`` yields ``None``.
    """
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
        description="DyTrustRec Phase 1 - data ingestion and cleaning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="data_config", help="Config stem under configs/.")
    parser.add_argument("--tier", default=None, choices=["smoke", "dev", "full"],
                        help="Override active_tier from the config.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="Dotted-key config override. Repeatable.")
    parser.add_argument("--seed", type=int, default=None, help="Override sampling.seed.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", default=None, help="Also write a full DEBUG log here.")
    parser.add_argument("--no-save", action="store_true",
                        help="Run the pipeline but write no artefacts (for testing).")
    args = parser.parse_args(argv)

    overrides = _parse_overrides(args.overrides)
    if args.tier:
        overrides["active_tier"] = args.tier
    if args.seed is not None:
        overrides["sampling.seed"] = args.seed

    config = load_config(args.config, overrides)
    tier_name = config.require("active_tier")

    log_file = args.log_file or (resolve_path(config.require("paths.results")) / f"prepare_data_{tier_name}.log")
    setup_logging(args.log_level, log_file=log_file)

    seed = int(config.get("sampling.seed", 42))
    set_seed(seed)

    logger.info("DyTrustRec | Phase 1 | tier=%s | seed=%d", tier_name, seed)
    if overrides:
        logger.info("Config overrides: %s", overrides)

    # ---- ingest ----------------------------------------------------------
    loader = DatasetLoader(config)
    loaded = loader.load()

    # ---- clean -----------------------------------------------------------
    cleaner = DataCleaner(config)
    cleaned = cleaner.clean(loaded)

    if args.no_save:
        logger.info("--no-save set; skipping artefact writes.")
        return 0

    # ---- persist ---------------------------------------------------------
    interim = resolve_path(config.require("paths.interim"))
    processed = resolve_path(config.require("paths.processed"))
    results = resolve_path(config.require("paths.results"))
    for directory in (interim, processed, results):
        directory.mkdir(parents=True, exist_ok=True)

    loaded.interactions.to_parquet(interim / "interactions_raw.parquet", index=False)
    if loaded.products is not None:
        loaded.products.to_parquet(interim / "products_raw.parquet", index=False)
    (interim / "load_manifest.json").write_text(
        json.dumps(loaded.manifest, indent=2, default=str), encoding="utf-8"
    )

    cleaned.interactions.to_parquet(processed / "interactions_clean.parquet", index=False)
    if cleaned.products is not None:
        cleaned.products.to_parquet(processed / "products_clean.parquet", index=False)

    cleaner.save_report()

    write_run_manifest(
        results / "prepare_data_run.json",
        seed=seed,
        config=config.to_dict(),
        extra={
            "phase": "1-data-ingestion-and-cleaning",
            "tier": tier_name,
            "load_manifest": loaded.manifest,
            "cleaning_summary": cleaned.report.to_dict()["summary"],
        },
    )

    logger.info("=" * 72)
    logger.info("Phase 1 complete.")
    logger.info("  raw subset  -> %s", interim)
    logger.info("  cleaned     -> %s", processed)
    logger.info("  reports     -> %s", resolve_path(config.require("paths.tables")))
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
