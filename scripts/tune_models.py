#!/usr/bin/env python
"""Hyper-parameter search on the VALIDATION split, for every trainable model.

Test data is never scored here. Each job trains one configuration (with the same
validation early stopping the final runs use) and records its best validation
NDCG@10; ``run_experiments.py`` then reads the winner for each model.

Giving the baselines the same search is deliberate. An untuned baseline next to
a tuned model is not a comparison: the first experiment grid had a LightGCN that
scored below popularity for exactly that reason.

    python scripts/tune_models.py --model hgt --jobs base,keep_0.5
    python scripts/tune_models.py --model mf
    python scripts/tune_models.py --model lightgcn
    python scripts/tune_models.py --model knn
    python scripts/tune_models.py --summary
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import active_dirs  # noqa: E402

DIRS = active_dirs()

from src.experiments.context import ExperimentContext  # noqa: E402
from src.graph.weighting import compute_edge_weights  # noqa: E402
from src.models.collaborative_filtering import ItemKNNRecommender  # noqa: E402
from src.models.hgt import HGTRecommender  # noqa: E402
from src.models.lightgcn import LightGCNRecommender  # noqa: E402
from src.models.matrix_factorization import MatrixFactorizationRecommender  # noqa: E402
from src.models.sasrec import SASRecRecommender  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger("tune_models")
TUNING_DIR = DIRS["results"] / "tuning"
SEED = 42

# One-at-a-time moves around the base configuration. The base keeps the values
# the first diagnostics settled on; each other job changes exactly one thing.
HGT_JOBS: dict[str, dict] = {
    "base": {},
    "keep_0.5": {"training.edge_mask.keep_prob": 0.5},
    "keep_0.65": {"training.edge_mask.keep_prob": 0.65},
    "keep_0.9": {"training.edge_mask.keep_prob": 0.9},
    "keep_0.35": {"training.edge_mask.keep_prob": 0.35},
    "keep_0.25": {"training.edge_mask.keep_prob": 0.25},
    "lr_0.01": {"optimizer.learning_rate": 0.01},
    "lr_0.002": {"optimizer.learning_rate": 0.002},
    "wd_1e-3": {"optimizer.weight_decay": 1e-3},
    "wd_1e-5": {"optimizer.weight_decay": 1e-5},
    "dropout_0.1": {"hgt.dropout": 0.1},
    "dropout_0.05": {"hgt.dropout": 0.05},
    "dropout_0.0": {"hgt.dropout": 0.0},
    "dropout_0.4": {"hgt.dropout": 0.4},
    "layers_1": {"hgt.num_layers": 1},
    "layers_3": {"hgt.num_layers": 3},
    "hidden_64": {"hgt.hidden_channels": 64},
    "hidden_256": {"hgt.hidden_channels": 256},
    "neg_4": {"negative_sampling.num_negatives": 4},
    "neg_pop": {"negative_sampling.strategy": "popularity"},
}

# Objective weighting, tried on top of the best structural configuration found
# above. These decide, on validation data, whether recency and trust earn a place
# in the final model and how strongly.
WEIGHT_JOBS: dict[str, dict] = {
    "rec_0.1": {"recency_lambda": 0.1},
    "rec_0.25": {"recency_lambda": 0.25},
    "rec_0.5": {"recency_lambda": 0.5},
    "rec_1.0": {"recency_lambda": 1.0},
    "rec_2.0": {"recency_lambda": 2.0},
    "rec_4.0": {"recency_lambda": 4.0},
    "trust_0.5": {"trust_alpha": 0.5},
    "trust_1.0": {"trust_alpha": 1.0},
}

# Learning-rate schedule jobs: the arm-E recipe (trust x recency weights on the best
# structure) with a cosine decay over a fixed epoch budget. Validation decides the budget.
SCHEDULE_JOBS: dict[str, dict] = {
    # Early stopping is off: it would end the run on the first noisy plateau, long
    # before the decay reaches its low-rate tail. The best validation checkpoint is
    # still restored at the end.
    "cos_150": {"training.max_epochs": 150, "scheduler.name": "cosine", "early_stopping.enabled": False},
    "cos_250": {"training.max_epochs": 250, "scheduler.name": "cosine", "early_stopping.enabled": False},
}

MF_GRID = {"embedding_dim": [64, 128], "learning_rate": [0.01, 0.003, 0.001],
           "weight_decay": [1e-5, 1e-4, 1e-3]}
LIGHTGCN_GRID = {"embedding_dim": [64, 128], "num_layers": [2, 3],
                 "learning_rate": [0.05, 0.01, 0.003], "weight_decay": [0.0, 1e-5]}
SASREC_GRID = {"embedding_dim": [64, 128], "dropout": [0.2, 0.5],
               "learning_rate": [0.001, 0.003], "max_len": [20, 50]}
KNN_GRID = {"k": [10, 20, 50, 100, 200]}


def _save(model: str, name: str, record: dict) -> None:
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    (TUNING_DIR / f"{model}__{name}.json").write_text(json.dumps(record, indent=2),
                                                      encoding="utf-8")


def _split_overrides(overrides: dict) -> tuple[dict, dict]:
    model_keys = {k: v for k, v in overrides.items() if k.startswith("hgt.")}
    training_keys = {k: v for k, v in overrides.items() if not k.startswith("hgt.")}
    return model_keys, training_keys


def _best_structural_overrides() -> dict:
    """Best finished structural HGT job so far (the weight jobs build on it)."""
    best = None
    for path in sorted(TUNING_DIR.glob("hgt__*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if best is None or record["val_ndcg@10"] > best["val_ndcg@10"]:
            best = record
    return dict(best["overrides"]) if best else {}


def _combined_overrides() -> dict:
    """Union of each parameter's winning setting, where it beat the base config.

    One-at-a-time search finds the best value of each parameter with the others
    fixed; whether those winners work together is a separate question, so the
    union is trained and validated as its own job rather than assumed.
    """
    records = {}
    for path in sorted(TUNING_DIR.glob("hgt__*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        records[record["name"]] = record
    base = records.get("base", {}).get("val_ndcg@10", float("-inf"))
    winners: dict[str, dict] = {}
    for name, record in records.items():
        if name in ("base", "combined") or record["val_ndcg@10"] <= base:
            continue
        group = name.split("_")[0]
        if group not in winners or record["val_ndcg@10"] > winners[group]["val_ndcg@10"]:
            winners[group] = record
    merged: dict = {}
    for record in winners.values():
        merged.update(record["overrides"])
    return merged


def _best_family_value(family: str, key: str, default: float) -> float:
    """``key`` from the best-validation record of a tuning family, else ``default``."""
    best = None
    for path in sorted(TUNING_DIR.glob(f"{family}__*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if best is None or record["val_ndcg@10"] > best["val_ndcg@10"]:
            best = record
    return float(best[key]) if best else default


def _weights(ctx: ExperimentContext, recency_lambda: float | None, trust_alpha: float | None):
    """Per-interaction loss weights; ``None`` for a factor means that factor is off."""
    use_recency, use_trust = recency_lambda is not None, trust_alpha is not None
    config = ctx.model_config.with_overrides({
        "edge_weighting.use_trust": use_trust, "edge_weighting.use_temporal": use_recency,
        "edge_weighting.trust_exponent": trust_alpha if use_trust else 1.0})
    return compute_edge_weights(
        ctx.train, config,
        trust=ctx.row_trust(ctx.behavioural_trust()) if use_trust else None,
        temporal=ctx.recency(recency_lambda) if use_recency else None)


def tune_hgt(ctx: ExperimentContext, names: list[str]) -> None:
    bundle = ctx.bundle(sbert=True)
    for name in names:
        weights, family, extra = None, "hgt", {}
        if name in SCHEDULE_JOBS:
            lam = _best_family_value("hgt_recency", "recency_lambda", 0.5)
            alpha = _best_family_value("hgt_trust", "trust_alpha", 1.0)
            overrides = {**_best_structural_overrides(), **SCHEDULE_JOBS[name]}
            weights = _weights(ctx, recency_lambda=lam, trust_alpha=alpha)
            family = "hgt_sched"
            extra = {"sched_overrides": SCHEDULE_JOBS[name], "recency_lambda": lam, "trust_alpha": alpha}
        elif name == "combined":
            overrides = _combined_overrides()
            logger.info("Combined configuration: %s", overrides)
        elif name in WEIGHT_JOBS:
            spec = WEIGHT_JOBS[name]
            overrides = _best_structural_overrides()
            weights = _weights(ctx, recency_lambda=spec.get("recency_lambda"),
                               trust_alpha=spec.get("trust_alpha"))
            family = "hgt_recency" if "recency_lambda" in spec else "hgt_trust"
            extra = dict(spec)
        else:
            overrides = HGT_JOBS[name]

        if name == "combined" and not overrides:
            logger.info("No parameter beat the base configuration; nothing to combine.")
            continue
        model_over, training_over = _split_overrides(overrides)
        model_config = ctx.model_config.with_overrides(model_over)
        training_config = ctx.training_config.with_overrides(training_over)
        started = time.perf_counter()
        model = HGTRecommender(
            ctx.vocab.n_users, ctx.vocab.n_items, bundle=bundle, model_config=model_config,
            training_config=training_config, device=str(ctx.device), use_fusion=False,
            use_weighted_loss=weights is not None, seed=SEED)
        model.fit(ctx.train_idx, sampler=ctx.sampler(SEED, training_over.get("negative_sampling.strategy")),
                  weights=weights,
                  validator=lambda m: ctx.val_eval.ndcg(m))
        _save(family, name, {
            "model": family, "name": name, "overrides": overrides, "seed": SEED, **extra,
            "val_ndcg@10": model.best_val, "best_epoch": model.best_epoch,
            "epochs_run": len(model.history),
            "seconds": round(time.perf_counter() - started, 1)})
        logger.info("HGT %-12s val %.4f (epoch %s)", name, model.best_val, model.best_epoch)


def _grid(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


def tune_mf(ctx: ExperimentContext) -> None:
    for params in _grid(MF_GRID):
        name = "_".join(f"{k}={v}" for k, v in params.items())
        started = time.perf_counter()
        model = MatrixFactorizationRecommender(
            ctx.vocab.n_users, ctx.vocab.n_items, device=str(ctx.device), seed=SEED,
            epochs=200, **params)
        model.fit(ctx.train_idx, sampler=ctx.sampler(SEED),
                  validator=lambda m: ctx.val_eval.ndcg(m))
        _save("mf", name, {"model": "mf", "name": name, "params": params, "seed": SEED,
                           "val_ndcg@10": model.best_val, "best_epoch": model.best_epoch,
                           "seconds": round(time.perf_counter() - started, 1)})
        logger.info("MF %-60s val %.4f (epoch %s)", name, model.best_val, model.best_epoch)


def tune_lightgcn(ctx: ExperimentContext) -> None:
    for params in _grid(LIGHTGCN_GRID):
        name = "_".join(f"{k}={v}" for k, v in params.items())
        started = time.perf_counter()
        model = LightGCNRecommender(
            ctx.vocab.n_users, ctx.vocab.n_items, device=str(ctx.device), seed=SEED,
            epochs=200, **params)
        model.fit(ctx.train_idx, sampler=ctx.sampler(SEED),
                  validator=lambda m: ctx.val_eval.ndcg(m))
        _save("lightgcn", name, {"model": "lightgcn", "name": name, "params": params,
                                 "seed": SEED, "val_ndcg@10": model.best_val,
                                 "best_epoch": model.best_epoch,
                                 "seconds": round(time.perf_counter() - started, 1)})
        logger.info("LightGCN %-60s val %.4f (epoch %s)", name, model.best_val, model.best_epoch)


def tune_sasrec(ctx: ExperimentContext) -> None:
    for params in _grid(SASREC_GRID):
        name = "_".join(f"{k}={v}" for k, v in params.items())
        started = time.perf_counter()
        model = SASRecRecommender(ctx.vocab.n_users, ctx.vocab.n_items, device=str(ctx.device),
                                  seed=SEED, epochs=200, **params)
        model.fit(ctx.train_idx, sampler=ctx.sampler(SEED), validator=lambda m: ctx.val_eval.ndcg(m))
        _save("sasrec", name, {"model": "sasrec", "name": name, "params": params, "seed": SEED,
                               "val_ndcg@10": model.best_val, "best_epoch": model.best_epoch,
                               "seconds": round(time.perf_counter() - started, 1)})
        logger.info("SASRec %-60s val %.4f (epoch %s)", name, model.best_val, model.best_epoch)


def tune_knn(ctx: ExperimentContext) -> None:
    for params in _grid(KNN_GRID):
        name = "_".join(f"{k}={v}" for k, v in params.items())
        model = ItemKNNRecommender(ctx.vocab.n_users, ctx.vocab.n_items, **params)
        model.fit(ctx.train_idx)
        value = ctx.val_eval.ndcg(model)
        _save("knn", name, {"model": "knn", "name": name, "params": params, "seed": SEED,
                            "val_ndcg@10": value})
        logger.info("kNN %-12s val %.4f", name, value)


def summary() -> dict:
    """Best validation configuration per model, from whatever jobs have finished."""
    best: dict[str, dict] = {}
    for path in sorted(TUNING_DIR.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        current = best.get(record["model"])
        if current is None or record["val_ndcg@10"] > current["val_ndcg@10"]:
            best[record["model"]] = record
    return best


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validation-only hyper-parameter search.")
    parser.add_argument("--model", choices=["hgt", "mf", "lightgcn", "knn", "sasrec"])
    parser.add_argument("--jobs", default=None, help="Comma-separated HGT job names.")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    if args.summary:
        rows = []
        for path in sorted(TUNING_DIR.glob("*.json")):
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        for row in sorted(rows, key=lambda r: (r["model"], -r["val_ndcg@10"])):
            print(f"{row['model']:9s} {row['name']:62s} val {row['val_ndcg@10']:.4f}")
        return 0

    setup_logging("INFO")
    ctx = ExperimentContext(device=args.device)
    if args.model == "hgt":
        names = args.jobs.split(",") if args.jobs else list(HGT_JOBS) + list(WEIGHT_JOBS) + list(SCHEDULE_JOBS)
        tune_hgt(ctx, names)
    elif args.model == "mf":
        tune_mf(ctx)
    elif args.model == "lightgcn":
        tune_lightgcn(ctx)
    elif args.model == "knn":
        tune_knn(ctx)
    elif args.model == "sasrec":
        tune_sasrec(ctx)
    else:
        parser.error("choose --model or --summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
