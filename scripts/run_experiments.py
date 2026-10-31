#!/usr/bin/env python
"""Baselines and HGT ablations over several seeds.

Every model is trained on the same chronological training split, tuned on the
validation split only, stopped by validation NDCG@10, and scored once on the
test split by the same evaluator. Each (model, seed) run writes its own file
under ``reports/results/runs/``, so the grid can be sharded over processes and
resumed after an interruption.

    python scripts/run_experiments.py --only baselines
    python scripts/run_experiments.py --only hgt --arms A,B --seeds 42,43
    python scripts/run_experiments.py --aggregate          # merge runs, write tables
    python scripts/run_experiments.py --protocol refit     # refit on train + validation

Two protocols. ``tune`` (default) trains on the training split, stops by validation
NDCG@10 and is scored on test. ``refit`` then retrains each model on train +
validation for exactly the number of epochs its ``tune`` run validated best, and
scores it on the same test split, so the gap between the last training interaction
and the test period is as short as it was between train and validation.

Outputs:
    reports/results/experiment_results.json   mean/std over seeds per model
    reports/results/significance.json         paired bootstrap between models
    reports/tables/model_comparison.md
    reports/tables/trust_scores.csv           CoLA scores (kept: the negative control)
    reports/tables/trust_scores_behavioural.csv
    checkpoints/best_model.pt, training_history.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import active_dirs  # noqa: E402

DIRS = active_dirs()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.experiments.context import ExperimentContext, paired_bootstrap  # noqa: E402
from src.graph.weighting import compute_edge_weights  # noqa: E402
from src.models.collaborative_filtering import ItemKNNRecommender  # noqa: E402
from src.models.ensemble import EnsembleRecommender  # noqa: E402
from src.models.hgt import HGTRecommender  # noqa: E402
from src.models.lightgcn import LightGCNRecommender  # noqa: E402
from src.models.matrix_factorization import MatrixFactorizationRecommender  # noqa: E402
from src.models.popularity import PopularityRecommender  # noqa: E402
from src.models.sasrec import SASRecRecommender  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = get_logger("run_experiments")

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = DIRS["results"] / "runs"
TUNE_RUNS_DIR = RUNS_DIR
SUFFIX = ""          # "_refit" for the refit protocol
TUNING_DIR = DIRS["results"] / "tuning"
BASELINES = ("popularity", "item_knn_cf", "matrix_factorization", "lightgcn", "sasrec")
COMPARISON_COLUMNS = ["precision@10", "recall@10", "ndcg@10", "map@10", "mrr",
                      "hit_rate@10", "coverage@10", "diversity@10", "novelty@10"]


# -- tuned settings ---------------------------------------------------------
def best_tuned() -> dict[str, dict]:
    """Highest-validation record per model family from ``tune_models.py``."""
    best: dict[str, dict] = {}
    for path in sorted(TUNING_DIR.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        family = record["model"]
        if family not in best or record["val_ndcg@10"] > best[family]["val_ndcg@10"]:
            best[family] = record
    return best


def split_hgt_overrides(overrides: dict) -> tuple[dict, dict]:
    model = {k: v for k, v in overrides.items() if k.startswith("hgt.")}
    training = {k: v for k, v in overrides.items() if not k.startswith("hgt.")}
    return model, training


# -- run bookkeeping --------------------------------------------------------
def run_path(name: str, seed: int) -> Path:
    return RUNS_DIR / f"{name}__seed{seed}.json"


def save_run(name: str, seed: int, payload: dict, per_user: np.ndarray) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_path(name, seed).write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    np.save(RUNS_DIR / f"{name}__seed{seed}.npy", per_user)


def finish(ctx, model, name: str, seed: int, started: float, extra: dict | None = None) -> None:
    """Score on test once, then persist the run."""
    result = ctx.test_eval.evaluate(model, name=name)
    per_user = ctx.test_eval.per_user_ndcg(model, k=10)
    payload = result.to_dict()
    payload.update({
        "seed": seed, "train_seconds": round(time.perf_counter() - started, 1),
        "val_ndcg@10": getattr(model, "best_val", None),
        "best_epoch": getattr(model, "best_epoch", None), **(extra or {}),
    })
    save_run(name, seed, payload, per_user)


# -- trust ------------------------------------------------------------------
def cola_trust(ctx: ExperimentContext, device) -> np.ndarray:
    """CoLA trust per user. Reused from the saved scores when they exist."""
    path = ctx.tables_dir / "trust_scores.csv"
    if path.is_file():
        frame = pd.read_csv(path)
        mapping = dict(zip(frame["entity_id"].astype(str), frame["trust_score"]))
        values = np.array([mapping.get(u, np.nan) for u in ctx.vocab.user_ids], dtype=np.float32)
        if not np.isnan(values).any():
            return values
        logger.warning("Saved CoLA scores do not cover the vocabulary; recomputing.")

    from src.anomaly.cola import CoLADetector
    from src.anomaly.trust import build_trust_scores
    from src.features.transforms import LeakageSafeScaler
    from src.features.user_features import build_user_profiles

    profiles = build_user_profiles(ctx.train).reindex(ctx.vocab.user_ids)
    numeric = profiles.select_dtypes(include=[np.number])
    scaler = LeakageSafeScaler().fit(numeric, scope="TRAIN SPLIT ONLY (CoLA features)")
    features = np.nan_to_num(scaler.transform(numeric).to_numpy(dtype=np.float32))
    detector = CoLADetector(ctx.model_config, device=str(device), seed=42)
    adjacency = detector.build_projection(ctx.train_idx, ctx.vocab.n_users, ctx.vocab.n_items)
    result = detector.fit_score(features, adjacency, node_ids=ctx.vocab.user_ids,
                                entity_type="user")
    scores = build_trust_scores(result, ctx.model_config, entity_type="user")
    scores.save(path)
    return np.asarray(scores.trust, dtype=np.float32)


def arm_weights(ctx: ExperimentContext, flags: dict, user_trust: dict[str, np.ndarray],
                recency_lambda: float, trust_alpha: float) -> np.ndarray | None:
    """Per-interaction loss weight w = trust^alpha * recency^beta for one arm."""
    source = str(flags.get("trust", "none"))
    use_trust = source != "none"
    use_recency = bool(flags.get("recency", False))
    if not (use_trust or use_recency):
        return None
    config = ctx.model_config.with_overrides({"edge_weighting.use_trust": use_trust,
                                              "edge_weighting.use_temporal": use_recency,
                                              "edge_weighting.trust_exponent": trust_alpha})
    return compute_edge_weights(
        ctx.train, config,
        trust=ctx.row_trust(user_trust[source]) if use_trust else None,
        temporal=ctx.recency(recency_lambda) if use_recency else None,
    )


# -- model families ---------------------------------------------------------
def tuned_epochs(name: str, seed: int) -> int | None:
    """Best epoch the same model and seed validated at under the ``tune`` protocol."""
    path = TUNE_RUNS_DIR / f"{name}__seed{seed}.json"
    if not path.is_file():
        raise SystemExit(f"Refit needs {path.name}; run the tune protocol first.")
    return json.loads(path.read_text(encoding="utf-8")).get("best_epoch")


def run_baseline(ctx: ExperimentContext, name: str, seed: int, tuned: dict,
                 refit: bool = False) -> None:
    started = time.perf_counter()
    validator = None if refit else ctx.val_eval.ndcg
    params = dict(tuned.get(_family(name), {}).get("params", {}))
    if name == "popularity":
        model = PopularityRecommender(ctx.vocab.n_users, ctx.vocab.n_items).fit(ctx.train_idx)
    elif name == "item_knn_cf":
        model = ItemKNNRecommender(ctx.vocab.n_users, ctx.vocab.n_items,
                                   **{"k": 50, **params}).fit(ctx.train_idx)
    elif name == "matrix_factorization":
        model = MatrixFactorizationRecommender(
            ctx.vocab.n_users, ctx.vocab.n_items, device=str(ctx.device), seed=seed,
            **{"epochs": 200, **params, **({"epochs": tuned_epochs(name, seed)} if refit else {})})
        model.fit(ctx.train_idx, sampler=ctx.sampler(seed), validator=validator)
    elif name == "lightgcn":
        model = LightGCNRecommender(
            ctx.vocab.n_users, ctx.vocab.n_items, device=str(ctx.device), seed=seed,
            **{"epochs": 200, **params, **({"epochs": tuned_epochs(name, seed)} if refit else {})})
        model.fit(ctx.train_idx, sampler=ctx.sampler(seed), validator=validator)
    elif name == "sasrec":
        model = SASRecRecommender(
            ctx.vocab.n_users, ctx.vocab.n_items, device=str(ctx.device), seed=seed,
            **{"epochs": 200, **params, **({"epochs": tuned_epochs(name, seed)} if refit else {})})
        model.fit(ctx.train_idx, sampler=ctx.sampler(seed), validator=validator)
    else:
        raise ValueError(name)
    finish(ctx, model, name, seed, started, {"tuned_params": params})


def _family(name: str) -> str:
    return {"item_knn_cf": "knn", "matrix_factorization": "mf"}.get(name, name)


def run_hgt_arm(ctx: ExperimentContext, arm: dict, seed: int, tuned: dict,
                user_trust: dict[str, np.ndarray], recency_lambda: float,
                trust_alpha: float, refit: bool = False) -> HGTRecommender:
    flags = arm["flags"]
    name = f"{arm['id']}_{arm['name']}"
    started = time.perf_counter()
    overrides = tuned.get("hgt", {}).get("overrides", {})
    model_over, training_over = split_hgt_overrides(overrides)
    if not flags.get("mask", True):
        training_over = {**training_over, "training.edge_mask.enabled": False}
    scheduled = flags.get("schedule") == "cosine"
    if scheduled:
        training_over = {**training_over, **tuned.get("hgt_sched", {}).get(
            "sched_overrides", {"training.max_epochs": 150, "scheduler.name": "cosine"})}
    weights = arm_weights(ctx, flags, user_trust, recency_lambda, trust_alpha)

    model = HGTRecommender(
        ctx.vocab.n_users, ctx.vocab.n_items, bundle=ctx.bundle(sbert=bool(flags["sbert"])),
        model_config=ctx.model_config.with_overrides(model_over),
        training_config=ctx.training_config.with_overrides(training_over),
        device=str(ctx.device), use_fusion=bool(flags["fusion"]),
        use_weighted_loss=weights is not None, seed=seed,
        epochs=tuned_epochs(name, seed) if (refit and not scheduled) else None,
    )
    # A fresh sampler per run: arms must not share (and consume) one random stream.
    model.fit(ctx.train_idx,
              sampler=ctx.sampler(seed, training_over.get("negative_sampling.strategy")),
              weights=weights, validator=None if refit else ctx.val_eval.ndcg)
    finish(ctx, model, name, seed, started, {"flags": flags, "hgt_overrides": overrides,
                                             "recency_lambda": recency_lambda,
                                             "trust_alpha": trust_alpha,
                                             "history": model.history})
    user_emb, item_emb = model.embeddings()
    np.savez_compressed(RUNS_DIR / f"{name}__seed{seed}.emb.npz",
                        user=user_emb.astype(np.float16), item=item_emb.astype(np.float16))
    return model


# -- aggregation ------------------------------------------------------------
def aggregate(ctx: ExperimentContext) -> list[dict]:
    """Merge per-seed runs into one record per model."""
    by_model: dict[str, list[dict]] = {}
    for path in sorted(RUNS_DIR.glob("*__seed*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        by_model.setdefault(record["model"], []).append(record)

    merged = []
    for name, runs in by_model.items():
        runs.sort(key=lambda r: r["seed"])
        keys = runs[0]["metrics"].keys()
        mean = {k: float(np.mean([r["metrics"][k] for r in runs])) for k in keys}
        std = {k: float(np.std([r["metrics"][k] for r in runs], ddof=1)) if len(runs) > 1
               else 0.0 for k in keys}
        vals = [r["val_ndcg@10"] for r in runs if r.get("val_ndcg@10") is not None]
        merged.append({
            "model": name, "split": "test", "n_seeds": len(runs),
            "seeds": [r["seed"] for r in runs], "metrics": mean, "metrics_std": std,
            "per_seed_ndcg@10": [r["metrics"]["ndcg@10"] for r in runs],
            "val_ndcg@10": float(np.mean(vals)) if vals else None,
            "mean_best_epoch": float(np.mean([r["best_epoch"] for r in runs
                                              if r.get("best_epoch") is not None]))
            if any(r.get("best_epoch") is not None for r in runs) else None,
            "mean_train_seconds": float(np.mean([r["train_seconds"] for r in runs])),
            "diagnostics": runs[0]["diagnostics"], "notes": runs[0]["notes"],
        })
    merged.sort(key=lambda r: -r["metrics"]["ndcg@10"])
    return merged


def significance(merged: list[dict]) -> dict:
    """Paired bootstrap over users, on per-user NDCG averaged across seeds."""
    per_user = {}
    for record in merged:
        if "ensemble_of" in record:
            per_user[record["model"]] = np.load(RUNS_DIR / f"{record['model']}__ensemble.npy")
            continue
        arrays = [np.load(RUNS_DIR / f"{record['model']}__seed{s}.npy") for s in record["seeds"]]
        per_user[record["model"]] = np.mean(arrays, axis=0)

    baselines = [r["model"] for r in merged if r["model"] in BASELINES]
    best_baseline = baselines[0] if baselines else None   # merged is sorted by ndcg@10
    out = {"best_baseline": best_baseline, "comparisons": {}}
    if best_baseline is None:
        return out
    for record in merged:
        if record["model"] in BASELINES:
            continue
        out["comparisons"][record["model"]] = {
            "vs_" + best_baseline: paired_bootstrap(per_user[record["model"]],
                                                    per_user[best_baseline]),
        }
        if "matrix_factorization" in per_user and best_baseline != "matrix_factorization":
            out["comparisons"][record["model"]]["vs_matrix_factorization"] = paired_bootstrap(
                per_user[record["model"]], per_user["matrix_factorization"])
    return out


def comparison_table(merged: list[dict]) -> str:
    lines = [
        "# Model Comparison", "",
        "Mean ± standard deviation over seeds. All models share one chronological training",
        "split, were tuned on the validation split only, stopped by validation NDCG@10, and",
        "were scored once on the test split by the same evaluator (full-catalogue ranking,",
        "training positives masked). Sorted by ndcg@10.", "",
        "| Model | seeds | " + " | ".join(COMPARISON_COLUMNS) + " |",
        "| --- | ---: |" + " ---: |" * len(COMPARISON_COLUMNS),
    ]
    for record in merged:
        cells = [f"{record['metrics'][c]:.4f} ± {record['metrics_std'][c]:.4f}"
                 for c in COMPARISON_COLUMNS]
        lines.append(f"| {record['model']} | {record['n_seeds']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def ensemble_records(ctx: ExperimentContext, merged: list[dict]) -> list[dict]:
    """Score-average every HGT arm that has at least three seeds with saved embeddings."""
    records = []
    for record in merged:
        files = [RUNS_DIR / f"{record['model']}__seed{s}.emb.npz" for s in record["seeds"]]
        files = [f for f in files if f.is_file()]
        if len(files) < 3:
            continue
        members = []
        for f in files:
            with np.load(f) as data:
                members.append((data["user"].astype(np.float32), data["item"].astype(np.float32)))
        ensemble = EnsembleRecommender(ctx.vocab.n_users, ctx.vocab.n_items, members)
        name = f"{record['model']}+ens{len(members)}"
        result = ctx.test_eval.evaluate(ensemble, name=name)
        per_user = ctx.test_eval.per_user_ndcg(ensemble, k=10)
        np.save(RUNS_DIR / f"{name}__ensemble.npy", per_user)
        payload = result.to_dict()
        payload.update({"seeds": record["seeds"], "n_seeds": len(members),
                        "metrics_std": {k: 0.0 for k in payload["metrics"]},
                        "per_seed_ndcg@10": [payload["metrics"]["ndcg@10"]],
                        "ensemble_of": record["model"], "val_ndcg@10": None,
                        "mean_best_epoch": None, "mean_train_seconds": record["mean_train_seconds"]
                        * len(members)})
        records.append(payload)
    return records


def write_outputs(ctx: ExperimentContext) -> None:
    merged = aggregate(ctx)
    merged += ensemble_records(ctx, merged)
    merged.sort(key=lambda r: -r["metrics"]["ndcg@10"])
    if not merged:
        logger.warning("No runs found in %s.", RUNS_DIR)
        return
    (ctx.results_dir / f"experiment_results{SUFFIX}.json").write_text(
        json.dumps(merged, indent=2), encoding="utf-8")
    (ctx.results_dir / f"significance{SUFFIX}.json").write_text(
        json.dumps(significance(merged), indent=2), encoding="utf-8")
    (ctx.tables_dir / f"model_comparison{SUFFIX}.md").write_text(comparison_table(merged),
                                                                encoding="utf-8")
    logger.info("=" * 72)
    for record in merged:
        logger.info("  %-28s ndcg@10 %.4f ± %.4f  recall@10 %.4f  (%d seeds)",
                    record["model"], record["metrics"]["ndcg@10"],
                    record["metrics_std"]["ndcg@10"], record["metrics"]["recall@10"],
                    record["n_seeds"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="DyTrustRec experiments.")
    parser.add_argument("--only", choices=["baselines", "hgt", "all"], default="all")
    parser.add_argument("--arms", default=None, help="Comma-separated ablation ids, e.g. A,B,F")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds (default: config).")
    parser.add_argument("--protocol", choices=["tune", "refit"], default="tune")
    parser.add_argument("--tag", default=None,
                        help="Write runs to reports/results/runs_<tag>/ (for side studies such as saved curves).")
    parser.add_argument("--force", action="store_true", help="Re-run even if a run file exists.")
    parser.add_argument("--aggregate", action="store_true", help="Only merge existing runs.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    global RUNS_DIR, SUFFIX
    refit = args.protocol == "refit"
    if refit:
        RUNS_DIR, SUFFIX = DIRS["results"] / "runs_refit", "_refit"
    if args.tag:
        RUNS_DIR, SUFFIX = DIRS["results"] / f"runs_{args.tag}", f"_{args.tag}"
    ctx = ExperimentContext.for_refit(args.device) if refit else ExperimentContext(device=args.device)
    setup_logging(args.log_level, log_file=ctx.results_dir / f"experiments{SUFFIX}.log")
    set_seed(int(ctx.experiment_config.get("seed", 42)))

    if args.aggregate:
        write_outputs(ctx)
        return 0

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else list(ctx.experiment_config.get("runtime.repeat_seeds", [42])))
    tuned = best_tuned()
    logger.info("Tuned configurations in use: %s",
                {k: v["name"] for k, v in tuned.items()} or "none (defaults)")

    if args.only in ("baselines", "all"):
        for name in BASELINES:
            for seed in ([seeds[0]] if name in ("popularity", "item_knn_cf") else seeds):
                if run_path(name, seed).is_file() and not args.force:
                    continue
                logger.info("-" * 72)
                logger.info("Baseline %s (seed %d)", name, seed)
                run_baseline(ctx, name, seed, tuned, refit)

    if args.only in ("hgt", "all"):
        arms = list(ctx.experiment_config.require("ablations"))
        if args.arms:
            wanted = {a.strip().upper() for a in args.arms.split(",")}
            arms = [a for a in arms if str(a["id"]).upper() in wanted]
        if refit:
            # CoLA scores exist for the training users only; the negative control is not refit.
            arms = [a for a in arms if a["flags"].get("trust") != "cola"]

        # Trust arrays are computed once, from the training split only.
        user_trust: dict[str, np.ndarray] = {}
        needed = {a["flags"].get("trust", "none") for a in arms}
        if "behavioural" in needed:
            scores = ctx.behavioural_scores()
            scores.save(ctx.tables_dir / "trust_scores_behavioural.csv")
            user_trust["behavioural"] = np.asarray(scores.trust, dtype=np.float32)
        if "cola" in needed:
            user_trust["cola"] = cola_trust(ctx, ctx.device)
        recency_lambda = float(tuned.get("hgt_recency", {}).get(
            "recency_lambda", ctx.model_config.get("temporal.loss_lambda_per_year", 2.0)))
        trust_alpha = float(tuned.get("hgt_trust", {}).get(
            "trust_alpha", ctx.model_config.get("edge_weighting.trust_exponent", 1.0)))
        logger.info("Objective weighting: recency lambda %.2f, trust exponent %.2f",
                    recency_lambda, trust_alpha)

        for arm in arms:
            name = f"{arm['id']}_{arm['name']}"
            for seed in seeds:
                if run_path(name, seed).is_file() and not args.force:
                    continue
                logger.info("-" * 72)
                logger.info("Ablation %s, seed %d: %s", arm["id"], seed, arm["description"])
                model = run_hgt_arm(ctx, arm, seed, tuned, user_trust, recency_lambda, trust_alpha, refit)
                if (not refit and arm["id"] == str(ctx.experiment_config.get("export_arm", "F"))
                        and seed == int(ctx.experiment_config.get("seed", 42))):
                    ckpt = DIRS["checkpoints"]
                    ckpt.mkdir(exist_ok=True)
                    torch.save({"state_dict": model.model.state_dict(),
                                "model_config": ctx.model_config.to_dict(),
                                "training_config": ctx.training_config.to_dict(),
                                "flags": arm["flags"], "seed": seed, "ablation": arm["id"]},
                               ckpt / "best_model.pt")
                    (ckpt / "training_history.json").write_text(
                        json.dumps(model.history, indent=2), encoding="utf-8")

    write_outputs(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
