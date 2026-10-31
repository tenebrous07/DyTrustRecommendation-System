#!/usr/bin/env python
"""Shill-attack robustness of the HGT recommender, with a working and a broken trust signal.

The trust component exists to protect the recommender from coordinated review
manipulation. Real Amazon data has no manipulation labels, so the only way to
test it is on a perturbed COPY of the training split with scripted shills added
(Module 45). The original data is never modified, and every number here is
labelled SYNTHETIC: it measures whether a defence handles *this* attack, not
whether it detects real fraud.

For each injection rate the same HGT (arm B settings) is trained with:

    none        no defence
    cola        CoLA trust (the original design) scales the BPR objective
    trust       behavioural trust scales the BPR objective
    trust+gate  trust weighting, and rows of users with shared campaign text are
                removed from the graph and the objective
    gate_v2     the same gate, but flagging users whose reviews READ like other accounts'
                reviews (sentence embeddings), so paraphrased campaigns are caught too
    oracle      true shill labels used as trust (upper bound for loss weighting)
    oracle_gate true shill labels used to remove the shills' rows from the graph and the
                objective (upper bound for gating)
    inverted    trust reversed - what a detector that ranks campaigns as
                trustworthy (CoLA, AUC 0.295) would do (negative control)

Reported on the CLEAN test split: NDCG@10, plus how often a promoted item reaches
a user's top 10 (target exposure), which is what an attacker wants.

    python scripts/run_robustness.py --rates 0.1,0.2 --seeds 42,43
    python scripts/run_robustness.py --aggregate
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

from src.anomaly.behavioural import BehaviouralTrustDetector  # noqa: E402
from src.anomaly.synthetic import _TEMPLATES, SyntheticAnomalyInjector  # noqa: E402
from src.evaluation.evaluator import Vocabulary  # noqa: E402
from src.experiments.context import ExperimentContext  # noqa: E402
from src.features.review_features import ReviewFeatureBuilder  # noqa: E402
from src.features.sbert import SBERTEncoder  # noqa: E402
from src.features.sentiment import add_sentiment_features  # noqa: E402
from src.graph.weighting import anomaly_to_trust, compute_edge_weights  # noqa: E402
from src.models.hgt import HGTRecommender  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import best_tuned, split_hgt_overrides  # noqa: E402

logger = get_logger("run_robustness")

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = DIRS["results"] / "robustness_runs"
OUT_NAME = "robustness_hgt"
CONDITIONS = ("none", "cola", "trust", "trust+gate", "gate_v2", "oracle", "oracle_gate", "inverted")
ADAPTIVE = False      # set by --adaptive: paraphrased text spread over weeks
GATE_THRESHOLD = 0.5   # share of a user's reviews that carry text other accounts also posted


def poisoned_inputs(ctx: ExperimentContext, injection) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Poisoned training split plus row-aligned SBERT and structured features.

    Genuine rows reuse the features already computed for them; only the scripted
    rows are encoded, and those reuse a handful of template texts.
    """
    poisoned = injection.interactions.reset_index(drop=True)
    is_shill = poisoned["user_id"].astype(str).str.startswith("SYNTH_SHILL_").to_numpy()

    key = lambda f: f["user_id"].astype(str) + "||" + f["item_id"].astype(str)  # noqa: E731
    original_pos = pd.Series(np.arange(len(ctx.train)), index=key(ctx.train))
    genuine_pos = original_pos.reindex(key(poisoned[~is_shill])).to_numpy()
    if np.isnan(genuine_pos).any():
        raise RuntimeError("A genuine row of the poisoned split is missing from the original split.")
    genuine_pos = genuine_pos.astype(int)

    encoder = SBERTEncoder(ctx.model_config)
    sbert_dim = ctx.review_embeddings().shape[1]
    embeddings = np.zeros((len(poisoned), sbert_dim), dtype=np.float32)
    embeddings[~is_shill] = ctx.review_embeddings()[genuine_pos]

    shill_frame = poisoned[is_shill]
    texts = encoder.build_texts(shill_frame)
    unique_texts = pd.Series(texts.unique())
    encoded = encoder.encode(unique_texts)
    lookup = dict(zip(unique_texts, range(len(unique_texts))))
    embeddings[is_shill] = encoded[[lookup[t] for t in texts]]

    # Sentiment: reuse for genuine rows, score the few distinct shill texts once.
    original = pd.read_parquet(ctx.processed / "review_features_train.parquet")
    sentiment = np.zeros(len(poisoned), dtype=np.float32)
    sentiment[~is_shill] = original["sentiment_compound"].to_numpy()[genuine_pos]
    if ADAPTIVE:
        # Paraphrases are nearly all unique, so scoring each with the BERT sentiment model would
        # cost tens of thousands of forward passes. They keep the template's meaning, so every
        # scripted review gets the mean sentiment of the templates: strongly positive praise.
        scored = add_sentiment_features(pd.DataFrame({"review_text": list(_TEMPLATES)}),
                                        model_config=ctx.model_config)
        sentiment[is_shill] = float(scored["sentiment_compound"].mean())
    else:
        unique_frame = pd.DataFrame({"review_text": shill_frame["review_text"].astype(str).unique()})
        scored = add_sentiment_features(unique_frame, model_config=ctx.model_config)
        by_text = dict(zip(scored["review_text"], scored["sentiment_compound"]))
        sentiment[is_shill] = shill_frame["review_text"].astype(str).map(by_text).to_numpy()

    with_sentiment = poisoned.assign(sentiment_compound=sentiment)
    builder = ReviewFeatureBuilder(
        lambda_per_year=float(ctx.model_config.get("temporal.lambda_per_year", 0.5)),
        min_weight=float(ctx.model_config.get("temporal.min_weight", 0.01))).fit(poisoned)
    features = builder.transform(with_sentiment)
    wanted = ctx.model_config.get("fusion.structured_features", [])
    return poisoned, embeddings, features[[c for c in wanted if c in features.columns]]


def cola_user_trust(pctx: ExperimentContext) -> np.ndarray:
    """CoLA trust on the poisoned split, computed once per context."""
    cached = getattr(pctx, "_cola_trust", None)
    if cached is not None:
        return cached
    from src.anomaly.cola import CoLADetector
    from src.anomaly.trust import build_trust_scores
    from src.features.transforms import LeakageSafeScaler
    from src.features.user_features import build_user_profiles

    profiles = build_user_profiles(pctx.train).reindex(pctx.vocab.user_ids)
    numeric = profiles.select_dtypes(include=[np.number])
    scaler = LeakageSafeScaler().fit(numeric, scope="poisoned training split (CoLA features)")
    features = np.nan_to_num(scaler.transform(numeric).to_numpy(dtype=np.float32))
    detector = CoLADetector(pctx.model_config, device="cpu", seed=42)
    adjacency = detector.build_projection(pctx.train_idx, pctx.vocab.n_users, pctx.vocab.n_items)
    result = detector.fit_score(features, adjacency, node_ids=pctx.vocab.user_ids, entity_type="user")
    pctx._cola_trust = np.asarray(build_trust_scores(result, pctx.model_config, entity_type="user").trust,
                                  dtype=np.float32)
    return pctx._cola_trust


def condition_inputs(pctx: ExperimentContext, injection, condition: str):
    """(loss weights, drop-row mask) for one defence condition."""
    n = len(pctx.train)
    config_on = pctx.model_config.with_overrides({"edge_weighting.use_trust": True,
                                                  "edge_weighting.use_temporal": False})
    if condition == "none":
        return None, None

    if condition == "oracle_gate":
        labels = injection.label_array(pctx.vocab.user_ids)
        return None, (labels == 1)[pctx.train_idx["user_idx"].to_numpy()]

    detector = BehaviouralTrustDetector().fit_score(pctx.train, pctx.vocab.user_ids)
    trust_cfg = pctx.model_config.require("trust")
    user_trust = anomaly_to_trust(detector.anomaly_scores, normalisation="rank",
                                  floor=float(trust_cfg.get("floor", 0.05)))
    if condition == "cola":
        user_trust = cola_user_trust(pctx)
    elif condition == "oracle":
        labels = injection.label_array(pctx.vocab.user_ids)
        user_trust = np.where(labels == 1, 0.05, 1.0).astype(np.float32)
    elif condition == "inverted":
        user_trust = np.clip(1.05 - user_trust, 0.05, 1.0).astype(np.float32)

    row_trust = pctx.row_trust(user_trust)
    weights = compute_edge_weights(pctx.train, config_on, trust=row_trust)

    drop = None
    if condition == "gate_v2":
        v2 = BehaviouralTrustDetector().fit_score(pctx.train, pctx.vocab.user_ids,
                                                  embeddings=pctx.review_embeddings())
        flagged = ((v2.indicators["semantic_shared"].fillna(0.0).to_numpy() >= GATE_THRESHOLD)
                   | (v2.indicators["shared_text"].fillna(0.0).to_numpy() >= GATE_THRESHOLD))
        drop = flagged[pctx.train_idx["user_idx"].to_numpy()]
        logger.info("Gate v2 flags %s of %s rows.", f"{int(drop.sum()):,}", f"{n:,}")
    if condition == "trust+gate":
        shared = detector.indicators["shared_text"].fillna(0.0).to_numpy()
        drop = (shared >= GATE_THRESHOLD)[pctx.train_idx["user_idx"].to_numpy()]
        logger.info("Gate flags %s of %s rows.", f"{int(drop.sum()):,}", f"{n:,}")
    return weights, drop


def run_one(rate: float, condition: str, seed: int, ctx: ExperimentContext, cache: dict) -> None:
    path = RUNS_DIR / f"rate{rate}__{condition}__seed{seed}.json"
    if path.is_file():
        return

    if rate not in cache:
        if rate == 0.0:
            cache[rate] = (ctx, None, set())
        else:
            experiment_config = ctx.experiment_config.with_overrides(
                {"synthetic_anomaly.injection_rate": rate,
                 "synthetic_anomaly.patterns.adaptive.enabled": ADAPTIVE})
            injection = SyntheticAnomalyInjector(experiment_config, seed=42).inject(ctx.train)
            poisoned, embeddings, structured = poisoned_inputs(ctx, injection)
            pctx = ExperimentContext(device=str(ctx.device), train=poisoned,
                                     embeddings=embeddings, structured=structured)
            targets = {pctx.vocab.items[t] for t in injection.target_items if t in pctx.vocab.items}
            cache[rate] = (pctx, injection, targets)
    pctx, injection, targets = cache[rate]

    tuned = best_tuned().get("hgt", {}).get("overrides", {})
    model_over, training_over = split_hgt_overrides(tuned)
    weights, drop = condition_inputs(pctx, injection, condition)

    started = time.perf_counter()
    model = HGTRecommender(
        pctx.vocab.n_users, pctx.vocab.n_items, bundle=pctx.bundle(sbert=True),
        model_config=pctx.model_config.with_overrides(model_over),
        training_config=pctx.training_config.with_overrides(training_over),
        device=str(pctx.device), use_fusion=False,
        use_weighted_loss=weights is not None, seed=seed)
    model.fit(pctx.train_idx, sampler=pctx.sampler(seed), weights=weights, drop_rows=drop,
              validator=pctx.val_eval.ndcg)

    ndcg = pctx.test_eval.ndcg(model, 10)
    tops = pctx.test_eval.top_lists(model, 10)
    hits = np.array([sum(int(i) in targets for i in lst) for lst in tops]) if targets else np.zeros(1)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "rate": rate, "condition": condition, "seed": seed, "label": "SYNTHETIC",
        "ndcg@10": ndcg, "val_ndcg@10": model.best_val, "best_epoch": model.best_epoch,
        "users_with_target_in_top10": float((hits > 0).mean()),
        "targets_per_list": float(hits.mean()),
        "n_train_rows": int(len(pctx.train)),
        "rows_gated": int(drop.sum()) if drop is not None else 0,
        "seconds": round(time.perf_counter() - started, 1)}, indent=2), encoding="utf-8")
    logger.info("rate %.2f %-10s seed %d: NDCG@10 %.4f, target exposure %.2f%%",
                rate, condition, seed, ndcg, 100 * float((hits > 0).mean()))


def clean_reference_rows() -> list[dict]:
    """The undefended model on clean data is arm B of the main grid, same settings and seeds."""
    rows = []
    for path in sorted((DIRS["results"] / "runs").glob("B_hgt_sbert__seed*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"rate": 0.0, "condition": "none", "seed": record["seed"],
                     "ndcg@10": record["metrics"]["ndcg@10"],
                     "users_with_target_in_top10": 0.0, "targets_per_list": 0.0})
    return rows


def aggregate() -> dict:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RUNS_DIR.glob("*.json"))]
    if not any(r["rate"] == 0.0 and r["condition"] == "none" for r in rows):
        wanted = {r["seed"] for r in rows}
        rows += [r for r in clean_reference_rows() if not wanted or r["seed"] in wanted]
    table: dict = {}
    for row in rows:
        table.setdefault(row["rate"], {}).setdefault(row["condition"], []).append(row)
    summary = {}
    for rate, conditions in sorted(table.items()):
        summary[str(rate)] = {}
        for condition, runs in conditions.items():
            def stat(key):
                values = [r[key] for r in runs]
                return {"mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
            summary[str(rate)][condition] = {
                "n_seeds": len(runs), "ndcg@10": stat("ndcg@10"),
                "users_with_target_in_top10": stat("users_with_target_in_top10"),
                "rows_gated": {"mean": float(np.mean([r.get("rows_gated", 0) for r in runs]))},
                "targets_per_list": stat("targets_per_list")}
    out = {"label": "SYNTHETIC - scripted shills added to a copy of the training split",
           "summary": summary}
    (DIRS["results"] / f"{OUT_NAME}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    lines = ["# HGT robustness to scripted shill campaigns", "",
             "> **SYNTHETIC RESULT.** Shills are scripted by `SyntheticAnomalyInjector` and added to a",
             "> copy of the training split. Evaluation is on the clean test split. These numbers measure",
             "> resistance to this attack, not detection of real Amazon fraud.", "",
             "| Injection rate | Defence | seeds | NDCG@10 | users with a promoted item in top-10 |",
             "| ---: | --- | ---: | ---: | ---: |"]
    for rate, conditions in summary.items():
        for condition in CONDITIONS:
            if condition in conditions:
                c = conditions[condition]
                lines.append(
                    f"| {float(rate):.0%} | {condition} | {c['n_seeds']} | "
                    f"{c['ndcg@10']['mean']:.4f} ± {c['ndcg@10']['std']:.4f} | "
                    f"{100 * c['users_with_target_in_top10']['mean']:.2f}% |")
    (DIRS["tables"] / f"{OUT_NAME}.md").write_text("\n".join(lines) + "\n",
                                                                  encoding="utf-8")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="HGT shill-attack robustness.")
    parser.add_argument("--rates", default="0.0,0.1,0.2")
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--adaptive", action="store_true",
                        help="Paraphrased reviews spread over weeks; results go to *_adaptive files.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    global ADAPTIVE, RUNS_DIR, OUT_NAME
    if args.adaptive:
        ADAPTIVE = True
        RUNS_DIR = DIRS["results"] / "robustness_runs_adaptive"
        OUT_NAME = "robustness_hgt_adaptive"

    if args.aggregate:
        aggregate()
        return 0

    setup_logging("INFO")
    ctx = ExperimentContext(device=args.device)
    cache: dict = {}
    for rate in [float(r) for r in args.rates.split(",")]:
        for condition in args.conditions.split(","):
            if rate == 0.0 and condition not in ("none", "trust", "trust+gate", "gate_v2"):
                continue   # oracle / inverted need injected labels
            for seed in [int(s) for s in args.seeds.split(",")]:
                run_one(rate, condition, seed, ctx, cache)
    aggregate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
