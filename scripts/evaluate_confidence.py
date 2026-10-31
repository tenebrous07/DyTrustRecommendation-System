#!/usr/bin/env python
"""Does the confidence score mean anything?

A confidence that never varies, or varies independently of whether the
recommendation was right, is decoration. This checks the Module 18 blend on the
held-out test split: for every (user, top-10 item) pair of a seed ensemble, did
the user actually interact with the item, and does higher confidence go with a
higher hit rate?

Predictive stability comes from disagreement between independently trained
ensemble members (the same normalisation as MC dropout). Evidence volume and
evidence trust are the user-level components. The components are also scored
alone, so it is clear which part carries the signal.

    python scripts/evaluate_confidence.py --arm B_hgt_sbert
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import active_dirs  # noqa: E402

DIRS = active_dirs()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.experiments.context import ExperimentContext  # noqa: E402
from src.explainability.explanations import Explainer  # noqa: E402
from src.explainability.graph_paths import EvidenceRetriever  # noqa: E402
from src.features.product_features import extract_category_levels  # noqa: E402
from src.models.ensemble import EnsembleRecommender  # noqa: E402
from src.recommendation.calibration import HitProbabilityCalibrator, expected_calibration_error  # noqa: E402
from src.recommendation.confidence import ConfidenceEstimator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = DIRS["results"] / "runs"
TOP_K = 10


def load_members(arm: str) -> list[tuple[np.ndarray, np.ndarray]]:
    members = []
    for path in sorted(RUNS_DIR.glob(f"{arm}__seed*.emb.npz")):
        with np.load(path) as data:
            members.append((data["user"].astype(np.float32), data["item"].astype(np.float32)))
    if len(members) < 3:
        raise SystemExit(f"{arm}: need at least 3 seeds with saved embeddings, found {len(members)}.")
    return members


def zscore(scores: np.ndarray) -> np.ndarray:
    spread = scores.std(axis=1, keepdims=True)
    return (scores - scores.mean(axis=1, keepdims=True)) / np.where(spread > 0, spread, 1.0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Confidence usefulness on the test split.")
    parser.add_argument("--arm", required=True, help="Run name prefix, e.g. B_hgt_sbert")
    args = parser.parse_args(argv)

    ctx = ExperimentContext(device="cpu")
    members = load_members(args.arm)
    evaluator = ctx.test_eval

    trust_path = ctx.tables_dir / "trust_scores_behavioural.csv"
    trust_map = (dict(zip(*pd.read_csv(trust_path)[["entity_id", "trust_score"]].T.values))
                 if trust_path.is_file() else {})
    counts = ctx.train.groupby("user_id").size().to_dict()
    estimator = ConfidenceEstimator(ctx.model_config, user_interaction_counts=counts,
                                    trust_map=trust_map)

    products = extract_category_levels(
        ctx.products, node_level=str(ctx.model_config.get("graph.category_level", "l2")))
    ensemble = EnsembleRecommender(ctx.vocab.n_users, ctx.vocab.n_items, members)
    explainer = Explainer(EvidenceRetriever(ctx.train, products, ctx.vocab, trust_map=trust_map))

    def build_pairs(evaluator, explain: bool) -> list[dict]:
        out = []
        for users, ranked in evaluator._ranked_batches(ensemble, TOP_K):
            member_scores = [zscore(ue[users] @ ie.T) for ue, ie in members]
            mean_scores = np.mean(member_scores, axis=0)
            for row, user in enumerate(users):
                top = ranked[row]
                stability = ConfidenceEstimator.ensemble_stability(
                    np.stack([m[row, top] for m in member_scores]))
                user_id = ctx.vocab.user_ids[int(user)]
                estimates = estimator.estimate(None, user_id, int(user), top,
                                               [ctx.vocab.item_ids[i] for i in top], stability=stability)
                truth = evaluator.ground_truth[int(user)]
                for position, (item, est) in enumerate(zip(top, estimates)):
                    record = {
                        "user": int(user), "rank": position + 1, "hit": int(int(item) in truth),
                        "confidence": est.confidence, "stability": est.components.get("predictive_stability", np.nan),
                        "volume": est.components["evidence_volume"], "trust": est.components["evidence_trust"],
                        "score": float(mean_scores[row, item]), "n_history": counts.get(user_id, 0),
                        "cold": int(counts.get(user_id, 0) < 3)}
                    if explain:
                        explained = explainer.explain(user_id, ctx.vocab.item_ids[int(item)])
                        record.update(supported=int(explained.has_support), n_evidence=len(explained.evidence))
                    out.append(record)
        return out

    rows = build_pairs(evaluator, explain=True)
    val_frame = pd.DataFrame(build_pairs(ctx.val_eval, explain=False))
    frame = pd.DataFrame(rows)

    # ---- a calibrated hit probability, fitted on validation pairs and scored once on test
    from sklearn.metrics import brier_score_loss
    calibrator = HitProbabilityCalibrator().fit(val_frame)
    frame["p_hit"] = calibrator.predict(frame)
    base = float(val_frame["hit"].mean())          # the only thing known without the model
    calibrated = {
        "auroc": float(roc_auc_score(frame["hit"], frame["p_hit"])),
        "brier": float(brier_score_loss(frame["hit"], frame["p_hit"])),
        "brier_constant_base_rate": float(brier_score_loss(frame["hit"], np.full(len(frame), base))),
        "ece": float(expected_calibration_error(frame["p_hit"].to_numpy(), frame["hit"].to_numpy())),
        "validation_pairs": int(len(val_frame)), "validation_base_rate": base,
        "weights": calibrator.weights(),
    }
    frame["p_bin"] = pd.qcut(frame["p_hit"].rank(method="first"), 5, labels=False)
    calibrated["reliability"] = frame.groupby("p_bin").agg(
        pairs=("hit", "size"), mean_predicted=("p_hit", "mean"), observed=("hit", "mean")
    ).reset_index().to_dict(orient="records")

    def auc(column: str) -> float:
        return float(roc_auc_score(frame["hit"], frame[column]))

    frame["bin"] = pd.qcut(frame["confidence"].rank(method="first"), 5, labels=False)
    reliability = frame.groupby("bin").agg(pairs=("hit", "size"), hit_rate=("hit", "mean"),
                                           mean_confidence=("confidence", "mean"))
    cold = frame[frame["cold"] == 1]
    warm = frame[frame["cold"] == 0]
    supported = frame[frame["supported"] == 1]
    unsupported = frame[frame["supported"] == 0]
    explanation = {
        "pairs_with_evidence": float(frame["supported"].mean()),
        "hit_rate_supported": float(supported["hit"].mean()) if len(supported) else None,
        "hit_rate_unsupported": float(unsupported["hit"].mean()) if len(unsupported) else None,
        "mean_evidence_items": float(frame["n_evidence"].mean()),
    }
    result = {
        "calibrated": calibrated,
        "explanation": explanation,
        "arm": args.arm, "members": len(members), "pairs": int(len(frame)),
        "users": int(frame["user"].nunique()), "base_hit_rate": float(frame["hit"].mean()),
        "auroc": {name: auc(name) for name in ("confidence", "stability", "volume", "trust", "score")},
        "reliability": reliability.reset_index().to_dict(orient="records"),
        "cold_start": {"cold_pairs": int(len(cold)),
                       "mean_confidence_cold": float(cold["confidence"].mean()) if len(cold) else None,
                       "mean_confidence_warm": float(warm["confidence"].mean())},
    }
    (ctx.results_dir / "confidence_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = ["# Confidence usefulness (test split)", "",
             f"Seed ensemble of {len(members)} runs of `{args.arm}`; {result['pairs']:,} (user, top-{TOP_K} item) "
             f"pairs over {result['users']:,} users; base hit rate {result['base_hit_rate']:.4f}.", "",
             "AUROC for predicting whether a recommended item was actually interacted with (0.5 = no signal):", "",
             "| Signal | AUROC |", "| --- | ---: |"]
    for name, value in result["auroc"].items():
        lines.append(f"| {name} | {value:.3f} |")
    lines += ["", "Hit rate by confidence quintile (low to high):", "",
              "| Quintile | pairs | mean confidence | hit rate |", "| ---: | ---: | ---: | ---: |"]
    for r in result["reliability"]:
        lines.append(f"| {int(r['bin']) + 1} | {int(r['pairs']):,} | {r['mean_confidence']:.3f} | {r['hit_rate']:.4f} |")
    c = result["calibrated"]
    lines += ["", "## A calibrated hit probability (fitted on validation recommendations, scored once on test)", "",
              f"AUROC {c['auroc']:.3f}; Brier score {c['brier']:.5f} against {c['brier_constant_base_rate']:.5f} for always "
              f"predicting the base rate; expected calibration error {c['ece']:.5f} (base rate {result['base_hit_rate']:.4f}).", "",
              "| Quintile of predicted probability | pairs | mean predicted | observed hit rate |", "| ---: | ---: | ---: | ---: |"]
    for r in c["reliability"]:
        lines.append(f"| {int(r['p_bin']) + 1} | {int(r['pairs']):,} | {r['mean_predicted']:.4f} | {r['observed']:.4f} |")
    lines.append("")
    lines.append("Logistic-regression weights on standardised features: "
                 + ", ".join(f"{k} {v:+.2f}" for k, v in c["weights"].items()) + ".")
    ex = result["explanation"]
    lines += ["", "Explanation coverage: "
              f"{100 * ex['pairs_with_evidence']:.1f}% of recommendations had at least one retrieved evidence item "
              f"(mean {ex['mean_evidence_items']:.2f} items per recommendation). Hit rate "
              + (f"{ex['hit_rate_supported']:.4f} with evidence" if ex["hit_rate_supported"] is not None else "n/a")
              + (f" against {ex['hit_rate_unsupported']:.4f} without." if ex["hit_rate_unsupported"] is not None else ".")]
    cs = result["cold_start"]
    if cs["mean_confidence_cold"] is not None:
        lines += ["", f"Cold-start check: mean confidence {cs['mean_confidence_cold']:.3f} for users with fewer than 3 "
                      f"training interactions ({cs['cold_pairs']:,} pairs) against {cs['mean_confidence_warm']:.3f} for the rest."]
    (ctx.tables_dir / "confidence_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
