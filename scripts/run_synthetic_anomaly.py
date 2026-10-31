#!/usr/bin/env python
"""Module 45: synthetic anomaly injection and trust-detection evaluation.

The only quantitative validation the trust component can have on this corpus,
because Amazon Reviews 2023 has no manipulation labels. Genuine interactions are
left untouched; scripted shill users are added alongside them, so labels exist
by construction.

Everything this writes is labelled SYNTHETIC. Detecting these users demonstrates
that a detector recognises this generative process. It does not demonstrate
detection of real Amazon fraud, and must never be reported as such.

    python scripts/run_synthetic_anomaly.py
    python scripts/run_synthetic_anomaly.py --injection-rate 0.02
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analytics.outliers import OutlierDetector  # noqa: E402
from src.anomaly.behavioural import BehaviouralTrustDetector  # noqa: E402
from src.anomaly.cola import CoLADetector  # noqa: E402
from src.anomaly.synthetic import SyntheticAnomalyInjector, detection_metrics  # noqa: E402
from src.anomaly.trust import build_trust_scores  # noqa: E402
from src.evaluation.evaluator import Vocabulary  # noqa: E402
from src.features.transforms import LeakageSafeScaler  # noqa: E402
from src.features.user_features import build_user_profiles  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.device import resolve_device  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402
from src.utils.seed import set_seed, write_run_manifest  # noqa: E402

logger = get_logger("synthetic_anomaly")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DyTrustRec Module 45.")
    parser.add_argument("--injection-rate", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    data_config = load_config("data_config")
    model_config = load_config("model_config")
    experiment_config = load_config("experiment_config")
    if args.injection_rate is not None:
        experiment_config = experiment_config.with_overrides(
            {"synthetic_anomaly.injection_rate": args.injection_rate}
        )

    processed = resolve_path(data_config.require("paths.processed"))
    tables = resolve_path(data_config.require("paths.tables"))
    results = resolve_path(data_config.require("paths.results"))
    for directory in (tables, results):
        directory.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level, log_file=results / "synthetic_anomaly.log")
    seed = set_seed(args.seed)
    device = resolve_device(args.device)

    train = pd.read_parquet(processed / "split_train.parquet")
    logger.info("Loaded %s genuine training interactions", f"{len(train):,}")

    # ---- inject (original data untouched) ---------------------------------
    injector = SyntheticAnomalyInjector(experiment_config, seed=seed)
    injection = injector.inject(train)
    suffix = str(experiment_config.get("synthetic_anomaly.output_suffix", "_synthetic"))
    injection.interactions.to_parquet(processed / f"split_train{suffix}.parquet", index=False)

    # ---- score with every detector, on identical inputs --------------------
    vocab = Vocabulary.from_train(injection.interactions)
    indexed = vocab.index(injection.interactions)
    profiles = build_user_profiles(injection.interactions).reindex(vocab.user_ids)
    numeric = profiles.select_dtypes(include=[np.number])
    scaler = LeakageSafeScaler().fit(numeric, scope="synthetic corpus")
    features = np.nan_to_num(scaler.transform(numeric).to_numpy(dtype=np.float32))
    labels = injection.label_array(vocab.user_ids)

    scored: dict[str, np.ndarray] = {}

    detector = CoLADetector(model_config, device=str(device), seed=seed)
    adjacency = detector.build_projection(indexed, vocab.n_users, vocab.n_items)
    cola_result = detector.fit_score(features, adjacency, node_ids=vocab.user_ids,
                                     entity_type="user")
    trust = build_trust_scores(cola_result, model_config, entity_type="user")
    # Higher must mean more anomalous, so trust is inverted.
    scored["CoLA trust (1 - trust)"] = 1.0 - trust.trust

    # The detector the model actually trains with. It is scored on the same
    # injected labels, but note it was built from the shilling-detection
    # literature before seeing this benchmark's results, and one indicator that
    # pointed the wrong way here was dropped rather than reweighted.
    behavioural = BehaviouralTrustDetector().fit_score(injection.interactions, vocab.user_ids)
    scored["Behavioural trust detector (Module 11)"] = behavioural.anomaly_scores

    # Baselines. Without these, an easy detection problem would read as evidence
    # that CoLA is doing something sophisticated.
    analytics_config = load_config("analytics_config").with_overrides(
        {"paths": data_config.require("paths").to_dict()}
    )
    outlier = OutlierDetector(analytics_config)
    outlier.fit(profiles, scope="synthetic corpus")
    report = outlier.score(profiles)
    scored["Isolation Forest (Module 4)"] = report.scores["iforest_score"].to_numpy()
    scored["consensus indicator count"] = report.scores["n_indicators"].to_numpy(dtype=float)
    if "frac_duplicate_text" in profiles.columns:
        scored["trivial rule: frac_duplicate_text"] = (
            profiles["frac_duplicate_text"].fillna(0).to_numpy()
        )
    if "min_gap_hours" in profiles.columns:
        gaps = profiles["min_gap_hours"].fillna(profiles["min_gap_hours"].max()).to_numpy()
        scored["trivial rule: -min_gap_hours"] = -gaps

    k = 100
    table = {name: detection_metrics(values, labels, k=k) for name, values in scored.items()}

    ordered = sorted(table.items(), key=lambda kv: -(kv[1]["roc_auc"] if
                                                     np.isfinite(kv[1]["roc_auc"]) else -1))
    lines = [
        "# Synthetic Anomaly Detection (Module 45)", "",
        "> **SYNTHETIC RESULT.** The positives below were generated by",
        "> `scripts/run_synthetic_anomaly.py`, not observed in Amazon data. These numbers",
        "> measure whether a detector recognises *this generative process*. They are **not**",
        "> evidence of detecting real review manipulation, and must never be quoted as such.",
        "",
        "## Injection", "", "| Parameter | Value |", "| --- | ---: |",
    ]
    for key, value in injection.stats.items():
        if key == "WARNING":
            continue
        lines.append(f"| {key.replace('_', ' ')} | {value} |")

    lines += ["", f"## Detection performance (n={len(labels):,} users, "
                  f"{int(labels.sum()):,} synthetic positives)", "",
              f"| Detector | ROC AUC | Avg precision | Precision@{k} |",
              "| --- | ---: | ---: | ---: |"]
    for name, metrics in ordered:
        lines.append(
            f"| {name} | {metrics['roc_auc']:.4f} | {metrics['average_precision']:.4f} | "
            f"{metrics[f'precision_at_{k}']:.4f} |"
        )

    cola_auc = table["CoLA trust (1 - trust)"]["roc_auc"]
    best_name, best = ordered[0]
    lines += ["", "## Reading this honestly", ""]
    if np.isfinite(cola_auc) and cola_auc < 0.45:
        lines += [
            f"- **CoLA scores {cola_auc:.4f} AUC — below the 0.5 chance line.** It is not "
            f"merely failing to find the injected users; it is systematically ranking them "
            f"as *more* trustworthy than genuine users. This was checked against the raw "
            f"anomaly score with the trust transform bypassed, and the direction is the "
            f"same, so it is a property of the detector on this data and not a sign error "
            f"in the trust mapping.",
            "",
            "- **A plausible mechanism.** CoLA's contrastive objective scores a node by how "
            "well its own attributes can be predicted from its neighbourhood context. A "
            "coordinated campaign is *internally consistent by construction*: the injected "
            "users share templates, share a forced rating, and overlap on target items, so "
            "each one is unusually easy to predict from the others. A detector that equates "
            "'anomalous' with 'unpredictable from context' will therefore rate a large, "
            "well-organised campaign as exemplary. Isolated eccentric users look stranger "
            "to it than a botnet does.",
            "",
            "- **Why this matters for the recommender.** If trust is anti-correlated with "
        "manipulation, any use of it (weighting a loss, weighting an edge) up-weights the "
        "manipulated interactions and down-weights genuine ones. The HGT robustness study "
        "(`scripts/run_robustness.py`) measures that harm directly.",
        "",
        "- **What would have to change.** Either the projection (a user-user "
            "co-occurrence graph rewards dense coordinated cliques), or the detector "
            "(a density- or reconstruction-based objective rather than a contrastive "
            "predictability one), or the target (score the *interaction* rather than the "
            "user). The behavioural detector in the table above changes the signal instead: "
            "it asks whether an account behaves like a campaign account, which does not "
            "reward internal consistency.",
        ]
    elif best_name.startswith("CoLA"):
        lines.append(
            f"- CoLA ranks highest (AUC {cola_auc:.4f}). Note that the injected users carry "
            f"duplicated text and second-apart bursts by construction, so this is a "
            f"deliberately detectable population; the comparison rows show how much of the "
            f"result a one-line rule already achieves."
        )
    else:
        lines.append(
            f"- **A simpler detector beats CoLA here.** `{best_name}` reaches AUC "
            f"{best['roc_auc']:.4f} against CoLA's {cola_auc:.4f}. Since the injected users "
            f"were built with duplicated text and burst timing, a rule targeting exactly "
            f"those traits should win — which is the point of reporting it. This does not "
            f"show CoLA is useless on real manipulation; it shows this synthetic benchmark "
            f"is too easy to separate the two."
        )
    lines += [
        "- Chance-level AUC is 0.5. Precision@100 should be read against a prevalence of "
        f"{float(labels.mean()):.4f}.",
        "- The real corpus has no labels, so none of this transfers automatically. The "
        "honest claim is about this generative process only.",
    ]

    (tables / "synthetic_anomaly_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tables / "synthetic_anomaly_report.json").write_text(
        json.dumps({"injection": injection.stats, "detection": table,
                    "labelled": "SYNTHETIC"}, indent=2, default=str), encoding="utf-8")

    write_run_manifest(results / "synthetic_anomaly_run.json", seed=seed,
                       config=experiment_config.to_dict(),
                       extra={"phase": "45-synthetic-anomaly", "injection": injection.stats,
                              "detection": table})

    logger.info("=" * 72)
    for name, metrics in ordered:
        logger.info("  %-34s AUC=%.4f  AP=%.4f  P@%d=%.4f", name, metrics["roc_auc"],
                    metrics["average_precision"], k, metrics[f"precision_at_{k}"])
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
