#!/usr/bin/env python
"""Detection of an attacker who knows the first detector exists (SYNTHETIC).

The standard synthetic campaign repeats eight templates in thirty-minute bursts, which
is exactly what exact-text and one-day-burst indicators look for, so a high score
against it says little. The adaptive campaign paraphrases every review and spreads
them over thirty days. This script scores the detectors against both:

  version 1   shared text, repeated text, one-day bursts
  version 2   version 1 plus seven-day bursts and a sentence-embedding comparison with
              other accounts' reviews

Everything here is labelled SYNTHETIC: it measures recognition of scripted attacks.

    python scripts/run_adaptive_attack.py --rate 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

import run_robustness  # noqa: E402
from src.anomaly.behavioural import BehaviouralTrustDetector  # noqa: E402
from src.anomaly.synthetic import SyntheticAnomalyInjector  # noqa: E402
from src.evaluation.evaluator import Vocabulary  # noqa: E402
from src.experiments.context import ExperimentContext  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Adaptive-attack detection (synthetic).")
    parser.add_argument("--rate", type=float, default=0.1)
    args = parser.parse_args(argv)

    ctx = ExperimentContext(device="cpu")
    report: dict = {"label": "SYNTHETIC", "rate": args.rate, "attacks": {}}
    for attack, adaptive in (("standard (templates, 30-minute bursts)", False),
                             ("adaptive (paraphrased, spread over 30 days)", True)):
        run_robustness.ADAPTIVE = adaptive
        config = ctx.experiment_config.with_overrides({
            "synthetic_anomaly.injection_rate": args.rate,
            "synthetic_anomaly.patterns.adaptive.enabled": adaptive})
        injection = SyntheticAnomalyInjector(config, seed=42).inject(ctx.train)
        poisoned = injection.interactions.reset_index(drop=True)
        from src.features.sbert import SBERTEncoder
        encoder = SBERTEncoder(ctx.model_config)
        started = time.perf_counter()
        embeddings = encoder.encode_frame(poisoned)       # cached by content hash
        vocab = Vocabulary.from_train(poisoned)
        labels = injection.label_array(vocab.user_ids)

        v1 = BehaviouralTrustDetector().fit_score(poisoned, vocab.user_ids)
        v2 = BehaviouralTrustDetector().fit_score(poisoned, vocab.user_ids, embeddings=embeddings)
        v2max = BehaviouralTrustDetector(combine="max").fit_score(poisoned, vocab.user_ids, embeddings=embeddings)
        rows = {"version 1 (mean of ranks)": v1.anomaly_scores, "version 2 (mean of ranks)": v2.anomaly_scores,
                "version 2 (strongest indicator)": v2max.anomaly_scores}
        for name in v2.indicators.columns:
            if name != "n_interactions":
                rows[f"indicator: {name}"] = v2.indicators[name].fillna(0.0).to_numpy()
        report["attacks"][attack] = {
            "shills": int(labels.sum()), "users": int(len(labels)), "seconds": round(time.perf_counter() - started, 1),
            "auc": {k: float(roc_auc_score(labels, v)) for k, v in rows.items()},
            "average_precision": {k: float(average_precision_score(labels, v)) for k, v in rows.items()},
        }

    (ctx.results_dir / "adaptive_attack_detection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Detecting a campaign that knows the detector (SYNTHETIC)", "",
             f"Injection rate {args.rate:.0%}. ROC-AUC against the injected labels (chance = 0.5).", "",
             "| Detector | " + " | ".join(report["attacks"]) + " |", "| --- |" + " ---: |" * len(report["attacks"])]
    names = list(next(iter(report["attacks"].values()))["auc"])
    for name in names:
        lines.append(f"| {name} | " + " | ".join(f"{a['auc'][name]:.4f}" for a in report["attacks"].values()) + " |")
    (ctx.tables_dir / "adaptive_attack_detection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
