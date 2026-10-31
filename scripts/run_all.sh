#!/bin/sh
# Reproduces every reported number, in dependency order. Run from anywhere.
#
# Roughly 6 hours on one Apple M4 (HGT trains on the GPU through MPS). Run the HGT
# stages one at a time: two HGT processes on the same GPU stall each other, and
# heavy CPU work alongside one slows it badly.
#
#   sh scripts/run_all.sh           # tune, then run the full grid and the studies
#   sh scripts/run_all.sh results   # skip tuning; reuse reports/results/tuning/
#
# A second dataset has its own script and its own output directories:
#   DYTRUSTREC_DATASET=Musical_Instruments sh scripts/run_second_dataset.sh
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python

if [ "$1" != "results" ]; then
  # Validation-only searches. Test data is never scored here.
  $PY scripts/tune_models.py --model knn --device cpu
  $PY scripts/tune_models.py --model mf --device cpu
  $PY scripts/tune_models.py --model lightgcn --device cpu
  $PY scripts/tune_models.py --model sasrec
  $PY scripts/tune_models.py --model hgt \
      --jobs base,keep_0.5,keep_0.9,keep_0.35,keep_0.25,lr_0.01,wd_1e-3,dropout_0.4,dropout_0.1,dropout_0.05,dropout_0.0,neg_4,neg_pop,layers_3,combined,rec_0.1,rec_0.25,rec_0.5,rec_1.0,rec_2.0,rec_4.0,trust_0.5,trust_1.0
fi

# Baselines, five seeds.
$PY scripts/run_experiments.py --only baselines

# HGT ablation arms: three seeds each (two for the CoLA control, G).
$PY scripts/run_experiments.py --only hgt --arms B,F --seeds 42,43,44
$PY scripts/run_experiments.py --only hgt --arms A,C,D,E,H --seeds 42,43,44
$PY scripts/tune_models.py --model hgt --jobs cos_150,cos_250
$PY scripts/run_experiments.py --only hgt --arms I --seeds 42,43,44
$PY scripts/run_experiments.py --only hgt --arms G --seeds 42,43
$PY scripts/run_experiments.py --aggregate

# Refit on train + validation for the fixed epoch count each model validated at.
$PY scripts/run_experiments.py --protocol refit --only baselines
$PY scripts/run_experiments.py --protocol refit --only hgt --arms B,D,E,F,I --seeds 42,43,44
$PY scripts/run_experiments.py --protocol refit --aggregate

# Studies that build on the trained models.
$PY scripts/run_synthetic_anomaly.py --device cpu
$PY scripts/run_robustness.py --rates 0.2 --seeds 42,43,44
$PY scripts/run_robustness.py --rates 0.0 --conditions trust,trust+gate --seeds 42,43
$PY scripts/run_adaptive_attack.py --rate 0.1
$PY scripts/run_robustness.py --adaptive --rates 0.2 --conditions none,trust+gate,gate_v2,oracle,oracle_gate --seeds 42,43,44,45
$PY scripts/run_robustness.py --adaptive --rates 0.0 --conditions gate_v2 --seeds 42,43
$PY scripts/export_model.py
$PY scripts/evaluate_confidence.py --arm E_hgt_sbert_trust_recency
$PY scripts/evaluate_cold_start.py --arm E_hgt_sbert_trust_recency
$PY scripts/explain_example.py
$PY scripts/benchmark_serving.py

# The explanation-fidelity test re-runs one trained model on edited graphs (needs its checkpoint).
$PY scripts/run_experiments.py --only hgt --arms E --seeds 42 --tag fidelity
$PY scripts/evaluate_explanations.py --users 300

# A re-run of arm B that keeps its training curves, for the masking figure.
$PY scripts/run_experiments.py --only hgt --arms B --seeds 42,43 --tag curves

# Figures, diagram and documents.
$PY docs/build/make_figures.py
$PY docs/diagrams/build_architecture.py
(cd docs/build && node make_paper.js && node build_explained.js)
