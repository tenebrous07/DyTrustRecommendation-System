#!/bin/sh
# Replicates the main findings on a second Amazon Reviews 2023 category.
#
#   DYTRUSTREC_DATASET=Musical_Instruments sh scripts/run_second_dataset.sh [first-stage]
#
# Stages, in order: prepare, graph, tune, tune-hgt, baselines, arms, defence, studies.
# Passing a stage name starts there (earlier stages are assumed done).
#
# Protocol: every model family, baselines and HGT alike, is tuned on THIS dataset's
# validation split with the same search grids as Video_Games. Test data is never scored
# during tuning. Everything goes to its own directories (reports/results_<slug>,
# data/processed_<slug>, checkpoints_<slug>); the Video_Games artefacts are never read or
# written.
#
# Run the HGT stages one at a time: two HGT processes on one MPS GPU stall each other.
set -e
cd "$(dirname "$0")/.."
: "${DYTRUSTREC_DATASET:?set DYTRUSTREC_DATASET to an Amazon 2023 category, e.g. Musical_Instruments}"
export DYTRUSTREC_DATASET
PY=.venv/bin/python
STAGES="prepare graph tune tune-hgt baselines arms defence studies"
START="${1:-prepare}"
case " $STAGES " in *" $START "*) ;; *) echo "unknown stage '$START'; choose from: $STAGES" >&2; exit 2;; esac

active=0
stage() {                       # stage NAME: true once NAME (or a later one) is reached
  [ "$active" = 1 ] && return 0
  if [ "$1" = "$START" ]; then active=1; return 0; fi
  return 1
}

if stage prepare; then
  $PY scripts/prepare_data.py --tier dev
fi
if stage graph; then
  $PY scripts/build_graph.py
fi
if stage tune; then
  $PY scripts/tune_models.py --model knn --device cpu
  $PY scripts/tune_models.py --model mf --device cpu
  $PY scripts/tune_models.py --model lightgcn --device cpu
  $PY scripts/tune_models.py --model sasrec
fi
if stage tune-hgt; then
  $PY scripts/tune_models.py --model hgt \
      --jobs base,keep_0.5,keep_0.9,keep_0.35,keep_0.25,lr_0.01,wd_1e-3,dropout_0.4,dropout_0.1,dropout_0.05,dropout_0.0,neg_4,neg_pop,layers_3,combined,rec_0.1,rec_0.25,rec_0.5,rec_1.0,rec_2.0,rec_4.0,trust_0.5,trust_1.0
fi
if stage baselines; then
  $PY scripts/run_experiments.py --only baselines
fi
if stage arms; then
  $PY scripts/run_experiments.py --only hgt --arms B,E --seeds 42,43,44
  $PY scripts/run_experiments.py --only hgt --arms A,C,D,F,H --seeds 42,43,44
  $PY scripts/run_experiments.py --aggregate
fi
if stage defence; then
  $PY scripts/run_synthetic_anomaly.py --device cpu
  $PY scripts/run_adaptive_attack.py --rate 0.1
  $PY scripts/run_robustness.py --adaptive --rates 0.2 --conditions none,trust+gate,gate_v2,oracle,oracle_gate --seeds 42,43,44,45,46
  $PY scripts/run_robustness.py --adaptive --rates 0.0 --conditions gate_v2 --seeds 42,43
fi
if stage studies; then
  $PY scripts/evaluate_cold_start.py --arm E_hgt_sbert_trust_recency
  $PY scripts/evaluate_confidence.py --arm E_hgt_sbert_trust_recency
  $PY scripts/evaluate_explanations.py --users 300
fi
