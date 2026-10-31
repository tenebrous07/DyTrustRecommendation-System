# DyTrustRec — Implementation Plan

Build order is strictly sequential. Each phase ends with tests passing before the next begins.
Status is updated only when a phase is actually built and verified — never in advance.

**Legend:** ✅ done and verified · 🔨 in progress · ⬜ not started

---

## Environment baseline (measured, not assumed)

| Property | Value | How it constrains the design |
|---|---|---|
| Hardware | Apple M4, 10 cores, 16 GB unified | Full-graph dense ops are out; sparse + mini-batch required |
| Free disk | 30 GB | Electronics (25.95 GB) cannot be materialised → streaming tier |
| Accelerator | MPS (no CUDA) | No `torch-scatter`/`torch-sparse` arm64 wheels; must avoid them |
| Python | 3.12.14 (uv-managed venv) | Meets the 3.10+ requirement |
| torch | 2.13.0 | HGTConv fwd+bwd **verified** on `mps` |
| torch-geometric | 2.8.0.post1 | Pure-Python layers; no compiled scatter needed |
| pyg-lib | 0.8.0+pt213 (`cp310-abi3` arm64 wheel) | **Unlocks `NeighborLoader`** → mini-batch training |
| SBERT | all-MiniLM-L6-v2 | 384-d, `max_seq_length` 256, **verified** |

Two findings changed the plan materially:

1. **`pyg-lib` has an arm64 macOS wheel for torch 2.13.** Without it, PyG raises
   `'NeighborSampler' requires either 'pyg-lib' or 'torch-sparse'` and the whole project
   would have been forced into full-batch training. With it, mini-batch neighbour sampling
   is available, so the dev tier is comfortably trainable.
2. **Amazon Reviews 2023 has no `brand` and no `seller` field.** The graph design lost the
   Seller node type as a result. See `docs/architecture.md` §3.

---

## Phase table

| Phase | Deliverable | Modules | Status |
|---|---|---|---|
| 1 | Dataset loader + cleaning | 1, 2 | ✅ **done** |
| 2 | EDA + analytics + outliers + graph stats | 3, 4 | ✅ **done** |
| 3 | Feature engineering | 5 | ✅ **done** |
| 4 | SBERT embeddings | 7 | ✅ **done** |
| 5 | Graph construction | 9 | ✅ **done** |
| 6 | Graph analytics | 10 | ✅ **done** |
| 7 | Baseline recommenders | 28 | ✅ **done** |
| 8 | CoLA anomaly detection | 11 | ✅ **done** |
| 9 | Trust weighting | 12 | ✅ **done** |
| 10 | HGT | 14 | ✅ **done** |
| 11 | Temporal weighting | 13 | ✅ **done** |
| 12 | Recommendation ranking | 15 | ✅ **done** |
| 13 | Explainability | 16 | ✅ **done** |
| 14 | Counterfactual reasoning | 17 | ✅ **done** |
| 15 | Confidence estimation | 18 | ✅ **done** |
| 16 | Evaluation + ablation | 19, 29 | ✅ **done** |
| 17 | Streamlit application | 39 | ✅ **done** |
| 18 | Documentation + viva notes | 47, 49 | ⬜ |

Interleaved, not separate phases: outlier detection (Module 4) lands with Phase 2, the
chronological splitter (Module 22) with Phase 5, cold start (Module 23) with Phase 12,
and the synthetic-anomaly protocol (Module 45) with Phase 16.

---

## Phase 1 — complete

**Built**

| File | Purpose |
|---|---|
| `configs/data_config.yaml` | Tiers, schema adapter map, cleaning thresholds, split policy |
| `src/utils/config.py` | Immutable config, dotted lookup, project-root resolution |
| `src/utils/logging.py` | Coloured console + DEBUG file log, `log_stage` timing |
| `src/utils/seed.py` | Seeding, environment fingerprint, run manifests, file hashing |
| `src/utils/device.py` | `auto/cpu/cuda/mps` resolution with MPS CPU-fallback |
| `src/data/validator.py` | Canonical schema, field detection, type coercion |
| `src/data/loader.py` | Acquisition, streaming, three sampling strategies, manifest |
| `src/data/cleaner.py` | Validation, dedup, flagging, k-core, cleaning report |
| `scripts/prepare_data.py` | CLI entrypoint with `--tier` and `--set` overrides |
| `tests/conftest.py`, `tests/test_data.py` | 64 tests |

**Verified output — dev tier (Video_Games), run 2026-08-24**

| Stage | Result |
|---|---|
| Source scanned | 4,624,615 interactions across 2,766,656 users (3.7 s, regex fast path, 0 slow-path lines) |
| Eligible users (≥5) | 117,742 |
| Sampled | 30,000 users → 262,547 interactions, **complete histories** |
| Product metadata | 47,089 / 47,089 items matched — **100 % coverage** |
| After cleaning | **144,178 interactions, 17,223 users, 7,692 items** (54.9 % retention) |
| Density | 1.088 × 10⁻³ |
| Time span | 1999-11-18 → 2023-08-29 (24 years — ample for temporal modelling) |
| k-core | converged in 10 passes; min user degree 5, min item degree 5 (asserted) |
| Flagged, retained | 39,261 rows (27.2 %) with explicit reasons |
| Brands after canonicalisation | 981 |
| Price missing | 25.5 % (never imputed; `price_missing` indicator carried) |

**Three defects found by running against real data, all fixed and pinned by regression tests**

1. *`user_core` truncated histories.* The row cap was applied by cutting the record stream
   part-way through, so users late in the file kept partial histories — destroying the exact
   property `user_core` exists to guarantee, and biasing the recency signal. The cap is now
   spent user-by-user: a user is either fully in or fully out.
2. *k-core shipped a violated invariant.* The ceiling was 10 passes with only a warning on
   failure, so a non-converged run silently returned data breaking the k-core guarantee —
   which is what happened on the first smoke run. It now raises, the ceiling is 100, and the
   post-condition is asserted before returning. The dev tier needs 10 passes, so the original
   ceiling was sitting exactly on the edge.
3. *`limits.max_products` was dead config.* It was declared in YAML and read by nothing. It is
   now a reported guard recorded in the run manifest, and it deliberately does **not**
   truncate: dropping items by popularity would bias the coverage and novelty metrics, and
   dropping users would undo fix 1. k-core reduces the catalogue on a principled criterion
   instead.

---

## Phase 2 — complete

**Built**

| File | Purpose |
|---|---|
| `configs/analytics_config.yaml` | EDA, outlier and graph parameters (documented 5th config) |
| `src/features/user_features.py` | Per-user behavioural profiles (pure aggregation) |
| `src/features/product_features.py` | Per-item profiles, category-level extraction, long-tail/Gini |
| `src/analytics/eda.py` | User/product/review/temporal analytics, 8 figures |
| `src/analytics/outliers.py` | IQR + log z-score + Isolation Forest + LOF, consensus |
| `src/analytics/graph_stats.py` | Bipartite graph metrics with labelled approximations |
| `scripts/run_eda.py` | CLI entrypoint |
| `notebooks/01_data_understanding.ipynb`, `02_eda.ipynb` | Presentation surfaces, no logic |
| `tests/test_analytics.py` | 43 tests (109 total) |

**Measured results — dev tier**

| Finding | Value | Why it matters downstream |
|---|---|---|
| Popularity Gini | **0.521** (top 20% of items hold 59.5%) | Popularity baseline will be strong; beating it is the real bar |
| 5-star share | **64.8%** (mean 4.29) | Rating is a feature, not a graded relevance label |
| Verified purchases | 78.4%, rating **+0.503** vs unverified | Justifies `verified_purchase` as a trust input |
| **Per-user category drift** | **0.433** (total-variation, first vs second half of history) | **Measured** premise for temporal weighting, not an assumption |
| Connected components | **1** (100% of nodes) | No isolated users; cold start is genuinely new-user, not disconnection |
| Louvain modularity | **0.458** across 11 communities | Real community structure for a graph model to exploit |
| Degree assortativity | **−0.137** (disassortative) | Active users attach to popular items — another reason popularity is hard to beat |
| Bipartite clustering | 0.058 (triangle clustering is 0 *by construction*) | Reported correctly rather than as a false finding |
| Corroborated anomalies | 3,218 users (18.7%) by ≥2 independent layers | Ranking for review, explicitly not a fraud verdict |

The category-trend figure independently corroborates the drift number: the "legacy
systems" share falls from 0.88 to ~0.2, Nintendo Switch appears as a step in **2017** and
PlayStation 5 in **2020** — matching the real console launches. The temporal component is
modelling something that demonstrably exists in this data.

**Design changes forced by the data**

1. *Category nodes will not come from `main_category`.* It is 80% a single value inside a
   single-category corpus. The `categories` path is populated for 97.5% of products and its
   levels are **orthogonal, not merely coarse-to-fine**: L2 is the platform (38 values —
   "playstation 4", "xbox one"), the leaf is the product type (118 — "games", "headsets").
   `extract_category_levels(node_level=...)` exposes the choice; Phase 5 should ablate it,
   and may find both deserve their own relation.
2. *IQR flags 54.9% of users on its own.* That is a multiple-comparisons artefact — the rule
   fires when any of 12 features leaves its fence. The report now says so, and the
   corroborated count is the number to quote.
3. *Isolation Forest and LOF agree on only 6.2% of flags (Jaccard).* Expected — they model
   different notions of outlyingness — and the reason the consensus count exists rather
   than a single detector.

**Two defects caught while writing the module**

1. *LOF was refit on every `score()` call* in transductive mode, so flags could differ
   between calls on identical input. Labels and scores are now captured once at fit time.
2. *`novelty=True` was going to be used to score the fitted rows*, which scikit-learn
   documents as invalid. The mode is now derived from `fit_on_train_split_only`:
   transductive (`novelty=False`) for exploration, inductive for scoring unseen users, and
   a transductive detector raises if asked to score rows it was not fitted on.

---

## Phases 3–18 — complete

All phases implemented and run. Tests pass (see README). ~12,000 lines of Python.

### Headline outcome — first grid withdrawn, second grid measured

The first ablation grid (single seed, untuned baselines) reported that no HGT arm beat matrix
factorisation and that trust weighting cost 17–20%. An audit then found why that grid could not be
believed: trust and recency were stored as edge weights that `HGTConv` cannot read, so arms C, D and
E were the same model as B and their differences were seed noise; early stopping existed but was never
called; the baselines had no search (LightGCN scored below popularity); and the supervision edge was
visible to message passing. All four were fixed (see the README, §20).

The repaired grid (tuned baselines, validation-only search, 3–5 seeds, paired bootstrap):

| Step | Change | Effect on test NDCG@10 |
|---|---|---|
| A → B | add SBERT semantics | about +7% (within noise of the seed spread) |
| B → H | remove edge masking | **−32%**: an unmasked HGT equals matrix factorisation |
| B → C | add behavioural trust weighting | about −3% (within noise, clean data) |
| B → D | add recency weighting | +6% on test, +43% on validation (horizon gap) |
| D → E | add trust on top of recency | +4% (within noise) |
| E → F | add the learned fusion layer | about −7%: fusion does not beat concatenation |

Best single arm E 0.0308 and a three-seed ensemble 0.0334, against 0.0189 for the best tuned baseline.
Numbers, intervals and the refit protocol are in `reports/tables/model_comparison.md`,
`reports/tables/model_comparison_refit.md` and the paper.

### Defects found by running, not by reading

1. **LightGCN never converged.** BPR loss pinned at exactly ln(2)=0.69315 — positive and
   negative scores identical. Diagnosed: layer-averaging shrinks embedding norms so dot
   products are ~1e-3, sigmoid sits at 0.5 and gradients vanish. Measured: at lr=1e-3 the
   item-embedding norm decays to 0.04; at lr=0.05 with no weight decay it reaches 4.27 and
   trains. Fixed and documented in the source.
2. **HGT crashed on MPS.** Installing `pyg-lib` (for `NeighborLoader`) routed `HGTConv`
   through `pyg::segment_matmul`, which has no MPS kernel; `PYTORCH_ENABLE_MPS_FALLBACK`
   does not help because it is a custom pyg op, not an aten op. Fixed by setting
   `torch_geometric.backend.use_segment_matmul = False` on MPS — NeighborLoader is retained.
3. **A configured feature was silently dropped.** `fusion.structured_features` named
   `verified_purchase` but the builder produces `verified_purchase_f`, so review nodes were
   389-d instead of 390-d. The mismatch is now fatal rather than a warning.

### Known limitations of the results

- Two datasets, one split each; on Video_Games only 1,862 of 17,223 users are scoreable on the test split (6,101
  cold-user and 8,603 cold-item rows excluded). The second dataset (Musical_Instruments) repeats some findings and
  not others: the single-model lead over the best baseline is +6% there (not distinguishable from zero) against +52%.
- Three seeds per HGT arm; one-at-a-time hyper-parameter search on a single seed.
- Trust is validated only on synthetic campaigns scripted by the same author as the defence.
- The confidence score tracks evidence volume, not correctness (AUROC 0.456 for predicting hits).
- No sequence-model or LLM baselines; the protocol differs from the leave-one-out protocol of most
  published results on this dataset, so no claim against published numbers is made.

### Next steps

1. Real manipulation data or a published attack benchmark; a setting where the scripted attack actually lowers accuracy.
2. A content path for cold users and items (about 60% of test rows are excluded as cold).
3. Learned temporal encodings inside the attention in place of one tuned decay rate.
4. A fidelity test for explanations and counterfactuals (remove the cited interactions and re-score).
5. A third category and a non-Amazon domain. A second Amazon category is done (`scripts/run_second_dataset.sh`).

## Superseded plan for Phase 3

1. `src/data/splitter.py` — the chronological split. Must come before any leakage-sensitive
   feature is fitted, because "train-only" is undefined until the split exists.
2. `src/features/review_features.py` — length, helpfulness, recency, rating centred on the
   item mean.
3. `src/features/sentiment.py` — VADER (Module 6). Auxiliary feature, never the recommender.
   Adds the sentiment panel the EDA currently declares missing.
4. The **transform** half of `user_features.py` / `product_features.py` — scalers and
   encodings with an explicit fit/transform split, fitted on the training partition only.
5. Re-run outlier detection with `fit_on_train_split_only: true` so the scores are usable as
   model features rather than exploratory only.
6. `tests/test_features.py`, including an explicit assertion that no transform has seen
   validation or test rows.

**Exit criterion:** a leakage test proves every fitted statistic derives from train rows
alone, and `pytest` is green.

---

## Standing rules for every remaining phase

- Explain the maths, state the input/output tensor shapes, build the smallest working
  version, unit-test it, run a tiny example, verify shapes — *then* integrate.
- No result is written to `README.md` or `reports/` until the run that produced it exists.
- Any approximation gets recorded where it is made and surfaces in the README limitations.
- If a dataset field is missing, adapt the pipeline. Never invent the field.
