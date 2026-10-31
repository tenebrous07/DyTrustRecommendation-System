# DyTrustRec

**Dynamic Trust-Aware Explainable Heterogeneous Graph Recommendation System**

> **Status: implemented, repaired and re-evaluated end to end.** Everything below is from real
> runs on the dev tier (Video_Games), repeated on a second category (Musical_Instruments), with tuned baselines, multiple seeds and a validation-only
> search. Test data is scored once per run. **269 passing tests.**
>
> **Headline.** An HGT recommender trained the usual way is no better than matrix factorisation
> (NDCG@10 0.0189 vs 0.0189). Hiding a random share of the interaction
> edges from message passing each epoch takes the same architecture to 0.0278; adding recency and
> trust weighting of the objective reaches 0.0308 (0.0334 as a three-seed ensemble), **+64% over the best tuned baseline**.
> The learned fusion layer does not beat concatenation. CoLA rates a coordinated shill campaign
> as *more* trustworthy than genuine users (AUC 0.295); three behavioural indicators reach 0.9997.
> Under a 20% scripted shill campaign the HGT loses about 10% NDCG@10, and no trust variant that only
> re-weights the loss, even one given the true labels, recovers more than seed noise. Against a campaign that
> paraphrases its reviews, exact-text matching fails (AUC 0.52) and a sentence-embedding gate holds (0.99) and
> restores accuracy to the level of a label-knowing oracle on Video_Games (suggestive, about 1.7 standard errors; not reproduced on the second dataset). A separate strip of new products ranked by text
> lifts mixed-list NDCG@10 by about 22%. The evidence an explanation cites matters to the model (removal test),
> but the confidence score tracks evidence volume, **not** correctness; a separately calibrated hit probability
> ranks well but is not calibrated in level. See §20.
>
> **Second dataset.** A replication on a second category (Musical_Instruments) reproduces the training-recipe and explanation findings but with a much smaller lead over the best baseline (+6% for a single model, not distinguishable from zero; +19% for the three-seed ensemble), and shows no accuracy gain from the defence or the new-product strip. The recipe's largest effect (edge masking), the embedding detector and the explanation test do repeat; the size of the lead, the new-product gain and the defence's accuracy benefit do not (§20, "A second dataset").
>
> An earlier version of this README reported that no HGT arm beat matrix factorisation and that
> trust weighting cost 17%. Those results came from a flawed grid (see §20, "What was wrong
> before") and are withdrawn.

---

## 1. Abstract

DyTrustRec is a recommendation pipeline that treats a user–item interaction not as a fact but
as *evidence of varying reliability and varying age*. It combines semantic review
representations (SBERT), a behavioural trust score computed from interaction history,
exponential recency weighting of the training objective, and a Heterogeneous Graph Transformer
(HGT) recommender trained with edge masking, and it accompanies each recommendation with a
graph-grounded explanation, a counterfactual "why not", and a confidence estimate computed
separately from the ranking score.

The individual techniques are existing published work. The contribution is an audit-and-repair
of the pipeline and what the repaired measurements show (§20): what edge masking is worth, how
validation overstates recency, why CoLA fails as a trust source and what replaces it, and what
the confidence score does and does not mean.

## 2. Problem statement

A conventional recommender answers one question: *what will this user like?* It cannot say why
it believes that, how much of its evidence comes from reviews that may be manipulated, whether
the user's taste has since moved on, or how confident it is. For a purchase decision, those
four gaps matter as much as the ranking itself.

## 3. Motivation

Three properties of real review data motivate the design:

1. **Not all interactions are equally trustworthy.** Review manipulation is well documented on
   large marketplaces. A recommender that weights a coordinated shill campaign identically to
   an organic purchase inherits the manipulation.
2. **Interest drifts.** A user who bought gaming hardware in 2019 and photography gear in 2023
   is poorly served by a model that treats both equally.
3. **Opaque recommendations are hard to act on.** "Because people like you bought it" is not
   evidence a user can evaluate.

## 4. Existing approaches

| Family | Example | Limitation this project targets |
|---|---|---|
| Neighbourhood CF | Item-kNN | No content, no trust, cold-start fails |
| Latent factor | BPR-MF | No side information, no explanation |
| Graph CF | LightGCN | Homogeneous, no semantics, no trust |
| Heterogeneous GNN | HGT | Rich structure, but treats all edges as equally reliable and equally recent |
| Review-based | DeepCoNN, NARRE | Uses text, but not graph structure or interaction reliability |

## 5. Research gap

Trust-aware recommendation and heterogeneous graph recommendation are both established
literatures, but trust is usually taken from an explicit social trust network — which most
e-commerce datasets, including this one, do not have. This project instead *derives* an
interaction-reliability signal from unsupervised behavioural indicators and folds it into a
heterogeneous graph recommender, then measures whether that derivation helps. The derivation
used here is behavioural (shared text, repeated text, bursts); a graph anomaly detector (CoLA) was
the first choice and is kept as a measured negative control.

## 6. Proposed architecture

See **[`docs/architecture.md`](docs/architecture.md)** for all 12 diagrams. Summary:

```
Ingest → Clean → EDA/Outliers → Features ──┬─→ SBERT ─────┐
                                            └─→ Sentiment ─┼→ review features (390-d; the learned
                                                           ↓   fusion layer is an ablation arm)
                            Heterogeneous Graph Builder
                                   ↓
      Behavioural trust (3 indicators) × recency  →  per-interaction weights
                                   ↓
      HGT: edge-masked training, weighted BPR, validation early stopping
                                   ↓
                  Recommendation scoring (optional seed ensemble)
                    ↓              ↓             ↓
              Explanation    Counterfactual   Confidence
```

## 7. Dataset

**Amazon Reviews 2023** (McAuley Lab, UCSD) — an existing public dataset, downloaded at run
time, never modified, never redistributed here.

Three configured tiers, with sizes measured from the HuggingFace CDN rather than estimated:

| Tier | Category | Reviews file | Meta file | Role |
|---|---|---:|---:|---|
| `smoke` | All_Beauty | 0.30 GB | 0.20 GB | CI and wiring checks |
| `dev` | Video_Games | 2.50 GB | 0.41 GB | **default working tier** |
| `full` | Electronics | 21.06 GB | 4.89 GB | streamed only — exceeds free disk |

A second category runs through the `dev` rules with `DYTRUSTREC_DATASET=Musical_Instruments` (reviews 1.45 GB, metadata 0.59 GB; 116,690 interactions, 13,725 users, 7,942 items after the same five-core filter). It writes to its own `data/*_musical_instruments`, `reports/*_musical_instruments` and `checkpoints_musical_instruments` directories.

Electronics is the category the brief preferred, but 25.95 GB against 30 GB of free disk
cannot be materialised safely. It is therefore read through capped HTTP range requests, and
because a file prefix is not a random sample, any result from that tier carries an explicit
sampling-bias declaration in its manifest.

### Real schema, and what is missing from it

Verified by fetching actual records:

- **Reviews:** `rating`, `title`, `text`, `images`, `asin`, `parent_asin`, `user_id`,
  `timestamp` (ms), `helpful_vote`, `verified_purchase`
- **Metadata:** `main_category`, `title`, `average_rating`, `rating_number`, `features`,
  `description`, `price`, `images`, `videos`, `store`, `categories`, `details`,
  `parent_asin`, `bought_together`

Three gaps the brief's assumed schema did not anticipate:

| Expected | Reality | Consequence |
|---|---|---|
| `brand` | **absent** — closest is `store` | Brand nodes are labelled *Brand (proxy: store)* |
| `seller` | **absent entirely** | **No Seller node type and no `sold_by` edge** |
| fraud labels | **absent** | Trust is unsupervised; validated only synthetically |

Also: `categories` is empty for a large share of products (`main_category` is used instead),
and `price` is frequently null and occasionally a currency string or a range.

## 8. Data preprocessing

Implemented in `src/data/cleaner.py`. The governing rule is **remove the uninterpretable,
flag the merely suspicious**:

- **Removed:** ratings outside [1,5], timestamps outside the collection window, null user or
  item keys, exact ingestion duplicates, repeat reviews of the same item by the same user
  (most recent kept).
- **Flagged, never removed:** empty text, extreme length, text repeated across items by one
  user, review bursts under 60 s, zero rating variance. Each flagged row carries an
  `outlier_reasons` string explaining itself.

Suspicious rows are kept deliberately: the anomaly detector cannot find what the cleaner
already deleted, and removing them would flatter the trust component's later results.

Every run writes `reports/tables/cleaning_report.{json,md}` accounting for every row.

## 8b. What the data actually looks like (Phase 2, measured)

Produced by `scripts/run_eda.py` on the dev tier. These are descriptive statistics of the
corpus, not model results.

| Finding | Value | Consequence for the model |
|---|---|---:|
| Interactions / users / items | 144,178 / 17,223 / 7,692 | density 1.09 × 10⁻³ |
| Popularity Gini | **0.521**; top 20% of items hold 59.5% | a popularity baseline will be strong |
| 5-star share | **64.8%**, mean rating 4.29 | rating is a feature, not graded relevance |
| Verified purchases | 78.4%, rating **+0.503** vs unverified | supports `verified_purchase` as a trust input |
| **Per-user category drift** | **0.433** total-variation, first vs second half of history | the measured premise for temporal weighting |
| Connected components | **1** (100% of nodes) | cold start is genuinely new-user, not disconnection |
| Louvain modularity | **0.458** over 11 communities | real structure for a graph model to exploit |
| Degree assortativity | **−0.137** | active users attach to popular items |
| Anomaly indicators | 3,218 users (18.7%) corroborated by ≥2 independent layers | a review queue, **not** a fraud verdict |

The category-trend figure corroborates the drift number independently: the "legacy systems"
share falls from 0.88 to about 0.2, Nintendo Switch appears as a step in **2017** and
PlayStation 5 in **2020**, matching the real console launches. Temporal weighting is
modelling something that demonstrably exists here rather than something assumed.

Two statistical caveats are carried in the generated reports rather than buried:
IQR alone flags 54.9% of users, which is a multiple-comparisons artefact of testing 12
features and not a finding; and Isolation Forest and LOF agree on only 6.2% of their flags
(Jaccard), which is why a consensus count is used instead of any single detector.

## 9–18. Feature engineering, SBERT, graph, trust, HGT, ranking, explanations, counterfactuals, confidence

All implemented. Specified in [`docs/architecture.md`](docs/architecture.md), configured in
`configs/model_config.yaml` and `configs/training_config.yaml`. What the training procedure now does
(each item below was measured, see §20):

- **Edge-masked training.** The interaction being scored is also a message-passing edge. Each epoch a
  random share of interaction rows (all relations encoding the same row, forward and reverse, share one
  mask) is hidden from message passing and those rows are the supervision targets. Evaluation uses the
  full graph.
- **Validation early stopping**, for every model including the baselines, with the best weights restored.
- **Weighted objective.** Trust and recency scale each BPR term. They are *not* edge weights:
  `HGTConv` cannot read an edge weight.
- **Trust from behaviour, not from CoLA.** Three label-free indicators (shared text, repeated text,
  24-hour burstiness), rank-averaged. CoLA is kept as a documented negative control.
- **Seed ensembles** average per-user z-scored scores of independently trained runs.
- **Refit protocol.** After tuning, models are refit on train + validation for the epoch count their
  validation run selected, and scored on the same test split.

## 19. Evaluation

Full-catalogue ranking, training positives masked, users with no held-out item excluded and counted.
Precision/Recall/NDCG/MAP/MRR/Hit Rate at K ∈ {5, 10, 20}; coverage, diversity, novelty. Every model's
hyper-parameters are searched on the **validation** split (`scripts/tune_models.py`; test is never
scored there), every model is stopped by validation NDCG@10, and test is scored once per run. Baselines:
five seeds; HGT arms: three (two for the CoLA control). Differences are tested with a paired bootstrap over
users on per-user NDCG@10 averaged across seeds.

## 20. Results

Dev tier (Video_Games), 1,862 evaluable test users.
`scripts/run_all.sh` reproduces everything; tables are regenerated into `reports/tables/`.

### Model comparison — tuning protocol (train only; stopped and selected on validation)

| Model | seeds | NDCG@10 | Recall@10 | MAP@10 | Hit@10 | Coverage@10 | Novelty@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **E — B + trust + recency** (3-seed ensemble) | 3 | 0.0334 ± 0.0000 | 0.0554 ± 0.0000 | 0.0225 | 0.0822 | 0.036 | 9.85 |
| G — F with CoLA trust | 2 | 0.0323 ± 0.0036 | 0.0553 ± 0.0095 | 0.0214 | 0.0795 | 0.036 | 9.88 |
| **E — B + trust + recency** | 3 | 0.0308 ± 0.0013 | 0.0512 ± 0.0034 | 0.0205 | 0.0772 | 0.041 | 10.05 |
| I — E + cosine decay (3-seed ensemble) | 3 | 0.0304 ± 0.0000 | 0.0504 ± 0.0000 | 0.0201 | 0.0752 | 0.040 | 10.01 |
| I — E + cosine decay | 3 | 0.0297 ± 0.0036 | 0.0480 ± 0.0067 | 0.0200 | 0.0736 | 0.044 | 10.13 |
| D — B + recency (3-seed ensemble) | 3 | 0.0297 ± 0.0000 | 0.0491 ± 0.0000 | 0.0201 | 0.0725 | 0.041 | 10.03 |
| C — B + trust (3-seed ensemble) | 3 | 0.0296 ± 0.0000 | 0.0440 ± 0.0000 | 0.0216 | 0.0644 | 0.054 | 9.16 |
| D — B + recency | 3 | 0.0295 ± 0.0017 | 0.0488 ± 0.0024 | 0.0202 | 0.0718 | 0.047 | 10.22 |
| B — + SBERT (3-seed ensemble) | 3 | 0.0295 ± 0.0000 | 0.0457 ± 0.0000 | 0.0210 | 0.0671 | 0.059 | 9.27 |
| F — E + learned fusion (3-seed ensemble) | 3 | 0.0295 ± 0.0000 | 0.0468 ± 0.0000 | 0.0205 | 0.0704 | 0.029 | 9.93 |
| F — E + learned fusion | 3 | 0.0287 ± 0.0011 | 0.0462 ± 0.0023 | 0.0198 | 0.0698 | 0.035 | 10.12 |
| B — + SBERT | 3 | 0.0278 ± 0.0018 | 0.0425 ± 0.0016 | 0.0200 | 0.0623 | 0.065 | 9.35 |
| C — B + trust | 3 | 0.0270 ± 0.0004 | 0.0422 ± 0.0006 | 0.0191 | 0.0616 | 0.060 | 9.25 |
| A — HGT only (3-seed ensemble) | 3 | 0.0267 ± 0.0000 | 0.0446 ± 0.0000 | 0.0178 | 0.0666 | 0.020 | 8.87 |
| A — HGT only | 3 | 0.0261 ± 0.0014 | 0.0424 ± 0.0011 | 0.0177 | 0.0636 | 0.034 | 9.05 |
| H — B, no edge masking (3-seed ensemble) | 3 | 0.0228 ± 0.0000 | 0.0398 ± 0.0000 | 0.0149 | 0.0580 | 0.015 | 8.52 |
| SASRec | 5 | 0.0203 ± 0.0020 | 0.0317 ± 0.0023 | 0.0142 | 0.0490 | 0.121 | 9.80 |
| BPR-MF | 5 | 0.0189 ± 0.0007 | 0.0306 ± 0.0011 | 0.0129 | 0.0467 | 0.071 | 9.30 |
| H — B, no edge masking | 3 | 0.0189 ± 0.0047 | 0.0326 ± 0.0053 | 0.0121 | 0.0501 | 0.017 | 8.87 |
| LightGCN | 5 | 0.0180 ± 0.0007 | 0.0309 ± 0.0014 | 0.0119 | 0.0463 | 0.102 | 9.44 |
| Popularity | 1 | 0.0121 ± 0.0000 | 0.0257 ± 0.0000 | 0.0066 | 0.0392 | 0.002 | 8.21 |
| Item-kNN | 1 | 0.0071 ± 0.0000 | 0.0132 ± 0.0000 | 0.0041 | 0.0220 | 0.770 | 13.19 |

Best baseline: SASRec at 0.0203. Best single HGT arm: **E — B + trust + recency** at 0.0308
(+52%); best ensemble 0.0334 (+64%).
Paired bootstrap over users, **E — B + trust + recency** (3-seed ensemble) against SASRec: mean difference +0.0131, 95% CI [+0.0093, +0.0171].

### Model comparison — refit protocol (train + validation, same test split)

Scores are **not comparable with the table above**: refitting makes more test users evaluable
(3,221 against 1,862). Compare models within a table.

| Model | seeds | NDCG@10 | Recall@10 | MAP@10 | Hit@10 | Coverage@10 | Novelty@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| I — E + cosine decay (3-seed ensemble) | 3 | 0.0329 ± 0.0000 | 0.0502 ± 0.0000 | 0.0214 | 0.0925 | 0.070 | 10.23 |
| **E — B + trust + recency** (3-seed ensemble) | 3 | 0.0329 ± 0.0000 | 0.0534 ± 0.0000 | 0.0204 | 0.0984 | 0.041 | 10.62 |
| D — B + recency (3-seed ensemble) | 3 | 0.0325 ± 0.0000 | 0.0520 ± 0.0000 | 0.0202 | 0.0981 | 0.054 | 10.65 |
| F — E + learned fusion (3-seed ensemble) | 3 | 0.0312 ± 0.0000 | 0.0505 ± 0.0000 | 0.0190 | 0.0947 | 0.037 | 10.42 |
| I — E + cosine decay | 3 | 0.0311 ± 0.0000 | 0.0494 ± 0.0023 | 0.0197 | 0.0901 | 0.078 | 10.38 |
| D — B + recency | 3 | 0.0303 ± 0.0020 | 0.0497 ± 0.0031 | 0.0185 | 0.0924 | 0.061 | 10.87 |
| **E — B + trust + recency** | 3 | 0.0298 ± 0.0035 | 0.0485 ± 0.0069 | 0.0184 | 0.0906 | 0.043 | 10.82 |
| F — E + learned fusion | 3 | 0.0263 ± 0.0025 | 0.0438 ± 0.0034 | 0.0156 | 0.0847 | 0.042 | 10.60 |
| B — + SBERT (3-seed ensemble) | 3 | 0.0221 ± 0.0000 | 0.0330 ± 0.0000 | 0.0142 | 0.0646 | 0.061 | 9.26 |
| B — + SBERT | 3 | 0.0210 ± 0.0023 | 0.0314 ± 0.0035 | 0.0136 | 0.0603 | 0.069 | 9.34 |
| LightGCN | 5 | 0.0169 ± 0.0005 | 0.0277 ± 0.0008 | 0.0103 | 0.0524 | 0.096 | 9.46 |
| SASRec | 5 | 0.0163 ± 0.0023 | 0.0260 ± 0.0035 | 0.0101 | 0.0505 | 0.130 | 9.90 |
| BPR-MF | 5 | 0.0161 ± 0.0007 | 0.0247 ± 0.0010 | 0.0102 | 0.0473 | 0.068 | 9.33 |
| Popularity | 1 | 0.0130 ± 0.0000 | 0.0219 ± 0.0000 | 0.0080 | 0.0397 | 0.002 | 8.27 |
| Item-kNN | 1 | 0.0079 ± 0.0000 | 0.0131 ± 0.0000 | 0.0044 | 0.0279 | 0.845 | 13.29 |

Best baseline LightGCN at 0.0169; best single arm I — E + cosine decay at 0.0311
(+84%); best ensemble 0.0329 (+95%).

### What each component does (tuning protocol, test NDCG@10)

| Step | Change | Before | After | Δ |
|---|---|---:|---:|---:|
| A→B | add SBERT + structured features | 0.0261 | 0.0278 | +7% |
| B→H | **remove edge masking** | 0.0278 | 0.0189 | -32% |
| B→C | add behavioural trust weighting | 0.0278 | 0.0270 | -3% |
| B→D | add recency weighting | 0.0278 | 0.0295 | +6% |
| D→E | then add trust | 0.0295 | 0.0308 | +4% |
| E→F | then add the learned fusion layer | 0.0308 | 0.0287 | -7% |
| F→G | swap behavioural trust for CoLA trust (2 seeds) | 0.0287 | 0.0323 | +12% |
| E→I | add cosine learning-rate decay | 0.0308 | 0.0297 | -3% |

Seed standard deviations are 0.0004–0.0047, so changes under about 5% are not distinguishable from noise.

### What this shows

1. **Edge masking is the largest effect.** Without it an HGT recommender peaks at epoch 5 and equals
   matrix factorisation; the same architecture with it is +47%.
   The first version's "graph arms triple catalogue coverage" was a symptom of memorisation, not a design strength.
2. **Recency helps a lot — but validation overstates it.** On validation, recency weighting is worth +43%; on
   the test split of the tuning protocol only +6%, because the validation
   period immediately follows training while the test period lies one period further out. Under the refit
   protocol it is worth +44%.
3. **Trust does not change accuracy on clean data** (within noise either way), and CoLA trust is not worse than behavioural trust there (arm G vs F). Its intended job is robustness; the shill study below finds that no defence recovers more than seed noise.
4. **The learned fusion layer does not beat concatenation** in either protocol.
5. **SBERT helps modestly** (+7%), far less than the first version's +23.6%.
6. **A cosine learning-rate decay stabilises refit training** (seed spread of arm I under refit is the smallest
   of any HGT arm) but does not help under validation-based stopping.

### Trust validation (Module 45) — synthetic shills

> **SYNTHETIC RESULT.** The positives were generated by `scripts/run_synthetic_anomaly.py`, not observed in
> Amazon data. They measure whether a detector recognises *this generative process*, not real fraud.

| Detector | ROC AUC | Avg precision | Precision@100 |
|---|---:|---:|---:|
| Behavioural trust detector (Module 11) | 0.9997 | 0.9882 | 0.9800 |
| trivial rule: frac_duplicate_text | 0.9895 | 0.7075 | 0.5600 |
| consensus indicator count | 0.9478 | 0.3272 | 0.3300 |
| trivial rule: -min_gap_hours | 0.9165 | 0.3109 | 0.8000 |
| Isolation Forest (Module 4) | 0.9129 | 0.2026 | 0.0000 |
| CoLA trust (1 - trust) | 0.2951 | 0.0318 | 0.0200 |

CoLA is the only detector below chance: a coordinated campaign is internally consistent by construction, so each
shill is easy to predict from the others and is rated *more* trustworthy. The behavioural detector asks instead
whether the account behaves like a campaign account. The benchmark is easy by design (shared templates, bursts).

### HGT under a shill campaign (`scripts/run_robustness.py`, SYNTHETIC)

| Shill rate | Defence | seeds | NDCG@10 (clean test) | Users shown a promoted item |
|---:|---|---:|---:|---:|
| 0% | none | 3 | 0.0278 ± 0.0018 | 0.00% |
| 0% | trust | 2 | 0.0267 ± 0.0009 | 0.00% |
| 0% | trust+gate | 2 | 0.0244 ± 0.0003 | 0.00% |
| 20% | none | 3 | 0.0250 ± 0.0020 | 0.20% |
| 20% | cola | 4 | 0.0253 ± 0.0022 | 0.59% |
| 20% | trust | 3 | 0.0245 ± 0.0012 | 0.32% |
| 20% | trust+gate | 3 | 0.0259 ± 0.0014 | 1.20% |
| 20% | oracle | 3 | 0.0261 ± 0.0014 | 0.59% |
| 20% | inverted | 3 | 0.0223 ± 0.0033 | 0.18% |

### Confidence and explanations (`scripts/evaluate_confidence.py`)

AUROC for predicting a hit among 18,620 (user, top-10 item) pairs: confidence 0.456, stability 0.470, evidence volume 0.458, evidence trust 0.507, raw score 0.556. **The confidence score does not predict correctness.** What it tracks is evidence volume: mean 0.292 for users with fewer than 3 training interactions against 0.648 for the rest. 83.6% of recommendations had at least one retrieved evidence item; the hit rate with evidence (0.0081) is not higher than without (0.0118).

### New products (`scripts/evaluate_cold_start.py`)

64% of the 8,317 test interactions of known users are with 1,073 products that did not exist in
the training split, and a model with no embedding for them cannot recommend them. Mapping product text into the
embedding space (CV R² 0.32) ranks new products worse than plain Sentence-BERT text similarity
(NDCG@10 0.0099 vs 0.0247, random 0.0040), and a single blended score
did not help. What works is a separate "new for you" strip: reserving the last r = 3 of the ten places (chosen on validation)
for the best new products by text lifts test NDCG@10 from 0.0126 to 0.0154
(+0.0028, 95% CI [+0.0017, +0.0040]) and Recall@10 from 0.0182 to 0.0262.

### A campaign that knows the detector (`scripts/run_adaptive_attack.py`, SYNTHETIC)

The adaptive campaign paraphrases every review and spreads them over 30 days. Detector v1 falls to
AUC 0.519 (chance); the one-day burst indicator points the wrong way (0.129). Comparing sentence embeddings with other accounts'
reviews holds at 0.994 (standard campaign 0.994). Averaging all indicators dilutes it (0.727);
taking the strongest indicator keeps it (0.944).

HGT trained on the adaptive campaign (`scripts/run_robustness.py --adaptive`):

| Shill rate | Defence | seeds | NDCG@10 (clean test) |
|---:|---|---:|---:|
| 0% | none | 3 | 0.0278 ± 0.0018 |
| 0% | gate_v2 | 2 | 0.0266 ± 0.0004 |
| 20% | none | 4 | 0.0246 ± 0.0017 |
| 20% | trust+gate | 4 | 0.0235 ± 0.0033 |
| 20% | gate_v2 | 4 | 0.0266 ± 0.0018 |
| 20% | oracle | 2 | 0.0236 ± 0.0029 |
| 20% | oracle_gate | 4 | 0.0269 ± 0.0014 |

### Are the explanations faithful? (`scripts/evaluate_explanations.py`)

The user's own interactions an explanation cites are removed from the graph, the trained model is re-run, and the drop in that recommendation's score is compared with removing as many random interactions.

| Cited evidence | users | score drop, cited | score drop, random | cited hurts more | difference (95% CI) |
|---|---:|---:|---:|---:|---|
| shared_category | 81 | 0.314 | -0.127 | 94% | +0.441 [+0.322, +0.604] |
| shared_brand | 73 | 0.475 | -0.239 | 95% | +0.714 [+0.582, +0.853] |

### Serving and training cost (`scripts/benchmark_serving.py`)

Serving is a dot product and a top-k: 0.08 ms per user here, 1.6 ms at 100,000 products and 16 ms at a million
(synthetic vectors), so brute force suffices below roughly a million products. Training is the part that does not scale: the HGT trains full-graph
(about 211 s per run against 17 s for matrix factorisation) and production use would need neighbour-sampled mini-batches.

A calibrated hit probability (logistic regression on score, rank, stability, history length and trust, fitted on validation recommendations)
has test AUROC 0.607 against 0.456 for the evidence blend, and the hit rate rises from 0.0059 in its lowest quintile to
0.0161 in its highest. It ranks well but is not calibrated in level: the validation period's hit rate (0.0161) is nearly double the test period's
(0.0087), so probabilities are about twice too high. A deployed system would refit the intercept on the most recent period.


### A second dataset (`scripts/run_second_dataset.sh`, Musical_Instruments)

The whole pipeline was repeated on a second Amazon Reviews 2023 category with the same tier rules,
five-core filter and chronological split. **Every model family, baselines and HGT alike, was tuned on this
dataset's own validation split** with the same grids; the test split was scored once. The HGT was first run
with the Video_Games settings, un-retuned (arm B 0.0187, arm E 0.0197); those runs are kept in
`reports/results_musical_instruments/runs_transfer/`. The tuned figures are below.

| Model (tuning protocol) | Video_Games | Musical_Instruments |
|---|---:|---:|
| Popularity | 0.0121 ± 0.0000 | 0.0187 ± 0.0000 |
| BPR-MF | 0.0189 ± 0.0007 | 0.0182 ± 0.0017 |
| LightGCN | 0.0180 ± 0.0007 | 0.0196 ± 0.0004 |
| SASRec | 0.0203 ± 0.0020 | 0.0154 ± 0.0013 |
| H: B, no edge masking | 0.0189 ± 0.0047 | 0.0164 ± 0.0014 |
| A: HGT only | 0.0261 ± 0.0014 | 0.0184 ± 0.0005 |
| B: + SBERT | 0.0278 ± 0.0018 | 0.0199 ± 0.0023 |
| C: B + trust | 0.0270 ± 0.0004 | 0.0188 ± 0.0010 |
| D: B + recency | 0.0295 ± 0.0017 | 0.0200 ± 0.0011 |
| E: B + trust + recency | 0.0308 ± 0.0013 | 0.0208 ± 0.0008 |
| F: DyTrustRec (full) | 0.0287 ± 0.0011 | 0.0214 ± 0.0005 |
| E: B + trust + recency, 3-seed ensemble | 0.0334 ± 0.0000 | 0.0234 ± 0.0000 |

| Finding | Video_Games | Musical_Instruments | Verdict |
|---|---|---|---|
| Edge masking helps (B vs H) | +47.3% | +21.8% | replicates |
| Review text (SBERT) helps (A to B) | +6.7% (within noise) | +8.6% (within noise) | no reliable effect on either |
| Trust weighting (B to C) | −2.9% (within noise) | −5.7% (within noise) | no reliable effect on either |
| Recency weighting (B to D) | +6.2% (within noise) | +0.5% (within noise) | no reliable effect on either |
| Learned fusion layer (E to F) | −6.8% | +2.8% (within noise) | no gain on either |
| Best single HGT vs best tuned baseline | +52% (interval excludes 0) | +6% (interval includes 0) | direction only |
| Three-seed ensemble vs best tuned baseline | +64% (interval excludes 0) | +19% (interval excludes 0) | replicates |
| Embedding indicator vs paraphrased campaign (AUC, old detector to new) | 0.519 to 0.994 | 0.489 to 0.997 | replicates |
| Semantic gate protects accuracy under a 20% paraphrased campaign | +8.2% (z = 1.7); the attack alone costs −11.4% | −1.8% (z = −0.6); the attack alone costs −3.7% | suggestive on the first, none on the second |
| Reserved new-product slots lift NDCG@10 | +22% with r = 3; 64% of test rows are new products | none (validation chose r = 0); 23% of test rows are new products | does not replicate |
| Raw evidence-sufficiency confidence predicts hits (AUROC) | 0.456 | 0.472 | replicates: no signal |
| Separately calibrated hit probability ranks hits (AUROC) | 0.607 | 0.579 | replicates, weakly |
| Cited category evidence matters more than random evidence (difference, 95% CI) | +0.441 [0.322, 0.604] | +0.273 [0.206, 0.351] | replicates |

Plain reading: **the training recipe and the checks on the model repeat; the size of the lead does not.** The best
single model is level with LightGCN within noise (the interval for the difference includes zero), and only the
three-seed ensemble is clearly ahead. Popularity is within 5% of the best baseline on this catalogue. The paraphrased
campaign is found again (the semantic gate removed exactly the injected rows plus the same genuine rows it flags on
clean data), but it barely lowers the model's accuracy here, so there was nothing for the gate to recover. The
new-product strip was switched off by its own validation rule because only 23% of test rows are on new products.
Tables: `reports/tables_musical_instruments/`. This is one more category with one split, not a general result.

### What was wrong before

The first grid (single seed, untuned baselines) reported that no HGT arm beat matrix factorisation and that trust cost 17–20%. It could not be believed: (1) trust and recency were stored as edge weights that `HGTConv` cannot read, so arms C, D and E were identical to B and their differences were seed noise; (2) early stopping was implemented but never called; (3) the baselines had no hyper-parameter search (LightGCN scored below popularity); (4) the supervision edge was visible to message passing; (5) the ablation's "CoLA inversion explains the −17% trust cost" story compared models that were the same. All five are fixed and the earlier numbers are withdrawn (archived in `reports/results/archive_v1_unmasked/`).

### Caveats that limit every number above

1. Two datasets, one split each (the second repeats some findings and not others, see above); only warm users are scored (about 60% of test rows involve an unseen user or item).
2. Three seeds per HGT arm; hyper-parameters searched one-at-a-time on a single seed (selection noise about 0.002 NDCG@10).
3. Before the protocol was fixed, about ten HGT variants were explored while watching test curves; that is where the edge-masking idea came from. The reported numbers come from a fresh pipeline that selects on validation only, but what to try was not chosen blind.
4. Trust is validated only on synthetic campaigns scripted by the same author as the defence (a paraphrased, slowly spread campaign defeats the exact-text detector and is caught by the embedding detector); no real manipulation data or published attack benchmark was used.
5. No sequence-model or LLM baselines, and the protocol differs from the leave-one-out protocol of most published results on this dataset, so no claim against published numbers is made.

## 21. Limitations

1. **No fraud ground truth.** Trust is validated only by synthetic injection, labelled synthetic everywhere.
2. **No temporal GNN.** Despite "Dynamic" in the name this is exponential recency weighting of the objective,
   not TGN, TGAT or temporal attention.
3. **k-core selection leakage.** Counts over the whole timeline include the test period; standard practice, disclosed.
4. **Brand is a proxy** derived from `store`; there is **no seller** entity.
5. **Coverage and novelty are catalogue-relative** (post-k-core).
6. **Reproducible up to backend non-determinism.** MPS offers no deterministic-algorithm guarantee.
7. **Explanations and counterfactuals were not evaluated with people**, and the counterfactual perturbs the user
   vector in embedding space, not the graph.
8. **Category granularity is a modelling choice** (`graph.category_level`).
9. **Electronics-tier sampling is biased** (a streamed file prefix is not a random sample).

## 22. Future work

Attacks that paraphrase text or spread timing; a content path for cold users and items; learned temporal
encodings in the attention; a fidelity test for explanations (remove the cited interactions and re-score); a
a third category and a domain outside Amazon (the lead over strong baselines shrank from +52% to +6% on the second category); comparison with published defences on their own attacks.

---

## Novelty statement

> This project does not claim novelty in HGT, SBERT, CoLA, BPR or edge masking. Related work on
> shilling-robust graph recommenders exists (Anti-FakeU, WWW 2023; Trust-GRS, AAAI 2025; FAHGT, IEEE Access
> 2021). Its contribution is an audit-and-repair of a trust-aware HGT recommender and what it measured: the
> size of the effect of edge masking, the evidence that validation overstates recency weighting on a later test
> period, the failure of CoLA as a trust source on coordinated campaigns and a behavioural detector that does
> not fail, and the finding that the confidence score tracks evidence volume and not correctness.

No claim of state-of-the-art performance is made. The tuned baselines are the comparison that was run.

---

## Setup

```bash
brew install uv
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
uv pip install -r requirements-pyg.txt -f https://data.pyg.org/whl/torch-2.13.0+cpu.html
```

`pyg-lib` comes from PyG's own wheel index, not PyPI. It supplies the neighbour-sampling
kernels; without it `NeighborLoader` raises `ImportError` and mini-batch training on the
heterogeneous graph is impossible. There are no arm64 macOS wheels for `torch-scatter` or
`torch-sparse` at this torch version, so the project is written not to need them.

## Usage

```bash
python scripts/prepare_data.py --tier smoke
```

```bash
python scripts/prepare_data.py --tier dev
```

Any config key can be overridden without editing YAML:

```bash
python scripts/prepare_data.py --tier dev --set sampling.strategy=reservoir --set tiers.dev.limits.max_reviews=50000
```

```bash
python scripts/run_eda.py
```

```bash
python -m pytest
```

### Reproducing the results

```bash
sh scripts/run_all.sh            # tune (validation only), run the grid, the refit protocol and the studies
sh scripts/run_all.sh results    # skip tuning; reuse reports/results/tuning/
```

Individual stages:

```bash
python scripts/tune_models.py --model hgt --jobs base,keep_0.5,combined,rec_2.0   # validation only
python scripts/run_experiments.py --only baselines
python scripts/run_experiments.py --only hgt --arms B,E --seeds 42,43,44
python scripts/run_experiments.py --protocol refit --only hgt --arms E --seeds 42,43,44
python scripts/run_robustness.py --rates 0.2 --seeds 42,43     # synthetic shills, HGT
python scripts/evaluate_confidence.py --arm E_hgt_sbert_trust_recency
python scripts/export_model.py && python scripts/explain_example.py
DYTRUSTREC_DATASET=Musical_Instruments sh scripts/run_second_dataset.sh   # the second dataset, own directories
node docs/build/make_paper.js && node docs/build/build_explained.js   # the paper and the explainer
```

Run HGT trainings one at a time: two on the same GPU stall each other, and CPU-heavy work alongside
one slows it badly.

## Project layout

```
configs/    YAML for data, analytics, model, training, experiments — no hyperparameter in code
data/       raw (immutable) → interim (sampled) → processed (cleaned) → embeddings (cached)
src/        the actual system: data, analytics, features, graph, anomaly, models,
            training, recommendation, explainability, evaluation, utils
scripts/    CLI entrypoints, one per pipeline stage
tests/      pytest suite
notebooks/  exploration and presentation only; no logic lives here
reports/    generated figures, tables and results
docs/       architecture.md, plan.md, viva.md
```

## Citation

```
Hou et al. (2024). Bridging Language and Items for Retrieval and Recommendation. arXiv:2403.03952.
Hu et al. (2020). Heterogeneous Graph Transformer. WWW '20.
Liu et al. (2021). Anomaly Detection on Attributed Networks via CoLA. IEEE TNNLS.
Reimers & Gurevych (2019). Sentence-BERT. EMNLP '19.
Rendle et al. (2009). BPR: Bayesian Personalized Ranking. UAI '09.
He et al. (2020). LightGCN. SIGIR '20.
```
