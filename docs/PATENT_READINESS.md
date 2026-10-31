# Patent-readiness notes for DyTrustRec

> **This is an engineering inventory, not legal advice.** Whether anything here is patentable is
> a question for a patent attorney after a prior-art search. What follows says which parts could
> plausibly be framed as inventive, what measured evidence each has, and what is missing.
> Nothing in it should be read as a claim that a patent would be granted.

## 1. The technical problem

A recommender that learns from product reviews has four practical weaknesses:

1. a graph model trained on its own interaction edges can read the answer off the graph, so it
   scores no better than matrix factorisation;
2. coordinated review campaigns distort it, and a defence that looks for repeated text is evaded
   by an attacker who rewords every review;
3. it cannot recommend products that did not exist when it was trained (64% of test
   interactions of known users here);
4. its explanations are true of the data but nobody checks whether the model used them.

## 2. Candidate concepts, with what was measured

Ratings below are the author's reading of novelty risk, not a legal opinion.

| # | Concept | Evidence in this repository | Novelty risk |
|---|---|---|---|
| C1 | Edge-masked supervised training of a heterogeneous graph recommender: interaction edges hidden from message passing each iteration and used as the targets | Same architecture, NDCG@10 0.0189 without it and 0.0278 with it (+47%, 3 seeds). Repeats on a second category: 0.0164 to 0.0199 (+22%, 3 seeds). | **High.** It is established link-prediction practice (separate message-passing and supervision edges; DropEdge). The size of the effect for this model is a result, not an invention. |
| C2 | Trust x recency weights applied to the training objective instead of the graph, because the attention layer cannot read an edge weight | Recency: +6% on a split with a horizon gap, +44% when refit on train + validation; +0.5% (within noise) on the second category under the tuning protocol. Trust: neutral on clean data on both categories. | Medium to high. Time weighting in collaborative filtering is old. |
| C3 | **A gate that removes accounts whose reviews read like other accounts' reviews (sentence-embedding comparison) from both the message-passing graph and the training objective** | Against a paraphrased, spread-out synthetic campaign: exact-text matching ROC-AUC 0.52, embedding comparison 0.994. HGT NDCG@10 at 20% shills: 0.0246 undefended, 0.0235 exact-text gate (it removed 262 rows, none of the campaign), **0.0266 semantic gate (removed about 52,000 rows, essentially the whole campaign)**, 0.0269 for a gate using the true labels. Clean-data cost 0.0266 vs 0.0278. **Second category (Musical_Instruments):** detection repeats (ROC-AUC against the paraphrased campaign 0.489 for the exact-text detector, 0.997 for the embedding indicator; the semantic gate removed exactly the injected rows plus the same 2,849 genuine rows it flags on clean data). **The accuracy benefit does not repeat:** the same campaign costs the undefended HGT only 3.7% there (0.0192 vs 0.0199 clean, inside the seed spread), and the gate scores 0.0189 (z = -0.6, 5 seeds), so there was no accuracy to recover. | **Medium to high, and weaker than before.** Detection by meaning is the part that has held on two categories and where the exact-text alternative demonstrably fails. The claim that the gate protects ranking accuracy rests on one dataset at about 1.7 standard errors. Shilling-robust graph recommenders exist (Anti-FakeU, WWW 2023; Trust-GRS, AAAI 2025; FAHGT, IEEE Access 2021), so the claim would have to be narrow and specific. |
| C4 | Verifying an explanation by removing the interactions it cites and re-running the model | Removing cited category evidence drops the recommendation's score by 0.314 spread units against -0.127 for as many random interactions (difference +0.441, 95% CI [+0.322, +0.604], 81 users); brand evidence +0.714 [+0.582, +0.853], 73 users. **Repeats on the second category:** category +0.273 [+0.206, +0.351] (300 users), brand +0.145 [+0.109, +0.183] (55 users). | Medium. Counterfactual and removal-based faithfulness tests exist in the explainable-AI literature; applying one to a trust-aware graph recommender with retrieved-evidence explanations is the narrow part. |
| C5 | A separate strip of new products ranked by text similarity, with the number of slots chosen on validation | NDCG@10 0.0126 -> 0.0154, paired-bootstrap difference +0.0028, 95% CI [+0.0017, +0.0040]. A single blended score and a text-to-embedding mapping both did worse. **Does not repeat:** on the second category validation chose r = 0 (23% of test rows are new products against 64%), and any reserved slot lowered NDCG@10. | High. Reserved or exploration slots are common in practice. |
| C6 | Reporting an evidence-sufficiency score and a separately calibrated hit probability | Evidence score AUROC 0.456 for predicting hits; calibrated probability AUROC 0.607, hit rate 0.0059 -> 0.0161 from lowest to highest quintile; level over-predicted about 2x because the validation period's hit rate is double the test period's. Second category: evidence score AUROC 0.472; calibrated probability AUROC 0.579, quintile hit rate 0.0055 to 0.0120, level close to observed. | High. Calibration is standard. |

If anything is to be pursued, the narrow candidate is detection by comparing sentence embeddings across
accounts (the detection half of C3), possibly combined with the removal test of C4. Both repeat on two
categories and the obvious alternative (exact-text matching) fails against the paraphrased campaign. The
protective half of C3 (accuracy recovered under attack) has **not** repeated and should not be relied on.

## 3. Where the model stands against alternatives (same split, same evaluator, all tuned on validation)

| Model | Video_Games, tuning protocol | Video_Games, refit protocol | Musical_Instruments, tuning protocol |
|---|---:|---:|---:|
| Popularity | 0.0121 | 0.0130 | 0.0187 |
| SASRec | 0.0203 | 0.0163 | 0.0154 |
| BPR-MF | 0.0189 | 0.0161 | 0.0182 |
| LightGCN | 0.0180 | 0.0169 | 0.0196 |
| **DyTrustRec, best single arm** | **0.0308** | **0.0311** | **0.0208** |
| DyTrustRec, 3-seed ensemble | 0.0334 | 0.0329 | 0.0234 |

On Musical_Instruments the single model is +6% over LightGCN with a paired-bootstrap interval that includes zero
([-0.0020, +0.0045]); only the ensemble is clearly ahead (+19%, [+0.0004, +0.0072]). On Video_Games the
corresponding leads were +52% and +64%. The HGT settings were tuned on each dataset's validation split; the
un-retuned Video_Games settings scored 0.0197 for arm E on the second category, level with LightGCN.

Not compared: LLM-based recommenders, industrial two-tower and multi-task ranking systems, and
published numbers on the standard leave-one-out protocol (the protocols differ, so no
above-or-below claim is made).

## 4. What is missing before this is a strong filing

1. **A third category and a non-Amazon domain.** A second Amazon category (Musical_Instruments) has now been run and repeats some
   findings and not others (see C3, C5 and the table above). Two categories from one release is still a small sample.
2. **Real manipulation data or a published attack benchmark.** All defence evidence is on scripted
   campaigns written by the same author as the defence.
3. **Comparison with published defences** (Anti-FakeU, Trust-GRS) on their own attacks. Without
   it, C3's advantage is only shown against the author's own earlier detector.
4. **A setting where the attack actually hurts**, then more seeds. The gate's accuracy gain is suggestive on one dataset (about 1.7 standard errors, 4 seeds)
   and absent on the other (5 seeds), where the campaign barely lowered accuracy. Raising the injection rate until it hurts would be a new
   experiment chosen after seeing this result, and should be reported as such.
5. **An online test.** Offline ranking gains are a proxy.
6. **A prior-art search** by someone qualified (Google Patents, USPTO, Espacenet, WIPO), covering at
   least: edge masking in graph recommenders, review-spam gating in graph neural networks,
   embedding-similarity fake-review detection, explanation faithfulness by removal.

## 5. Process cautions

- Publishing can damage novelty, and rules differ by country (some have no grace period). The
  paper in `docs/` and the README describe the methods in detail; talk to an attorney before
  sharing either outside the course.
- Keep dated lab notes of who did what; the repository history and `reports/results/` are good
  raw material for that.
- The tuned baselines, the withdrawn first grid (archived in `reports/results/archive_v1_unmasked/`)
  and the disclosed exploration phase all matter for honesty and should be disclosed to counsel.
