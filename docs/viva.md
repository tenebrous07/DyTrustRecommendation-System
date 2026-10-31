# DyTrustRec — Viva Preparation

Answers are written to be defensible, not impressive. Where something is weak, it says so —
an examiner finds those spots anyway, and conceding them first is cheaper than being caught.

---

## Fundamentals

**What is a recommender system?**
A system that ranks items for a user by predicted relevance. Here the feedback is *implicit*:
a review means the user engaged with the product, not that they graded their preference on a
scale we can trust.

**Why not plain collaborative filtering?**
CF only sees the user-item matrix. It cannot use review text, product metadata, or the
reliability and age of an interaction — the four things this project is about. It also fails
completely on cold items. We still implement item-kNN CF as a baseline, because a method
being limited is not the same as it being weak.

**Why graph-based recommendation?**
The data is natively a graph: users, products, reviews, brands and categories with typed
relations. Flattening it into a matrix discards the relations. Phase 2 measured Louvain
modularity of **0.458** over 11 communities — real structure a graph model can exploit that
matrix factorisation cannot see.

**What is a heterogeneous graph?**
A graph with multiple node types and multiple edge types. Ours has five node types
(user, product, review, brand, category) and five relations, each also added in reverse.

**Why reverse edges?**
Without them HGT can push information from users into products but never back, so user
embeddings would never see the items they interacted with. It is a correctness requirement,
not a tuning choice.

---

## The components

**Is HGT novel?**
No. Hu et al., WWW 2020. We use it; we do not claim it.

**Why HGT rather than a plain GNN?**
It learns type-specific projections and relation-aware attention, so a `user→purchased→product`
edge and a `product→belongs_to→category` edge are not forced through the same transformation.
On a graph with five node types that distinction is the whole point.

**What is SBERT and why not TF-IDF?**
SBERT (Reimers & Gurevych, 2019) produces a 384-dimensional sentence embedding where cosine
distance tracks meaning. TF-IDF is sparse and lexical: "the controller broke" and "this
gamepad stopped working" share almost no terms but mean the same thing. TF-IDF would also
produce a vector far too wide and sparse to use as a graph node feature.

**Verified numbers:** dimension 384, `max_seq_length` 256, unit-normalised, so a dot product
*is* the cosine.

**What is CoLA? Does it detect fake reviews?**
CoLA (Liu et al., TNNLS 2021) is contrastive self-supervised **graph anomaly detection**. It
trains a discriminator to tell "this local subgraph belongs to this node" from "it belongs to
someone else"; a node its own neighbourhood fails to predict is anomalous.

**It does not detect fake reviews.** It emits an unbounded anomaly score. Calling that "fraud"
would be an unsupported leap — and this dataset has **no fraud labels at all**, so we cannot
even check.

**Then how is trust calculated?**
Not from CoLA any more. The model's trust comes from a behavioural detector with three
label-free indicators, each rank-normalised and averaged with equal weight: the share of a
user's reviews whose text other accounts also posted, the share that repeat the user's own
text, and the most reviews posted in any 24-hour window. Then `trust = 1 − rank_normalised(score)`,
clamped to `[0.05, 1.0]`.

Three defensible choices:
- **Rank** normalisation, because it is invariant to the score distribution's shape. It is a
  config option and is ablatable.
- The **0.05 floor** is not cosmetic. A trust of exactly 0 in a weighted loss deletes that
  interaction from training and changes the effective dataset size.
- Users with **fewer than three interactions** get the median score, not an extreme one: no
  evidence is neither clean nor suspect.

**Why did you replace CoLA?** Because it fails the only test available: 0.2951 ROC AUC on the
synthetic shill benchmark, below chance (see the last section). The behavioural detector reaches
0.9997 on the same benchmark. The benchmark is easy by design, and one indicator (distance from the
item's mean rating) was dropped after it pointed the wrong way there, so the honest claim is that
the detector recognises this attack, not that it detects real fraud.

**What does trust do to the recommender?** It scales each interaction's BPR term. It does **not**
weight edges, because `HGTConv` cannot read an edge weight: the first version of this project stored
the weights on the graph, which made its trust and recency arms identical to the unweighted model.

**Why was CoLA run on a bipartite projection?** (kept as the negative control)
Its published formulation is for homogeneous attributed graphs. Rather than pretend it
natively understands five node types, we project to the user-product bipartite graph, run it
there, and map scores back. That adaptation is documented as an approximation. We also
replaced random-walk-with-restart sampling with uniform 2-hop sampling — cheaper, and on a
bipartite graph a 2-hop neighbour *is* a co-purchasing peer, so it samples nearly the same
neighbourhood.

---

## Temporal modelling

**Is this a temporal GNN?**
**No.** There is no TGN, no TGAT, no temporal attention. It is exponential recency weighting
of the training objective, `w = exp(−λ · age_years)` (λ = 2 per year, chosen on validation), plus
periodic preference snapshots. The project name
invites the opposite reading, so this is stated explicitly everywhere.

**Then why claim "dynamic"?**
Because the weighting addresses drift that is *measured*, not assumed. Phase 2 found a mean
per-user category drift of **0.433** (total-variation between the first and second half of
each user's history). The category-share plot corroborates it independently: the "legacy
systems" share falls from 0.88 to about 0.2, Nintendo Switch appears as a step in **2017** and
PlayStation 5 in **2020** — matching the actual console launches.

**Why is the recency reference `max(train.timestamp)` rather than today?**
Wall-clock "now" would place the reference inside the test period and make results depend on
what date the script was run. Both are disqualifying.

---

## Training and evaluation

**What is negative sampling, and what is the trap?**
For each observed (user, item), we sample items the user has not interacted with. The trap is
excluding only *training* positives — that puts validation and test items into the negative
pool and teaches the model to rank exactly what it is about to be evaluated on below noise,
inflating metrics for free. We exclude positives from **every** split.

**What is BPR loss?**
`L = −log σ(s_pos − s_neg)`. It optimises the pairwise *ordering* rather than an absolute
score, which is what ranking metrics actually measure. Chosen over sampled BCE for that
reason, and because LightGCN and MF use it too — so all models train on like terms.

**What is NDCG@K?**
Discounted cumulative gain normalised by the ideal. Gains are discounted by `1/log2(rank+1)`,
so a hit at rank 1 counts more than one at rank 10. We use binary relevance with the ideal DCG
over `min(K, |relevant|)`.

**Why chronological splitting?**
A random split lets the model train on a user's 2023 purchase and be evaluated on their 2019
one. That is not a recommendation task, it is interpolation.

**How is data leakage prevented?**
- Split by time before any statistic is fitted.
- The training graph contains only training edges.
- Scalers, item mean ratings and CoLA are all fitted on train only, and each records its
  fitted scope in the output.
- Negatives exclude positives from every split.
- Recency reference is the max *training* timestamp.

**Where does leakage still exist? Be honest.**
k-core filtering selects users and items using interaction counts over the *whole* timeline,
including the test period. It is standard practice in the literature, but it is a mild
selection leakage. It is disclosed in every cleaning report and can be disabled in config.

**Why are cold users excluded from the headline metrics?**
They cannot be scored by a model that never saw them. They are counted and reported
separately so a cold-heavy split cannot silently move the headline number in either direction.
Cold start is **not** claimed to be solved.

---

## Explanation, counterfactuals, confidence

**How are explanations generated?**
Evidence is *retrieved* from the graph first — shared category, shared brand, high-trust peers
who bought it, recent activity, semantic similarity, review sentiment. Templates may only
reference retrieved facts. If nothing is retrieved, the system says so rather than producing a
fluent sentence with nothing behind it. That failure mode is what makes explainable
recommendation untrustworthy.

**Are these attention weights?**
No, and we do not claim attention is a faithful explanation. Evidence is retrieved from data
and model embeddings, which is a weaker but honest claim.

**How does the counterfactual work?**
It perturbs the user vector along an interpretable direction (category centroid, brand
centroid, or the item itself) by a scalar α, re-scores, and **bisects on α** for the smallest
value that lifts the item into Top-K. If no α within bounds works, it reports **infeasible**
rather than inventing a reason.

**Why is confidence separate from score?**
Because a model can be maximally confident and wrong, and can rank an item first on almost no
evidence. Confidence blends MC-dropout predictive stability, a saturating function of the
user's interaction volume, and the trust of the supporting evidence. A cold-start user gets
low confidence *even when the score is high* — that behaviour is asserted in the tests.

---

## The hard questions

**What is your actual novelty?**
> Not HGT, SBERT, CoLA, BPR or edge masking. The contribution is an audit-and-repair of a
> trust-aware HGT recommender and what it measured: (i) how much the missing edge masking costs
> (an unmasked HGT equals matrix factorisation; masked it is +47% on the same data); (ii) that a
> recency decay tuned on validation overstates its benefit on a later test period, and refitting
> on train + validation is the fair reading; (iii) that CoLA rates a coordinated campaign as more
> trustworthy and a three-indicator behavioural detector does not; (iv) that the confidence score
> tracks evidence volume and not correctness. Related work on shilling-robust GNN recommenders
> exists (Anti-FakeU, WWW 2023; Trust-GRS, AAAI 2025) and is not claimed.

**Why is this better than a normal HGT recommender?**
Measured, not claimed: the only difference between arm H (a normal HGT) and arm B is edge masking,
and B is +47% NDCG@10. On top of B, recency weighting and trust weighting (arm E) add another +11%,
and a three-seed ensemble about +8% more. See `reports/tables/model_comparison.md`.

**What is the biggest weakness of this project?**
1. **Trust is validated only on synthetic campaigns.** No fraud labels exist. A paraphrased, slowly spread
   campaign does evade the exact-text and burst indicators; the embedding indicator catches it on both
   categories, but the campaign is scripted by the same author as the defence, and on the second category
   it barely hurt accuracy, so the defence's accuracy benefit is not established.
2. **Two datasets and one split each.** The lead over the best baseline fell from +52% to +6% (single model)
   on the second category. On Video_Games, about 60% of test rows involve a user or item never seen in
   training and are excluded, so only warm users are scored.
3. **Small seed counts** (three for HGT arms), and a hyper-parameter search that was one-at-a-time on
   a single seed (about 0.002 NDCG@10 of selection noise).

**Why Video_Games and not Electronics?**
Electronics is 25.95 GB against 30 GB of free disk — measured, not estimated. Video_Games is a
complete 2.5 GB category, so sampling is unbiased and reproducible. Electronics remains
configured as the `full` tier via capped HTTP range streaming, with an explicit sampling-bias
declaration attached to any result from it.

**Why is there no Seller node?**
Amazon Reviews 2023 has no seller field. Inventing one would put fabricated structure into the
graph. For the same reason `brand` is labelled a **proxy** — it comes from `store`, the
storefront name, which usually but not always matches the manufacturer.


---

## The question you will actually be asked

**"Does CoLA detect fake reviews?"**

No. On this project's own synthetic benchmark it scores **0.2951 ROC AUC — below the 0.5
chance line** — meaning it ranks injected shills as *more* trustworthy than genuine users.
A one-line rule (fraction of a user's reviews with duplicated text) scores 0.9895 on the
same data.

**Why?** CoLA scores a node by how predictable its attributes are from its neighbourhood
context. A coordinated campaign shares templates, ratings and targets, so its members are
*unusually easy* to predict from one another. A detector that equates "anomalous" with
"unpredictable from context" rates an organised botnet as exemplary. Isolated eccentric
users look stranger to it than a botnet does.

**What does that mean for the project?** If trust is anti-correlated with manipulation, any use of
it up-weights the manipulated interactions. The original grid's "trust costs 17%" result looked like
confirmation, but that grid was invalid (its weights never reached the model), so it is withdrawn.
The HGT robustness study measures the harm directly by training the same model with working trust,
oracle trust and deliberately inverted trust on a poisoned copy of the data.

**What would you change?** Done: change the *signal* to behaviour (shared text, repeated text,
bursts) instead of neighbourhood predictability. Not done: paraphrase and timing-spread attacks, and
scoring the interaction instead of the user.

**Is this a failed project?** No. The hypothesis that failed was the specific one (a contrastive
graph-anomaly detector yields a useful trust signal); two further results of the first version were
withdrawn because the grid behind them was flawed. What replaced them is a pipeline that beats tuned
baselines by a wide margin, with each component's contribution measured.
