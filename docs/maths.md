# Mathematical Reference

Every symbol is defined. Nothing here is decorative.

## Cosine similarity

$$\text{sim}(a,b) = \frac{a \cdot b}{\lVert a\rVert\,\lVert b\rVert}$$

`a`, `b` are embedding vectors. SBERT output is L2-normalised, so `‖a‖ = ‖b‖ = 1` and the
cosine reduces to the dot product — which is why the recommendation head can use a plain dot
product without a separate normalisation step.

## Recommendation score

$$s(u,i) = e_u \cdot e_i$$

`e_u`, `e_i` are the 128-dimensional user and item embeddings output by the final HGT layer.

## BPR loss

$$\mathcal{L}_{\text{BPR}} = -\frac{1}{|D|}\sum_{(u,i,j)\in D} \log \sigma\big(s(u,i) - s(u,j)\big)$$

`i` is an observed item for user `u`, `j` a sampled item they have not interacted with in
**any** split, `σ` the logistic function. It optimises the ordering `s(u,i) > s(u,j)`, not the
absolute values.

## Edge-masked, weighted BPR (the objective actually trained)

Each epoch a random subset $K$ of the training interactions is kept visible to message passing
(each row kept with probability $p$; all relations encoding the same row share one mask) and the
loss is taken on the **hidden** rows $H = D \setminus K$:

$$\mathcal{L} = -\frac{1}{|H|}\sum_{(u,i,j)\in H} \tilde w_{ui}\,\log \sigma\big(s_K(u,i) - s_K(u,j)\big),
\qquad \tilde w_{ui} = \frac{w_{ui}}{\overline{w}_H}$$

$s_K$ is the score computed from embeddings of the graph that contains only $K$. Without the mask
the target edge is inside the graph that produces $s$, and the model can read the label off it.
The weight $w_{ui}$ is the trust × recency weight below; dividing by its mean over the batch keeps
the average weight at 1, so weighting changes where the gradient goes and not its size.
Evaluation uses the full graph.

## HGT message passing

For a target node `t` with source `s` over relation `e = (τ(s), φ(e), τ(t))`:

$$\text{Attention}(s,e,t) = \underset{\forall s \in N(t)}{\text{softmax}}\left(
K(s)\,W^{ATT}_{\phi(e)}\,Q(t)^{\top}\cdot\frac{\mu_{\langle\tau(s),\phi(e),\tau(t)\rangle}}{\sqrt{d}}\right)$$

$$\text{Message}(s,e,t) = M(s)\,W^{MSG}_{\phi(e)}
\qquad
\tilde{H}[t] = \sum_{s\in N(t)} \text{Attention}(s,e,t)\cdot\text{Message}(s,e,t)$$

`K`, `Q`, `M` are type-specific linear projections of the source and target; `W_φ(e)` are
relation-specific matrices; `μ` is a learned per-relation scaling prior; `d` is the head
dimension; `N(t)` is the neighbourhood of `t`. The point is that **the transformation depends
on both node types and the relation**, which is what makes it heterogeneous rather than a GAT
with extra steps.

## Scaled dot-product attention

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$

`√d_k` prevents the dot products from growing with dimension and pushing softmax into
saturation, where gradients vanish.

## Behavioural anomaly score and trust transformation

For user $u$ with review set $R_u$, three indicators, each in $[0,1]$ after rank normalisation:

$$\text{shared}_u = \frac{|\{r\in R_u:\ \text{text}(r)\text{ posted by} \ge 2\text{ accounts}\}|}{|R_u|},\qquad
\text{repeat}_u = \frac{|\{r\in R_u:\ \text{text}(r)\text{ repeats within } R_u\}|}{|R_u|},\qquad
\text{burst}_u = \frac{\max_t |\{r\in R_u: t_r\in[t,t+24\text{h}]\}|}{|R_u|}$$

$$a_u = \tfrac13\big(\widehat{\text{shared}}_u + \widehat{\text{repeat}}_u + \widehat{\text{burst}}_u\big),
\qquad
t_u = \text{clip}\big(1 - \hat a_u,\; 0.05,\; 1.0\big)$$

where $\hat{\ }$ is rank$/(n-1)$. The rank transform makes the mapping invariant to the score
distribution's shape; users with fewer than three interactions get the median $a_u$. The floor of
0.05 prevents an interaction being silently deleted from a weighted loss. CoLA's unbounded score
went through the same rank-and-clamp mapping and is kept as a control only.

## Temporal decay

$$\tau_e = \max\!\left(\exp\!\left(-\lambda \cdot \frac{\max(0,\; T_{\text{train}} - t_e)}{365.25 \times 86400}\right),\; \tau_{\min}\right)$$

`t_e` is the interaction timestamp, `T_train` the **maximum training timestamp** (never
wall-clock now), `λ` the decay rate per year (selected on validation: 2, so a one-year-old interaction keeps
`e^{-2} ≈ 0.14` of the weight). Clamping the age at 0 stops future rows being *up*-weighted.

## Combined interaction weight

$$w_e = w_{\text{base}} \cdot t_e^{\alpha} \cdot \tau_e^{\beta}$$

Both exponents default to 1. When a component is disabled its factor is exactly 1.0, so an
ablation arm leaves the plain objective untouched. The weight multiplies the BPR term of the
interaction; it is not given to the message-passing layer, because `HGTConv` cannot read an
edge weight.

## Ranking metrics

For user `u`, ranked list `R_u`, held-out relevant set `Rel_u`:

$$\text{Precision@K} = \frac{|R_u^{1:K} \cap Rel_u|}{K}
\qquad
\text{Recall@K} = \frac{|R_u^{1:K} \cap Rel_u|}{|Rel_u|}$$

$$\text{DCG@K} = \sum_{p=1}^{K}\frac{\mathbb{1}[R_u^p \in Rel_u]}{\log_2(p+1)}
\qquad
\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

where `IDCG@K` is the DCG of a perfect ranking, computed over `min(K, |Rel_u|)`.

$$\text{AP@K} = \frac{1}{\min(|Rel_u|,K)}\sum_{p=1}^{K}\text{Precision@}p \cdot \mathbb{1}[R_u^p \in Rel_u]$$

MAP@K is the mean of AP@K over users. MRR is the mean of `1/rank` of the first relevant item.

## Beyond-accuracy

$$\text{Coverage@K} = \frac{\big|\bigcup_u R_u^{1:K}\big|}{|I|}
\qquad
\text{Novelty@K} = \frac{1}{|U|}\sum_u \frac{1}{K}\sum_{i \in R_u^{1:K}} -\log_2 p(i)$$

`p(i)` is item `i`'s share of training interactions. `|I|` is the **post-k-core** catalogue,
not the full Amazon catalogue — reported with that caveat.

## Gini coefficient (popularity concentration)

$$G = \frac{2\sum_{i=1}^{n} i \cdot c_{(i)}}{n\sum_i c_{(i)}} - \frac{n+1}{n}$$

`c_(i)` is the ascending-sorted interaction count of item `i`. `G=0` means every item is
equally popular; `G=1` means one item holds everything. Measured here: **0.521**.

## Total-variation drift

$$D(P,Q) = \tfrac{1}{2}\sum_{c} |P(c) - Q(c)|$$

`P`, `Q` are a user's category distributions over the first and second half of their history.
`0` = identical mix, `1` = disjoint. Measured mean: **0.433** — the empirical premise for
recency weighting.
