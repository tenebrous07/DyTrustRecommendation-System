// Builds docs/DyTrustRec_Research_Paper.docx (IEEE two-column) from the project's result files.
// Usage: node build_paper.js   (from docs/build)

const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, AlignmentType, SectionType, Footer, PageNumber } = require("docx");
const L = require("./lib");
const T = require("./results_tables");
const S = require("./paper_static");
const { loadFacts } = require("./facts");

const S2 = require("./second_dataset");
const F = loadFacts();
const DOCS = path.resolve(__dirname, "..");
const R = T.byModel(F.results || []);
const RR = T.byModel(F.refit || []);
const nd = (map, key) => map[key] ? map[key].metrics["ndcg@10"] : NaN;
const sd = (map, key) => map[key] ? map[key].metrics_std["ndcg@10"] : NaN;
const rc = (map, key) => map[key] ? map[key].metrics["recall@10"] : NaN;
const tv = (family, name) => { const r = F.tuned.records.find((x) => x.model === family && x.name === name); return r ? r["val_ndcg@10"] : NaN; };
const rel = (a, b) => (a - b) / b;
const P = (t, o) => L.para(t, { size: 19, line: 235, after: 70, ...o });
const H1 = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 150, after: 70 }, keepNext: true,
  children: [new TextRun({ text: t.toUpperCase(), size: 18, font: L.FONT })] });
const cap = (t, o) => L.caption(t, o);

const W = (x) => ({ wide: x });   // marks an element that spans both columns
const pending = (x) => (Number.isFinite(x) ? x : null);
const f4 = (x) => L.fmt(x, 4);

// ------------------------------------------------------------- data-driven figures of merit
const ARMS = ["A_hgt_only", "B_hgt_sbert", "C_hgt_sbert_trust", "D_hgt_sbert_recency", "E_hgt_sbert_trust_recency",
  "F_dytrustrec_full", "I_dytrustrec_cosine"];
const bestArm = (map) => ARMS.filter((a) => map[a]).sort((a, b) => nd(map, b) - nd(map, a))[0];
const bestA = bestArm(R), bestB = bestArm(RR);
const mf = R.matrix_factorization, lg = R.lightgcn, pop = R.popularity, knn = R.item_knn_cf;
const BASES = ["matrix_factorization", "lightgcn", "sasrec"];
const bestBaselineName = BASES.filter((k) => R[k]).sort((a, b) => nd(R, b) - nd(R, a))[0];
const bestBaselineRefit = BASES.filter((k) => RR[k]).sort((a, b) => nd(RR, b) - nd(RR, a))[0];
const nUsers = mf ? mf.diagnostics.n_users_evaluated : 1862;
const nUsersRefit = RR.matrix_factorization ? RR.matrix_factorization.diagnostics.n_users_evaluated : null;

// ------------------------------------------------------------- sections
const title = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [new TextRun({
    text: "DyTrustRec: A Dynamic Trust-Aware Explainable Heterogeneous Graph Recommendation System", bold: true, size: 36, font: L.FONT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 }, children: [new TextRun({ text: "Thejesh", size: 22, font: L.FONT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20 }, children: [new TextRun({
    text: "School of Computer Science Engineering and Information Systems, Vellore Institute of Technology, Vellore, India", italics: true, size: 18, font: L.FONT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({
    text: "Reg. No. 23MIA1033 — submitted for the course requirements of Foundation of Data Analytics", italics: true, size: 18, font: L.FONT })] }),
];


function robustnessAbstract() {
  const x = robustnessFacts();
  if (!x) return "";
  return x.within
    ? "; as protection for the recommender, though, no variant, even one given the true labels, recovers the accuracy a 20% scripted campaign costs by more than seed noise"
    : "; as protection for the recommender, the best variant recovers part of the accuracy a 20% scripted campaign costs";
}


function extensionAbstract() {
  const bits = [];
  const c = F.coldStart;
  if (c) {
    const r = c.reserved_slots, t = r.test, k = String(r.chosen_on_validation);
    bits.push(`a separate strip of new products ranked by text similarity lifts NDCG@10 on the mixed list by ${L.signedPct(rel(t[k]["ndcg@10"], t["0"]["ndcg@10"]), 0)} where a single blended score did not help`);
  }
  const d = F.adaptiveDetection;
  if (d) {
    const names = Object.keys(d.attacks), a = d.attacks[names[1]].auc;
    bits.push(`a sentence-embedding gate holds (AUC ${L.fmt(a["indicator: semantic_shared"], 2)}) against a paraphrased, spread-out campaign that defeats exact-text matching (${L.fmt(a["version 1 (mean of ranks)"], 2)})`);
  }
  const f = F.fidelity;
  if (f) {
    const ok = Object.values(f.tests).filter((t) => t.users && t.paired_gap.ci_low > 0).length;
    bits.push(`removing the interactions an explanation cites hurts the recommendation more than removing random ones in ${ok} of ${Object.values(f.tests).filter((t) => t.users).length} tests`);
  }
  return bits.length ? `Three extensions were then measured: ${bits.join("; ")}. ` : "";
}

function abstract() {
  const eB = nd(R, "E_hgt_sbert_trust_recency"), ens = nd(R, "E_hgt_sbert_trust_recency+ens3");
  const lastBest = bestA ? nd(R, bestA) : NaN;
  return [
    P(`**Abstract—**Recommenders usually assume that every training interaction is equally reliable and equally current. DyTrustRec tests that assumption on the Video_Games slice of Amazon Reviews 2023. Review text is encoded for meaning (Sentence-BERT) and for sentiment (a BERT model fine-tuned on product reviews); a Heterogeneous Graph Transformer (HGT) is trained over a graph of users, products, reviews, brands and categories; each interaction's contribution to the training objective is scaled by a behavioural trust score and by recency; and an explanation, counterfactual and confidence layer sits on top. Everything is compared with tuned baselines over several seeds and paired bootstrap tests. Four results stand out. First, an HGT trained the usual way is no better than matrix factorisation (NDCG@10 ${f4(nd(R, "H_hgt_sbert_unmasked"))} against ${f4(nd(R, "matrix_factorization"))}) because the interaction being predicted is also a message-passing edge; hiding a random share of those edges each epoch lifts it to ${f4(nd(R, "B_hgt_sbert"))}, and the full recipe reaches ${f4(eB)} (${f4(ens)} as a three-seed ensemble), ${L.signedPct(rel(ens, nd(R, bestBaselineName)), 0)} over the best baseline (${T.pretty(bestBaselineName)}). Second, recency weighting looks far better on validation data (${L.signedPct(rel(R.D_hgt_sbert_recency["val_ndcg@10"], R.B_hgt_sbert["val_ndcg@10"]), 0)}) than on test data (${L.signedPct(rel(nd(R, "D_hgt_sbert_recency"), nd(R, "B_hgt_sbert")), 0)}) because validation sits right after training; refitting on train plus validation is the fair way to read it. Third, a contrastive graph-anomaly detector (CoLA) rates a coordinated shill campaign as more trustworthy than genuine users (ROC-AUC 0.295), whereas three label-free behavioural indicators reach 0.9997 on the same benchmark${robustnessAbstract()}. Fourth, the learned fusion layer does not beat concatenation. ${extensionAbstract()}${S2.abstractSentence()}None of the components (HGT, SBERT, CoLA, BPR) is new; the contribution is the audit that found which parts of the pipeline did nothing, the repairs, and an evaluation that reports what happened.`, { before: 0, size: 18 }),
    P("***Index Terms—***Heterogeneous Graph Transformer; trust-aware recommendation; shilling attacks; temporal recommendation; Sentence-BERT; explainable recommendation; Bayesian Personalised Ranking; Amazon Reviews 2023.", { size: 18, after: 100 }),
  ];
}

function introduction() {
  return [
    H1("I. Introduction"),
    P("Recommender systems decide a large share of what people see first on shopping platforms. Underneath that sits a narrow question, what will this user probably like next, and the framing leaves two things out. It says nothing about how much of the model's belief rests on reviews that may have been manipulated, and nothing about whether a user's taste has moved since an interaction happened. Coordinated review campaigns are well documented, and a model that weights a purchase from five years ago like one from last week implicitly bets that preferences do not drift."),
    P("DyTrustRec starts from the premise that an interaction is evidence, not fact: its weight should depend on how reliable and how recent it is, with both estimated from the data. We fixed five falsifiable claims before running the experiments in this paper:"),
    ...L.bullets([
      "**Semantic signal helps.** Review text encoded with Sentence-BERT carries preference information that identifiers alone do not.",
      "**Trust helps.** An unsupervised detector can identify unreliable interactions, and down-weighting them should improve ranking or protect it under attack.",
      "**Recency helps.** User interest drifts, so recent interactions should count more.",
      "**Learned fusion beats concatenation.** A trained projection of semantic and behavioural review features should beat concatenating them.",
      "**The training recipe is not the bottleneck.** An HGT trained on its own edges, as is common, is a fair baseline for the architecture.",
    ], { size: 19 }),
    P("An earlier version of this study reported that no HGT arm beat matrix factorisation and that trust and recency did not help. An audit of that version found why: the trust and recency weights were stored as edge weights that HGTConv cannot read, so four of its seven arms were identical models whose score differences were seed noise; early stopping was implemented but never called; the baselines had no hyper-parameter search; and the supervision edge was visible to message passing. This paper reports the repaired study. Section VI tests each claim, and it reports the outcome that did not match the design (the fusion layer) as plainly as the ones that did."),
    P("The Heterogeneous Graph Transformer [1], Sentence-BERT [2], CoLA [3], Bayesian Personalised Ranking [4] and LightGCN [5] are existing methods. We do not claim edge masking as an idea either; it is established link-prediction practice (Section II-D). The contributions are (i) a measured account of how much the missing masking costs an HGT recommender, (ii) a recency-and-trust weighted objective together with the evidence that validation overstates the benefit of recency, (iii) a negative result on CoLA as a source of trust and a behavioural detector that works, with an attack study on the recommender itself, and (iv) a quantitative look at whether the confidence score and the explanations carry information."),
  ];
}

function resultsSection() {
  const out = [H1("VI. Results and Discussion")];
  const nT = "Every number comes from an actual run of the pipeline in Section IV on the dev tier. Hyper-parameters were chosen on validation data only and the test split was scored once per run.";
  out.push(P(nT));

  // A. tuning
  out.push(P("***A. Hyper-Parameter Search (Validation Only).*** Fig. 2 and Table III summarise the search; every row is one training run scored on the validation split, never on test. The share of interaction edges left visible is flat between 0.25 and 0.8 and worse at 0.9; dropout 0.1 is best; a third attention layer hurts; popularity-weighted negatives hurt badly. Recency weighting is the large effect: validation NDCG@10 rises with the decay rate from " + f4(tv("hgt_recency", "rec_0.1")) + " at λ = 0.1 to " + f4(tv("hgt_recency", "rec_2.0")) + " at λ = 2, against " + f4(tv("hgt", "combined")) + " unweighted, and falls again at λ = 4. Trust weighting on clean data is neutral to slightly negative on validation (" + f4(tv("hgt_trust", "trust_1.0")) + " at α = 1)."));
  const fig = path.join(__dirname, "fig_tuning.png");
  if (fs.existsSync(fig)) { out.push(L.imageParagraph(fig, 3.45, 3.45 * 2.0 / 7.1, { title: "tuning" })); out.push(cap("Fig. 2. Validation NDCG@10 against the three settings that mattered. Single seed; differences of about 0.002 are noise.")); }
  out.push(cap("**TABLE III** — Best HGT search results (validation NDCG@10, seed 42).", { after: 40 }), T.tuningTable(F));

  // B. masking
  const B = nd(R, "B_hgt_sbert"), H = nd(R, "H_hgt_sbert_unmasked");
  out.push(P(`***B. Edge Masking.*** Arm H is arm B with masking switched off. Under the same early stopping it peaks at epoch 5 in every seed and scores ${L.pm(H, sd(R, "H_hgt_sbert_unmasked"))} NDCG@10, indistinguishable from matrix factorisation (${L.pm(nd(R, "matrix_factorization"), sd(R, "matrix_factorization"))}). Masking lifts the same architecture to ${L.pm(B, sd(R, "B_hgt_sbert"))}, ${L.signedPct(rel(B, H), 0)}. Fig. 3 shows the symptom: without masking, validation NDCG peaks at epoch 5 and then decays while the training loss keeps falling. An earlier exploratory run without early stopping (not part of the grid) took the training loss from 0.30 to 0.04 over 300 epochs while held-out NDCG fell, which is the signature of reading the label off the graph. The coverage figures of the first version of this study, where every graph arm tripled catalogue coverage, were a symptom of that memorisation and not a design strength.`, { before: 60 }));
  const fig2 = path.join(__dirname, "fig_dynamics.png");
  if (fs.existsSync(fig2)) { out.push(L.imageParagraph(fig2, 3.45, 3.45 * 2.3 / 7.1, { title: "dynamics" })); out.push(cap("Fig. 3. Mean ± SD over seeds. Left: validation NDCG@10. Right: training loss.")); }

  return out;
}


// ------------------------------------------------------------------ C-G
function comparisonSection() {
  const out = [];
  const best = bestA;
  const mfN = nd(R, "matrix_factorization");
  const ensKey = Object.keys(R).filter((k) => k.endsWith("+ens3") || /\+ens\d/.test(k))
    .sort((a, b) => nd(R, b) - nd(R, a))[0];
  out.push(P(`***C. Model Comparison.*** Table IV lists every model on the test split (${nUsers.toLocaleString()} evaluable users), sorted by NDCG@10. The strongest baseline is ${T.pretty(bestBaselineName)} at ${L.pm(nd(R, bestBaselineName), sd(R, bestBaselineName))}; the popularity ranker reaches ${f4(nd(R, "popularity"))} and item-kNN ${f4(nd(R, "item_knn_cf"))}. The best single HGT arm is ${T.pretty(best)} at ${L.pm(nd(R, best), sd(R, best))}, ${L.signedPct(rel(nd(R, best), nd(R, bestBaselineName)), 0)} over that baseline, and a three-seed score ensemble of the same arm (${T.pretty(ensKey || "")}) reaches ${f4(nd(R, ensKey))}, ${L.signedPct(rel(nd(R, ensKey), nd(R, bestBaselineName)), 0)}. Recall@10 moves the same way (${f4(rc(R, best))} against ${f4(rc(R, bestBaselineName))}).`));
  out.push(W(cap("**TABLE IV** — Model comparison on the test split (mean ± SD over seeds; ensembles are single runs).", { after: 40 })));
  out.push(W(T.comparisonTable(F.results)));
  const sig = T.significanceRows(F.significance).filter((r) => r[1] === bestBaselineName && r[0] === T.pretty(best));
  if (sig.length) {
    out.push(P(`The gap is larger than the sampling noise of the test users. A paired bootstrap over users (4,000 draws) on per-user NDCG@10 averaged across seeds gives, for ${sig[0][0]} against ${T.pretty(bestBaselineName)}, the strongest baseline, a mean difference of ${sig[0][2]} with a 95% interval ${sig[0][3]}, which excludes zero.`, { before: 60 }));
  }
  return out;
}

function ablationSection() {
  const out = [];
  const g = (a) => nd(R, a);
  const step = (a, b) => L.signedPct(rel(g(b), g(a)), 1);
  out.push(P("***D. Ablation.*** Table V gives each step's change in test NDCG@10 and whether it exceeds the larger seed standard deviation of the two arms. The five claims from Section I come out as follows.", { before: 60 }));
  out.push(W(cap("**TABLE V** — Ablation steps on the test split.", { after: 40 })), W(T.ablationTable(F.results)));
  const fmtE = (k) => (R[k] ? f4(g(k)) : "n/a");
  out.push(...L.bullets([
    `**Semantic signal.** Adding SBERT and the structured review features to the plain graph (A→B) changes NDCG@10 by ${R.A_hgt_only ? step("A_hgt_only", "B_hgt_sbert") : "n/a"} (${fmtE("A_hgt_only")} to ${fmtE("B_hgt_sbert")}).`,
    `**Trust.** Weighting by behavioural trust on clean data changes B by ${step("B_hgt_sbert", "C_hgt_sbert_trust")} and D by ${step("D_hgt_sbert_recency", "E_hgt_sbert_trust_recency")}: inside the noise on either side. Trust is not meant to raise accuracy on clean data; Section VI-F tests what it does under attack.`,
    `**Recency.** Weighting by recency changes B by ${step("B_hgt_sbert", "D_hgt_sbert_recency")} on test. Section VI-E explains why this is much smaller than on validation and what happens when the model is refit.`,
    `**Fusion.** The learned fusion layer (E→F) changes NDCG@10 by ${step("E_hgt_sbert_trust_recency", "F_dytrustrec_full")}: it does not beat concatenation, and this repeats the direction of the first version of the study.`,
    `**Training recipe.** Removing edge masking (B→H) changes NDCG@10 by ${step("B_hgt_sbert", "H_hgt_sbert_unmasked")}; this is the largest single effect in the study.`,
  ], { size: 19 }));
  return out;
}

function horizonSection() {
  const out = [];
  const vB = R.B_hgt_sbert ? R.B_hgt_sbert["val_ndcg@10"] : NaN, vD = R.D_hgt_sbert_recency ? R.D_hgt_sbert_recency["val_ndcg@10"] : NaN;
  out.push(P(`***E. Why Validation Overstated Recency, and the Refit Protocol.*** On validation data recency weighting raises NDCG@10 from ${f4(vB)} (arm B) to ${f4(vD)} (arm D), ${L.signedPct(rel(vD, vB), 0)}. On the test split the same change is ${L.signedPct(rel(nd(R, "D_hgt_sbert_recency"), nd(R, "B_hgt_sbert")), 0)}. The validation split is the period immediately after training, so the last year of training data is a near-perfect guide to it; the test split lies one period further out, behind a gap that the model never saw. A decay rate tuned on validation is therefore tuned for a horizon the model will not face at test time.`, { before: 60 }));
  if (F.refit) {
    out.push(P(`The standard remedy for a chronological split is to refit after tuning. Each model is retrained on train plus validation for exactly the number of epochs its validation run selected, and scored on the same test split, so the gap between the newest training interaction and the test period is as short as it was between train and validation. The extra interactions also make ${nUsersRefit ? nUsersRefit.toLocaleString() : "more"} test users evaluable (against ${nUsers.toLocaleString()}), so scores in Table VI are not comparable with Table IV; models are compared within a table.`));
    out.push(W(cap("**TABLE VI** — Refit protocol: trained on train + validation, test split (mean ± SD over seeds).", { after: 40 })), W(T.comparisonTable(F.refit)));
    const rb = nd(RR, "B_hgt_sbert"), rd = nd(RR, "D_hgt_sbert_recency"), re = nd(RR, "E_hgt_sbert_trust_recency");
    out.push(P(`Under refit, recency weighting is worth ${L.signedPct(rel(rd, rb), 0)} (arm D against arm B), and the best baseline is ${T.pretty(bestBaselineRefit)} at ${f4(nd(RR, bestBaselineRefit))}. The unweighted model also loses ground when refit (arm B ${f4(nd(R, "B_hgt_sbert"))} to ${f4(rb)}) while the recency-weighted arms hold or gain; with no validation split to pick a checkpoint, the unweighted model's noisy training curve costs it.`, { before: 60 }));
  }
  return out;
}


function robustnessFacts() {
  const rb = F.robustness && F.robustness.summary;
  if (!rb || !rb["0.2"]) return null;
  const r = rb["0.2"], g = (c) => (r[c] ? r[c]["ndcg@10"] : null);
  const clean = rb["0.0"] && rb["0.0"].none ? rb["0.0"].none["ndcg@10"] : null;
  const exposure = Math.max(...Object.values(r).map((v) => v.users_with_target_in_top10.mean));
  const defended = ["cola", "trust", "trust+gate", "oracle"].filter((c) => r[c]);
  const means = defended.map((c) => g(c).mean).concat(g("none").mean);
  const spread = Math.max(...Object.values(r).map((v) => v["ndcg@10"].std));
  const nSeeds = Math.min(...Object.values(r).map((v) => v.n_seeds));
  return { r, g, clean, exposure, spread, nSeeds, within: Math.max(...means) - Math.min(...means) < 2 * spread,
    gateClean: rb["0.0"] && rb["0.0"]["trust+gate"] ? rb["0.0"]["trust+gate"]["ndcg@10"] : null,
    trustClean: rb["0.0"] && rb["0.0"].trust ? rb["0.0"].trust["ndcg@10"] : null };
}

function robustnessParagraph() {
  const x = robustnessFacts();
  if (!x) return null;
  const f = (v) => L.fmt(v.mean, 4);
  let t = `With ${x.nSeeds} seeds per condition at a 20% injection rate, the undefended model's NDCG@10 on the clean test split is ${f(x.g("none"))}${x.clean ? `, down from ${f(x.clean)} without shills (${L.signedPct(rel(x.g("none").mean, x.clean.mean), 0)})` : ""}. ${x.r.cola ? `Weighting the loss by CoLA trust gives ${f(x.g("cola"))}. ` : ""}Weighting the loss by behavioural trust gives ${f(x.g("trust"))}; trust plus the gate (rows of users whose reviews are mostly text other accounts also posted are removed from the graph and the objective) ${f(x.g("trust+gate"))}; oracle trust built from the true labels ${f(x.g("oracle"))}; deliberately inverted trust ${f(x.g("inverted"))}. `;
  t += x.within
    ? `The largest seed standard deviation is ${L.fmt(x.spread, 4)}, and the spread between the undefended, trust-weighted, gated and oracle conditions is less than twice that, so this study cannot show that any defence repairs the damage: the attack hurts the HGT, and the oracle that knows exactly who the shills are does no better than the rest within noise. `
    : `The best defence recovers part of the loss, and the gap to the undefended model is larger than twice the largest seed standard deviation (${L.fmt(x.spread, 4)}). `;
  t += `The attack barely reaches users: the share shown a promoted item in their top ten is at most ${(100 * x.exposure).toFixed(1)}% in every condition, so the harm is to accuracy and not to targeted promotion.`;
  if (x.gateClean && x.trustClean) t += ` On clean data the gate costs ${L.signedPct(rel(x.gateClean.mean, x.clean ? x.clean.mean : x.gateClean.mean), 0)} NDCG@10 and trust weighting ${L.signedPct(rel(x.trustClean.mean, x.clean ? x.clean.mean : x.trustClean.mean), 0)}.`;
  return t;
}

function trustSection() {
  const out = [];
  out.push(P("***F. Trust.*** Real Amazon data has no manipulation labels, so the trust component can only be validated on a perturbed copy of the training split with scripted shill accounts added (Module 45). The original data is never modified, and every number in this subsection is a synthetic-setting result: it measures resistance to this attack, not detection of real fraud. At the default 5% injection the campaign is 785 accounts, 11,775 rows and 78 target items, with shared templates, a forced five-star rating and thirty-minute bursts.", { before: 60 }));
  const det = T.detectionTable(F);
  if (det) out.push(cap("**TABLE VII** — Detecting synthetic shills, ROC-AUC (chance = 0.5).", { after: 40 }), det);
  out.push(P("CoLA is the only detector below chance. It scores a node by how predictable its attributes are from its neighbourhood, and a coordinated campaign is internally consistent by construction, so each shill is easy to predict from the others and is rated more trustworthy than a genuine user. The behavioural detector asks instead whether the account behaves like a campaign account. Two cautions apply. The benchmark is easy by design (the injected accounts share templates and bursts), so a high AUC shows the detector recognises this generative process; and one indicator that pointed the wrong way here (distance from the item's mean rating) was dropped, a choice made after seeing the benchmark.", { before: 60 }));
  const rob = T.robustnessTable(F);
  if (rob) {
    out.push(P("Detection is not the goal; protecting the recommender is. Table VIII trains the same HGT (arm B settings) on the poisoned split under six conditions (none; CoLA trust, the original design; behavioural trust; behavioural trust plus the gate; oracle trust from the true labels; and deliberately inverted trust) and reports NDCG@10 on the clean test split, and the share of users who are shown a promoted item in their top ten, which is what an attacker wants. Soft trust weighting acts on the loss only; the gate also removes the flagged rows from message passing, because the HGT layer passes messages along every edge it is given.", { before: 60 }));
    out.push(cap("**TABLE VIII** \u2014 HGT under a synthetic shill campaign (clean test split).", { after: 40 }), rob);
    const para = robustnessParagraph();
    if (para) out.push(P(para, { before: 60 }));
  }
  return out;
}

function confidenceSection() {
  const out = [];
  const C = F.confidence;
  if (!C) return out;
  const a = C.auroc, ex = C.explanation, q = C.reliability, cs = C.cold_start;
  const useful = a.confidence > 0.55;
  out.push(P(`***G. Confidence and Explanations.*** We asked whether the confidence score earns its place by testing it on ${C.pairs.toLocaleString()} (user, top-10 item) pairs from a ${C.members}-member ensemble of arm ${T.pretty(C.arm)}: does higher confidence go with the user actually interacting with the item? It does not. The AUROC of the full blend is ${L.fmt(a.confidence, 3)}, with ensemble stability at ${L.fmt(a.stability, 3)}, evidence volume ${L.fmt(a.volume, 3)}, evidence trust ${L.fmt(a.trust, 3)}, and the raw ranking score itself only ${L.fmt(a.score, 3)}. The hit rate in the lowest and highest confidence quintiles is ${L.fmt(q[0].hit_rate, 4)} and ${L.fmt(q[q.length - 1].hit_rate, 4)} (base rate ${L.fmt(C.base_hit_rate, 4)}); the lowest quintile consists of cold-start users, and why they are hit more often is not something this analysis explains. What the score does track is what it was specified to track, evidence sufficiency: mean confidence is ${L.fmt(cs.mean_confidence_cold, 3)} for users with fewer than three training interactions against ${L.fmt(cs.mean_confidence_warm, 3)} for the rest. It should therefore be read as a flag for thin evidence, not as the probability that a recommendation is right.`, { before: 60 }));
  if (ex) out.push(P(`The same holds for explanations. ${(100 * ex.pairs_with_evidence).toFixed(1)}% of recommendations had at least one retrieved evidence item (mean ${L.fmt(ex.mean_evidence_items, 2)}), and the hit rate with evidence (${L.fmt(ex.hit_rate_supported, 4)}) is no higher than without (${L.fmt(ex.hit_rate_unsupported, 4)}). Explanations are grounded in retrieved facts by construction, which is what they promise; they are not evidence of correctness.`));
  const live = F.live;
  if (live) out.push(P(`**An un-curated example.** ${live.summary.charAt(0).toUpperCase() + live.summary.slice(1)}`, { before: 40 }));
  return out;
}

function limitations() {
  return [
    H1("VII. Limitations"),
    ...L.bullets([
      "**Two categories, one split each.** Everything is Amazon Reviews 2023, Video_Games and Musical_Instruments, each with a single chronological 80/10/10 split. Section VI-K shows which findings repeat and which do not; no other category, no non-Amazon domain and no other split protocol was tried.",
      `**Few evaluable users.** The strict temporal split leaves ${nUsers.toLocaleString()} of 17,223 users scoreable (about 60% of test rows involve a user or item never seen in training). Cold start is not solved here, only reported separately.`,
      "**Small seed counts.** Baselines have five seeds and HGT arms three (two for the CoLA control). Standard deviations over three seeds are rough, so the ablation flags only changes larger than the seed spread, and the paired bootstrap is over users, not seeds.",
      "**Search effort is not identical.** The HGT search ran about twenty-five single-seed validation jobs, the matrix-factorisation grid eighteen, LightGCN twenty-four, item-kNN five. One-at-a-time search on one seed has a selection noise of about 0.002 NDCG@10, as the non-monotone dropout curve shows.",
      "**What to try was informed by test curves.** Before the protocol in Section V was fixed, a short exploratory phase of about ten HGT variants was run while watching test curves; that is where the edge-masking idea came from. The reported numbers come from a fresh pipeline that selects on validation data only, but the choice of what to try was not blind.",
      "**Trust is validated only on a synthetic, easy benchmark.** The injected accounts share templates and bursts, so the behavioural detector's near-perfect AUC shows it recognises this attack, not real fraud. A paraphrased, slowly spread campaign does evade the exact-text and burst indicators (Section VI-I); the embedding indicator catches it, but that campaign is also scripted by the same author as the defence, and no real manipulation data or published attack benchmark was used. On clean data the gate removed only 262 of 115,342 training rows, and Table VIII shows what even that cost; two seeds cannot say whether it is noise.",
      "**Recency is not a temporal GNN.** Despite “Dynamic” in the name there is no temporal attention or memory; recency enters as an exponential weight whose decay rate was tuned for one horizon, which is why Section VI-E exists.",
      "**Explanations and confidence were not evaluated with people.** The counterfactual module perturbs the user vector in embedding space; it is not a counterfactual on the graph, and it was not tested for fidelity against actually removing interactions.",
      "**Baselines and published numbers.** SASRec is included and tuned like the others, but no LLM-based recommender is. The protocol (global chronological split, multi-item targets, warm users) differs from the leave-one-out protocol used by most published results on this dataset, so no claim of being above or below published figures is made.",
      "**The second dataset replicates, it does not re-tune the method.** The HGT settings were searched again on the new validation split (the transferred settings scored lower, as Section VI-K reports), but no design choice was changed after seeing either test split. One more category is a small sample of catalogues: the lead over the best baseline fell from +52% to +6% for a single model, and nothing here says what a third category would show.",
      "**The new-product strip is not the graph model.** It ranks by text similarity, the value of r was chosen on validation, and it was measured on one split. The explanation-fidelity test covers only the 81 and 73 test users whose cited rows are a minority of their history, and only one trained model.",
      "**Other disclosed limits.** The five-core filter uses counts over the whole timeline, a mild and standard leakage; brand is a proxy derived from the storefront field; coverage and novelty are relative to the post-cleaning catalogue; Apple's MPS backend is not bit-reproducible.",
    ], { size: 19 }),
  ];
}

function conclusion() {
  const E = nd(R, "E_hgt_sbert_trust_recency"), ens = nd(R, "E_hgt_sbert_trust_recency+ens3"), m = nd(R, "matrix_factorization");
  const bestRefit = bestB ? nd(RR, bestB) : NaN;
  const mfRefit = nd(RR, bestBaselineRefit);
  return [
    H1("VIII. Conclusion and Future Work"),
    P(`The repaired DyTrustRec pipeline answers the five claims set out in Section I. SBERT semantics add ${L.signedPct(rel(nd(R, "B_hgt_sbert"), nd(R, "A_hgt_only")), 1)}, which is inside the seed spread here and on the second category. Behavioural trust is neutral on clean data, and the attack study shows how much of an attack it can and cannot repair (Table VIII). Recency helps a great deal once the model is refit (${F.refit ? L.signedPct(rel(nd(RR, "D_hgt_sbert_recency"), nd(RR, "B_hgt_sbert")), 0) : "n/a"} for arm D over arm B) but much less on a split with a horizon gap. The learned fusion layer does not beat concatenation, in either protocol. And the training recipe was the bottleneck: without edge masking an HGT recommender matches matrix factorisation (${f4(nd(R, "H_hgt_sbert_unmasked"))} against ${f4(m)}), with it the same architecture reaches ${f4(E)} (${f4(ens)} as a three-seed ensemble), and under the refit protocol the best arm reaches ${f4(bestRefit)} against ${f4(mfRefit)} for the strongest baseline. ${S2.conclusionSentence()}`),
    P("Two earlier headline claims did not survive the audit and are withdrawn: that no graph arm could beat matrix factorisation, and that trust weighting costs 17% of accuracy because CoLA inverts trust. The first was a training-recipe fault; the second was a comparison between models that were identical. CoLA's inversion on coordinated campaigns is real and is measured here directly, but its effect on ranking is what Table VIII shows, not what the first ablation implied."),
    P("What the extensions settled, and what they leave open. Against a campaign that paraphrases and spreads its reviews, exact-text matching fails and a sentence-embedding detector finds the campaign on both categories (ROC-AUC 0.994 and 0.997); whether removing it protects accuracy was suggestive on Video_Games and could not be shown on Musical_Instruments, where the attack barely hurt the model. Against real, labelled manipulation neither is tested. New products can be recommended through a separate text-based strip, which helped where most test purchases are new releases and switched itself off where they are not; it does not replace a content path for new USERS. The explanations pass a removal test on both categories. Four follow-ups remain. First, a third and larger category, and a domain outside Amazon: the lead over strong baselines shrank on the second category and nothing says where it settles. Second, comparison with published defences such as Anti-FakeU and Trust-GRS on their attacks, which this study did not run. Third, learned relative time encodings in the attention instead of one tuned decay rate. Fourth, a live A/B test, the only evidence that offline ranking gains reach users."),
  ];
}


// ------------------------------------------------------------------ H-K: extensions
function coldStartSection() {
  const c = F.coldStart;
  if (!c) return [];
  const out = [];
  const share = c.test_rows_on_new_items / c.test_rows;
  const chosen = c.reserved_slots.chosen_on_validation, gain = c.reserved_slots_gain_over_none;
  const t = c.reserved_slots.test;
  out.push(P(`***H. New Products.*** The comparison above scores only products that training saw, which leaves out most of what users actually do next: ${(100 * share).toFixed(0)}% of the ${c.test_rows.toLocaleString()} test interactions of known users are with ${c.new_items.toLocaleString()} products that did not exist in the training split. A model with no embedding for them cannot recommend them. We tested two ways to add a content path, both fitted on training items only and both using the title and description encoded with Sentence-BERT. Mapping the text into the model's embedding space with a ridge regression (cross-validated R${"²"} ${L.fmt(c.embedding_mapping.reduce((a, m) => a + m.cv_r2, 0) / c.embedding_mapping.length, 2)}) ranks new products worse than plain text similarity between the product and the user's past items (NDCG@10 ${L.fmt(c.new_products_only["model embedding mapped from text"]["ndcg@10"], 4)} against ${L.fmt(c.new_products_only["text similarity"]["ndcg@10"], 4)}, with random at ${L.fmt(c.new_products_only.random["ndcg@10"], 4)}), and a single blended score for both kinds of product did not help either: a product with no history cannot compete for a top-ten place against one with thousands of interactions.`, { before: 60 }));
  out.push(P(`What works is giving new products their own places. Reserving the last r of the ten positions for the best new products by text similarity, with r chosen on validation data (r = ${chosen}), lifts test NDCG@10 from ${L.fmt(t["0"]["ndcg@10"], 4)} to ${L.fmt(t[String(chosen)]["ndcg@10"], 4)} and Recall@10 from ${L.fmt(t["0"]["recall@10"], 4)} to ${L.fmt(t[String(chosen)]["recall@10"], 4)}; the paired bootstrap difference is ${gain.mean_diff >= 0 ? "+" : "−"}${L.fmt(Math.abs(gain.mean_diff), 4)} with a 95% interval [${L.fmt(gain.ci_low, 4)}, ${L.fmt(gain.ci_high, 4)}]. The gain keeps growing with r (Table IX), which is a statement about this catalogue, where most next purchases are new releases, and not an argument for filling the list with new items. This is a separate "new for you" strip, as shops run it, and it is a content-similarity ranker, not part of the graph model.`));
  out.push(L.caption("**TABLE IX** — Reserved new-arrival slots (test split, known users).", { after: 40 }), T.coldStartTable(F));
  return out;
}

function adaptiveSection() {
  const d = F.adaptiveDetection;
  if (!d) return [];
  const out = [];
  const [standard, adaptive] = Object.keys(d.attacks);
  const a = d.attacks[adaptive].auc, s0 = d.attacks[standard].auc;
  out.push(P(`***I. A Campaign That Knows the Detector.*** The standard synthetic campaign repeats eight templates in thirty-minute bursts, which is what exact-text and one-day-burst indicators look for, so a near-perfect score against it says little. We therefore added an adaptive campaign: every review is paraphrased (synonyms, openers, extra sentences, clause order) and the reviews are spread over thirty days. Against it the original detector falls to ROC-AUC ${L.fmt(a["version 1 (mean of ranks)"], 3)}, essentially chance, and the one-day burst indicator points the wrong way (${L.fmt(a["indicator: burstiness"], 3)}) because genuine users are burstier than the spread-out shills. Comparing sentence embeddings with other accounts' reviews (an account's share of reviews that read like at least three reviews by other accounts) holds at ${L.fmt(a["indicator: semantic_shared"], 3)} against the adaptive campaign and ${L.fmt(s0["indicator: semantic_shared"], 3)} against the standard one (Table X). Averaging the ranks of all indicators dilutes that signal (${L.fmt(a["version 2 (mean of ranks)"], 3)}); taking the strongest indicator keeps it (${L.fmt(a["version 2 (strongest indicator)"], 3)}). The lesson for the design is that an equal-weight average is only as good as its worst indicator, and an attacker chooses which one that is.`, { before: 60 }));
  out.push(L.caption("**TABLE X** — Detection ROC-AUC against the standard and the adaptive synthetic campaign (SYNTHETIC).", { after: 40 }), T.adaptiveDetectionTable(F));
  const rob = T.robustnessTable(F, "robustnessAdaptive");
  if (rob) {
    out.push(P(robustnessAdaptiveText(), { before: 60 }));
    out.push(L.caption("**TABLE XI** — HGT under the adaptive campaign (clean test split, SYNTHETIC).", { after: 40 }), rob);
  }
  return out;
}

function robustnessAdaptiveText() {
  const rb = F.robustnessAdaptive && F.robustnessAdaptive.summary;
  if (!rb || !rb["0.2"]) return "";
  const r = rb["0.2"], g = (c) => (r[c] ? r[c]["ndcg@10"] : null), f = (v) => L.fmt(v.mean, 4);
  const clean = rb["0.0"] && rb["0.0"].none ? rb["0.0"].none["ndcg@10"] : null;
  const spread = Math.max(...Object.values(r).map((v) => v["ndcg@10"].std));
  const names = { none: "no defence", "trust+gate": "the exact-text gate", gate_v2: "the semantic gate", oracle: "oracle trust" };
  const parts = Object.keys(names).filter((c) => g(c)).map((c) => `${names[c]} ${f(g(c))}`);
  let t = `Table XI trains the HGT on the poisoned split. At a 20% injection rate NDCG@10 on the clean test split is ${parts.join(", ")}${clean ? `, against ${f(clean)} without shills` : ""}. `;
  if (g("gate_v2") && g("trust+gate")) {
    const better = g("gate_v2").mean - g("trust+gate").mean;
    t += Math.abs(better) > 2 * spread
      ? `The semantic gate recovers more than the exact-text gate by ${L.fmt(Math.abs(better), 4)}, more than twice the largest seed standard deviation (${L.fmt(spread, 4)}). `
      : `The difference between the semantic and the exact-text gate (${L.fmt(Math.abs(better), 4)}) is within twice the largest seed standard deviation (${L.fmt(spread, 4)}), so this study cannot rank them. `;
  }
  const rowsOf = (c) => (r[c] && r[c].rows_gated ? Math.round(r[c].rows_gated.mean) : null);
  if (rowsOf("trust+gate") !== null && rowsOf("gate_v2") !== null) {
    t += `The reason is visible in how many rows each gate removed: the exact-text gate removed ${rowsOf("trust+gate").toLocaleString()} rows, none of them from the campaign, because no two paraphrased reviews are identical, while the semantic gate removed ${rowsOf("gate_v2").toLocaleString()}, essentially the whole campaign. ${g("oracle_gate") ? `Removing the shills' rows using the true labels, the ceiling for any gate, gives ${f(g("oracle_gate"))}, so the semantic gate is as good as a gate can be here. ` : ""}${g("none") && g("gate_v2") ? `Against no defence the semantic gate's gain is ${L.fmt(g("gate_v2").mean - g("none").mean, 4)}, about ${((g("gate_v2").mean - g("none").mean) / Math.sqrt(g("gate_v2").std ** 2 / r.gate_v2.n_seeds + g("none").std ** 2 / r.none.n_seeds)).toFixed(1)} standard errors with ${r.gate_v2.n_seeds} and ${r.none.n_seeds} seeds: suggestive, not conclusive.` : ""}`;
  }
  return t;
}

function fidelitySection() {
  const f = F.fidelity, v = F.serving;
  const out = [];
  if (f) {
    const tests = Object.entries(f.tests).filter(([, t]) => t.users);
    const verdict = tests.map(([k, t]) => {
      const lo = t.paired_gap.ci_low, hi = t.paired_gap.ci_high;
      const kind = k.replace("shared_", "");
      return lo > 0
        ? `for ${kind} evidence, removing the cited interactions hurts the recommendation more than removing as many random ones (difference ${t.paired_gap.mean_diff >= 0 ? "+" : "−"}${L.fmt(Math.abs(t.paired_gap.mean_diff), 3)}, 95% interval [${L.fmt(lo, 3)}, ${L.fmt(hi, 3)}])`
        : hi < 0
          ? `for ${kind} evidence, the cited interactions matter less than random ones (difference ${L.fmt(t.paired_gap.mean_diff, 3)}, interval [${L.fmt(lo, 3)}, ${L.fmt(hi, 3)}])`
          : `for ${kind} evidence, removing the cited interactions is not distinguishable from removing random ones (difference ${L.fmt(t.paired_gap.mean_diff, 3)}, interval [${L.fmt(lo, 3)}, ${L.fmt(hi, 3)}])`;
    });
    out.push(P(`***J. Are the Explanations Faithful?*** An explanation built from retrieved facts is true of the data; whether the model used those facts is a separate question. For sampled test users we explained the top recommendation, removed from the graph the user's own training interactions that the explanation cites (their items in the cited category, or from the cited brand), re-ran the trained model, and measured the fall in that recommendation's score in units of the user's score spread. The control removes as many of the user's other interactions at random. Results (Table XII): ${verdict.join("; ")}.`, { before: 60 }));
    out.push(L.caption("**TABLE XII** — Removing the interactions an explanation cites (test users, top-1 recommendation).", { after: 40 }), T.fidelityTable(F));
  }
  if (v) {
    const m = v.serving;
    out.push(P(`***K. Serving and Training Cost.*** Serving a user is a dot product and a top-k: ${L.fmt(m[0].ms_per_user_single, 2)} ms per user for this catalogue on one CPU, ${L.fmt(m[1].ms_per_user_single, 1)} ms at 100,000 products and ${L.fmt(m[2].ms_per_user_single, 0)} ms at a million (synthetic vectors), so brute force is enough here and an approximate index only becomes necessary beyond roughly a million products. Training is the part that does not scale: the HGT trains full-graph, one step per epoch over all ${F.graph.nodes} nodes, at ${Math.round(v.mean_train_seconds.E_hgt_sbert_trust_recency)} s per run against ${Math.round(v.mean_train_seconds.matrix_factorization)} s for matrix factorisation. Production use would need mini-batch neighbour sampling, which was tried earlier in this project and was slower and no better at that budget.`, { before: 60 }));
  }
  return out;
}

function confidenceCalibrationParagraph() {
  const C = F.confidence;
  if (!C || !C.calibrated) return [];
  const c = C.calibrated;
  const q = c.reliability;
  return [P(`Because the evidence-sufficiency score does not answer "how likely is this to be taken", we added a second number. A logistic regression on five features known at recommendation time (ensemble score, rank, ensemble stability, history length, user trust) is fitted on ${c.validation_pairs.toLocaleString()} validation recommendations and scored once on test. Its AUROC is ${L.fmt(c.auroc, 3)}, against ${L.fmt(C.auroc.confidence, 3)} for the blend, and the hit rate rises from ${L.fmt(q[0].observed, 4)} in the lowest predicted quintile to ${L.fmt(q[q.length - 1].observed, 4)} in the highest. It ranks well but is not calibrated in level: the validation period's hit rate (${L.fmt(c.validation_base_rate, 4)}) is nearly double the test period's (${L.fmt(C.base_hit_rate, 4)}), so the predicted probabilities are about twice too high (expected calibration error ${L.fmt(c.ece, 4)}) and the Brier score does not beat a constant (${L.fmt(c.brier, 5)} against ${L.fmt(c.brier_constant_base_rate, 5)}). A deployed system would refit the intercept on the most recent period.`)];
}

const REFS = [
  "Z. Hu, Y. Dong, K. Wang, and Y. Sun, “Heterogeneous Graph Transformer,” in *Proc. The Web Conference (WWW '20)*, 2020, pp. 2704–2710.",
  "N. Reimers and I. Gurevych, “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,” in *Proc. EMNLP*, 2019, pp. 3982–3992.",
  "Y. Liu, Z. Li, S. Pan, C. Gong, C. Zhou, and G. Karypis, “Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning,” *IEEE Trans. Neural Netw. Learn. Syst.*, vol. 33, no. 6, pp. 2378–2392, 2021.",
  "S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme, “BPR: Bayesian Personalized Ranking from Implicit Feedback,” in *Proc. UAI*, 2009, pp. 452–461.",
  "X. He, K. Deng, X. Wang, Y. Li, Y. Zhang, and M. Wang, “LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation,” in *Proc. ACM SIGIR*, 2020, pp. 639–648.",
  "Y. Hou, J. Li, Z. He, A. Yan, X. Chen, and J. McAuley, “Bridging Language and Items for Retrieval and Recommendation,” arXiv:2403.03952, 2024.",
  "X. You, C. Lee, D. Ding, M. Zhang, F. Feng, X. Pan, and M. Yang, “Anti-FakeU: Defending Shilling Attacks on Graph Neural Network based Recommender Model,” in *Proc. The ACM Web Conference (WWW '23)*, 2023.",
  "L. Mu, Z. Liu, Z. Zhu, and Z. Lin, “Trust-GRS: A Trustworthy Training Framework for Graph Neural Network Based Recommender Systems Against Shilling Attacks,” in *Proc. AAAI Conf. Artif. Intell.*, vol. 39, no. 12, 2025, pp. 12408–12416.",
  "S. Tang, L. Jin, and F. Cheng, “Fraud Detection in Online Product Review Systems via Heterogeneous Graph Transformer,” *IEEE Access*, vol. 9, pp. 167364–167373, 2021.",
  "Y. Rong, W. Huang, T. Xu, and J. Huang, “DropEdge: Towards Deep Graph Convolutional Networks on Node Classification,” in *Proc. ICLR*, 2020.",
  "Y. Ding and X. Li, “Time Weight Collaborative Filtering,” in *Proc. ACM CIKM*, 2005, pp. 485–492.",
  "W.-C. Kang and J. McAuley, “Self-Attentive Sequential Recommendation,” in *Proc. IEEE ICDM*, 2018, pp. 197–206.",
];

function references() {
  return [H1("References"), ...REFS.map((t, i) => L.para(`[${i + 1}] ${t}`, { size: 16, line: 220, after: 40, align: AlignmentType.LEFT, indent: { left: 300, hanging: 300 } }))];
}

module.exports = { F, R, RR, nd, sd, rc, tv, rel, P, H1, cap, f4, title, abstract, introduction, resultsSection,
  coldStartSection, adaptiveSection, fidelitySection, confidenceCalibrationParagraph,
  limitations, conclusion, W, comparisonSection, ablationSection, horizonSection, trustSection, confidenceSection, references, bestA, bestB };
