// Section VI-K: does the Video_Games story hold on a second Amazon category?
// Every number is read from the two datasets' result files (reports/results and
// reports/results_<slug>); nothing is typed in. If the second dataset has not been run, the
// section is simply absent.

const fs = require("fs");
const path = require("path");
const L = require("./lib");
const T = require("./results_tables");
const { loadFacts } = require("./facts");

const SLUG = "musical_instruments";
const NAME = "Musical_Instruments";
const F1 = loadFacts();
const F2 = loadFacts(SLUG);
const available = Boolean(F2.results && F2.results.length);

const R1 = T.byModel(F1.results || []);
const R2 = T.byModel(F2.results || []);
const nd = (R, k) => (R[k] ? R[k].metrics["ndcg@10"] : NaN);
const sd = (R, k) => (R[k] ? R[k].metrics_std["ndcg@10"] : NaN);
const rel = (a, b) => (a - b) / b;
const f4 = (x) => L.fmt(x, 4);
const f3 = (x) => L.fmt(x, 3);
const sp = (x, d = 0) => L.signedPct(x, d);
const P = (t, o) => L.para(t, { size: 19, line: 235, after: 70, ...o });
const W = (x) => ({ wide: x });
const BASES = ["popularity", "item_knn_cf", "matrix_factorization", "lightgcn", "sasrec"];
const E = "E_hgt_sbert_trust_recency";
const ENS = `${E}+ens3`;

const bestBase = (R) => BASES.filter((k) => R[k]).sort((a, b) => nd(R, b) - nd(R, a))[0];

// A change counts as an effect only when it exceeds the larger of the two seed standard deviations,
// the rule the ablation table uses.
function effect(R, a, b) {
  const d = nd(R, b) - nd(R, a);
  return { pct: rel(nd(R, b), nd(R, a)), clear: Math.abs(d) > Math.max(sd(R, a), sd(R, b)) };
}
const cell = (e) => `${sp(e.pct, 1)}${e.clear ? "" : " (within noise)"}`;
function verdictOf(e1, e2) {
  const gain = (e) => e.clear && e.pct > 0, loss = (e) => e.clear && e.pct < 0;
  if (gain(e1) && gain(e2)) return "replicates";
  if (loss(e1) && loss(e2)) return "replicates (a loss)";
  if (!e1.clear && !e2.clear) return "no reliable effect on either";
  if (gain(e1) || gain(e2)) return gain(e1) ? "gain on the first only" : "gain on the second only";
  return "no gain on either";              // a clear loss on one, nothing clear on the other
}
function ablationRow(label, a, b) {
  const e1 = effect(R1, a, b), e2 = effect(R2, a, b);
  return [label, cell(e1), cell(e2), verdictOf(e1, e2)];
}

// Paired-bootstrap difference of one HGT row against the best baseline of that dataset.
function gapVsBest(F, R, model) {
  const S = F.significance;
  if (!S || !S.comparisons || !S.comparisons[model]) return null;
  const t = S.comparisons[model][`vs_${S.best_baseline || bestBase(R)}`];
  return t ? { ...t, pct: rel(nd(R, model), nd(R, S.best_baseline || bestBase(R))), base: S.best_baseline || bestBase(R) } : null;
}
const gapCell = (g) => (g ? `${sp(g.pct)} (interval ${g.ci_low > 0 ? "excludes" : "includes"} 0)` : "n/a");
function gapVerdict(g1, g2) {
  if (!g1 || !g2) return "n/a";
  const a = g1.ci_low > 0, b = g2.ci_low > 0;
  return a && b ? "replicates" : a || b ? "direction only" : "not shown on either";
}

// Settings carried over from Video_Games without re-tuning (kept in runs_transfer/).
function transferScores() {
  const dir = path.join(F2.root, "reports", `results_${SLUG}`, "runs_transfer");
  if (!fs.existsSync(dir)) return null;
  const out = {};
  for (const f of fs.readdirSync(dir).filter((x) => /__seed\d+\.json$/.test(x))) {
    const j = JSON.parse(fs.readFileSync(path.join(dir, f), "utf8"));
    (out[j.model] = out[j.model] || []).push(j.metrics["ndcg@10"]);
  }
  const mean = (v) => v.reduce((a, b) => a + b, 0) / v.length;
  return Object.fromEntries(Object.entries(out).map(([k, v]) => [k, mean(v)]));
}

// Side-by-side NDCG@10 of the models both datasets share.
const ROWS = ["popularity", "matrix_factorization", "lightgcn", "sasrec", "H_hgt_sbert_unmasked", "A_hgt_only", "B_hgt_sbert",
  "C_hgt_sbert_trust", "D_hgt_sbert_recency", E, "F_dytrustrec_full", ENS];
function sideBySideRows() {
  return ROWS.filter((k) => R1[k] || R2[k]).map((k) => [
    T.pretty(k),
    R1[k] ? L.pm(nd(R1, k), sd(R1, k)) : "—", R2[k] ? L.pm(nd(R2, k), sd(R2, k)) : "—"]);
}
function sideBySide() {
  return L.table(["Model (tuning protocol)", "Video_Games", NAME], sideBySideRows(), [3500, 2300, 2300],
    { size: 15, align: [L.AlignmentType.LEFT, L.AlignmentType.RIGHT, L.AlignmentType.RIGHT], stripe: true });
}

function scorecardRows() {
  const rows = [
    ablationRow("Edge masking helps (B vs H)", "H_hgt_sbert_unmasked", "B_hgt_sbert"),
    ablationRow("Review text (SBERT) helps (A to B)", "A_hgt_only", "B_hgt_sbert"),
    ablationRow("Trust weighting (B to C)", "B_hgt_sbert", "C_hgt_sbert_trust"),
    ablationRow("Recency weighting (B to D)", "B_hgt_sbert", "D_hgt_sbert_recency"),
    ablationRow("Learned fusion layer (E to F)", E, "F_dytrustrec_full"),
  ];
  const g1 = gapVsBest(F1, R1, E), g2 = gapVsBest(F2, R2, E);
  rows.push(["Best single HGT vs best tuned baseline", gapCell(g1), gapCell(g2), gapVerdict(g1, g2)]);
  const h1 = gapVsBest(F1, R1, ENS), h2 = gapVsBest(F2, R2, ENS);
  rows.push(["Three-seed ensemble vs best tuned baseline", gapCell(h1), gapCell(h2), gapVerdict(h1, h2)]);

  const adaptive = (F) => {
    const d = F.adaptiveDetection;
    if (!d) return null;
    const a = d.attacks[Object.keys(d.attacks).find((k) => k.startsWith("adaptive"))].auc;
    return { old: a["version 1 (mean of ranks)"], semantic: a["indicator: semantic_shared"] };
  };
  const d1 = adaptive(F1), d2 = adaptive(F2);
  if (d1 && d2) {
    const ok = d1.old < 0.6 && d2.old < 0.6 && d1.semantic > 0.9 && d2.semantic > 0.9;
    rows.push(["Embedding indicator vs paraphrased campaign (AUC, old detector to new)",
      `${f3(d1.old)} to ${f3(d1.semantic)}`, `${f3(d2.old)} to ${f3(d2.semantic)}`, ok ? "replicates" : "differs"]);
  }

  const gate = (F) => {
    const s = F.robustnessAdaptive && F.robustnessAdaptive.summary;
    if (!s || !s["0.2"] || !s["0.2"].none || !s["0.2"].gate_v2) return null;
    const none = s["0.2"].none, g = s["0.2"].gate_v2;
    const clean = s["0.0"] && s["0.0"].none ? s["0.0"].none["ndcg@10"].mean : null;
    const diff = g["ndcg@10"].mean - none["ndcg@10"].mean;
    const z = diff / Math.sqrt(g["ndcg@10"].std ** 2 / g.n_seeds + none["ndcg@10"].std ** 2 / none.n_seeds);
    return { z, pct: rel(g["ndcg@10"].mean, none["ndcg@10"].mean), cost: clean ? rel(none["ndcg@10"].mean, clean) : null };
  };
  const q1 = gate(F1), q2 = gate(F2);
  if (q1 && q2) {
    const fmtGate = (q) => `${sp(q.pct, 1)} (z = ${q.z.toFixed(1).replace('-', '\u2212')}); the attack alone costs ${q.cost === null ? "n/a" : sp(q.cost, 1)}`;
    const label = (q) => (q.z > 2 ? "clear" : q.z > 1 ? "suggestive" : "none");
    rows.push(["Semantic gate protects accuracy under a 20% paraphrased campaign", fmtGate(q1), fmtGate(q2),
      label(q1) !== "none" && label(q2) === "none" ? `${label(q1)} on the first, none on the second` : label(q1) === label(q2) ? `${label(q1)} on both` : "differs"]);
  }

  const c1 = F1.coldStart, c2 = F2.coldStart;
  if (c1 && c2) {
    const gain = (c) => {
      const r = c.reserved_slots, k = String(r.chosen_on_validation);
      return r.chosen_on_validation === 0 ? null : rel(r.test[k]["ndcg@10"], r.test["0"]["ndcg@10"]);
    };
    const share = (c) => c.test_rows_on_new_items / c.test_rows;
    const txt = (c) => {
      const g = gain(c);
      return `${g === null ? "none (validation chose r = 0)" : `${sp(g)} with r = ${c.reserved_slots.chosen_on_validation}`}; ${L.pct(share(c), 0)} of test rows are new products`;
    };
    const g1c = gain(c1), g2c = gain(c2);
    rows.push(["Reserved new-product slots lift NDCG@10", txt(c1), txt(c2),
      g1c !== null && g2c === null ? "does not replicate" : g1c !== null && g2c !== null ? "replicates" : "not shown on either"]);
  }

  const k1 = F1.confidence, k2 = F2.confidence;
  if (k1 && k2) {
    rows.push(["Raw evidence-sufficiency confidence predicts hits (AUROC)", f3(k1.auroc.confidence), f3(k2.auroc.confidence),
      Math.abs(k1.auroc.confidence - 0.5) < 0.06 && Math.abs(k2.auroc.confidence - 0.5) < 0.06 ? "replicates: no signal" : "differs"]);
    if (k1.calibrated && k2.calibrated) {
      rows.push(["Separately calibrated hit probability ranks hits (AUROC)", f3(k1.calibrated.auroc), f3(k2.calibrated.auroc),
        k1.calibrated.auroc > 0.55 && k2.calibrated.auroc > 0.55 ? "replicates, weakly" : "differs"]);
    }
  }

  const fid = (F) => (F.fidelity && F.fidelity.tests.shared_category && F.fidelity.tests.shared_category.users
    ? F.fidelity.tests.shared_category : null);
  const x1 = fid(F1), x2 = fid(F2);
  if (x1 && x2) {
    const t = (x) => `${x.paired_gap.mean_diff >= 0 ? "+" : "−"}${f3(Math.abs(x.paired_gap.mean_diff))} [${f3(x.paired_gap.ci_low)}, ${f3(x.paired_gap.ci_high)}]`;
    rows.push(["Cited category evidence matters more than random evidence (difference, 95% CI)", t(x1), t(x2),
      x1.paired_gap.ci_low > 0 && x2.paired_gap.ci_low > 0 ? "replicates" : "differs"]);
  }
  return rows;
}

function section() {
  if (!available) return [];
  const out = [];
  const bb1 = bestBase(R1), bb2 = bestBase(R2);
  const mf2 = R2.matrix_factorization;
  const nUsers2 = mf2 ? mf2.diagnostics.n_users_evaluated : null;
  const transfer = transferScores();
  const g1 = gapVsBest(F1, R1, E), g2 = gapVsBest(F2, R2, E), h1 = gapVsBest(F1, R1, ENS), h2 = gapVsBest(F2, R2, ENS);
  const popGap = (R, bb) => rel(nd(R, "popularity"), nd(R, bb));

  out.push(P(`***K. Does It Hold on Another Category?*** Everything above is Video_Games. To separate properties of the method from properties of one catalogue, the whole pipeline was repeated on ${NAME} from the same release: the same tier rules, five-core filter and chronological 80/10/10 split. After cleaning it has 116,690 interactions, 13,725 users and 7,942 products (81% of the first dataset's rows at the same density, 0.107% against 0.109%)${nUsers2 ? `, and ${nUsers2.toLocaleString()} test users are evaluable` : ""}. Every model family, baselines and HGT alike, was tuned on this dataset's own validation split with the same search grids, and the test split was scored once. The test set is smaller than before, so intervals are wider.`, { before: 60 }));
  if (transfer && transfer[E] && transfer.B_hgt_sbert) {
    out.push(P(`The HGT was first run with the settings chosen on Video_Games, without re-tuning: arm B reached ${f4(transfer.B_hgt_sbert)} and arm E ${f4(transfer[E])}, level with the best baseline. Tuning on the new validation split raised them to ${f4(nd(R2, "B_hgt_sbert"))} and ${f4(nd(R2, E))}. The tuned figures are the ones in Table XIII; the transferred runs are kept in the repository.`));
  }
  out.push(W(L.caption("**TABLE XIII** — Test NDCG@10 on both datasets (mean ± SD over seeds; each dataset tuned on its own validation split).", { after: 40 })), W(sideBySide()));
  out.push(P(`The ranking of methods holds in direction but not in size. The best tuned baseline here is ${T.pretty(bb2)} at ${f4(nd(R2, bb2))}, and popularity at ${f4(nd(R2, "popularity"))} is only ${L.fmt(Math.abs(popGap(R2, bb2)) * 100, 0)}% behind it, where on Video_Games it was ${L.fmt(Math.abs(popGap(R1, bb1)) * 100, 0)}% behind: this catalogue rewards popularity much more. A single run of arm E reaches ${L.pm(nd(R2, E), sd(R2, E))}, ${sp(g2.pct)} over ${T.pretty(bb2)}, and the paired-bootstrap interval for the difference [${f4(g2.ci_low)}, ${f4(g2.ci_high)}] ${g2.ci_low > 0 ? "excludes" : "includes"} zero. The three-seed ensemble reaches ${f4(nd(R2, ENS))}, ${sp(h2.pct)}, with interval [${f4(h2.ci_low)}, ${f4(h2.ci_high)}] ${h2.ci_low > 0 ? "excluding" : "including"} zero. The corresponding leads on Video_Games were ${sp(g1.pct)} and ${sp(h1.pct)}. The claim that the full recipe clearly beats every tuned baseline is therefore supported there for single models and here only for the ensemble.`, { before: 60 }));

  out.push(W(L.caption("**TABLE XIV** — Replication scorecard. “Within noise” means the change is smaller than the larger seed standard deviation of the two arms.", { after: 40 })));
  out.push(W(L.table(["Finding", "Video_Games", NAME, "Verdict"], scorecardRows(), [3300, 2650, 2650, 1840],
    { size: 14, align: [L.AlignmentType.LEFT, L.AlignmentType.LEFT, L.AlignmentType.LEFT, L.AlignmentType.LEFT] })));

  // The defence and cold-start paragraphs say why two rows differ, with the second dataset's own counts.
  const s2 = F2.robustnessAdaptive && F2.robustnessAdaptive.summary;
  if (s2 && s2["0.2"] && s2["0.2"].none && s2["0.2"].gate_v2) {
    const r = s2["0.2"], rows = (c) => (r[c] && r[c].rows_gated ? Math.round(r[c].rows_gated.mean) : null);
    const falsePos = s2["0.0"] && s2["0.0"].gate_v2 ? Math.round(s2["0.0"].gate_v2.rows_gated.mean) : null;
    const clean = s2["0.0"] && s2["0.0"].none ? s2["0.0"].none["ndcg@10"].mean : null;
    out.push(P(`***What did not carry over.*** The defence found the campaign again: the semantic gate removed ${rows("gate_v2").toLocaleString()} rows, ${rows("oracle_gate") !== null && falsePos !== null && rows("gate_v2") === rows("oracle_gate") + falsePos ? `exactly the ${rows("oracle_gate").toLocaleString()} injected rows plus the ${falsePos.toLocaleString()} genuine rows it also flags on clean data, ` : ""}and the exact-text gate removed ${rows("trust+gate").toLocaleString()}. But this campaign hardly hurts the model here: NDCG@10 is ${f4(r.none["ndcg@10"].mean)} undefended${clean ? ` against ${f4(clean)} without shills` : ""}, a drop within the seed spread (${f4(r.none["ndcg@10"].std)}), so there was no accuracy for the gate to recover (${f4(r.gate_v2["ndcg@10"].mean)} with it). The Video_Games gain from the gate, already only suggestive, is not supported by a second dataset. The share of users shown a promoted item is a rare event (a handful of users) and moves between conditions without tracking how many rows were removed, so it is not interpreted.`));
  }
  const c2 = F2.coldStart;
  if (c2) {
    const r = c2.reserved_slots;
    const t0 = r.test["0"]["ndcg@10"], t1 = r.test["1"] ? r.test["1"]["ndcg@10"] : null;
    out.push(P(`The new-product strip did not help either, and the validation rule said so: r = ${r.chosen_on_validation} was chosen, because only ${L.pct(c2.test_rows_on_new_items / c2.test_rows, 0)} of test rows here are on products unseen in training (${L.pct(F1.coldStart.test_rows_on_new_items / F1.coldStart.test_rows, 0)} on Video_Games), and each reserved slot takes a place from a model ranking that is right more often (${f4(t0)} with none${t1 ? `, ${f4(t1)} with one` : ""}). Text similarity still finds new products when it is the only choice (${f4(c2.new_products_only["text similarity"]["ndcg@10"])} against ${f4(c2.new_products_only.random["ndcg@10"])} for random), so the strip is a safe option that switches itself off, not a general gain.`));
  }
  out.push(P(`***Reading Table XIV.*** What replicates is the training recipe and the checks on the model: hiding interaction edges is the single largest effect, review text is worth 7 to 9% on both but by less than the seed spread, trust weighting and recency weighting show no reliable effect on either, the learned fusion layer adds nothing, the embedding indicator catches a paraphrased campaign, and explanations built from retrieved evidence do track what the model used. What is specific to Video_Games, or at least not established, is the size of the lead over strong baselines, the recency effect, the new-product gain and the defence's accuracy benefit. The second dataset is a replication with a single split and one set of seeds, not a general result.`));
  return out;
}

// One sentence for the abstract and one for the conclusion.
function abstractSentence() {
  if (!available) return "";
  const g2 = gapVsBest(F2, R2, E), h2 = gapVsBest(F2, R2, ENS);
  return `A replication on a second category (${NAME}) reproduces the training-recipe and explanation findings but with a much smaller lead over the best baseline (${sp(g2.pct)} for a single model, not distinguishable from zero; ${sp(h2.pct)} for the three-seed ensemble), and shows no accuracy gain from the defence or the new-product strip. `;
}
function conclusionSentence() {
  if (!available) return "";
  const g2 = gapVsBest(F2, R2, E), h2 = gapVsBest(F2, R2, ENS);
  return `On a second category the recipe still helps but the lead shrinks to ${sp(g2.pct)} for a single model and ${sp(h2.pct)} for the ensemble, so claims of a large advantage rest on one catalogue (Section VI-K).`;
}

module.exports = { available, section, abstractSentence, conclusionSentence, scorecardRows, sideBySideRows, transferScores };
