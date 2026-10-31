// Result tables and figures, built from the project's JSON output (never typed in by hand).

const fs = require("fs");
const path = require("path");
const L = require("./lib");
const A = L.AlignmentType;

const NAMES = {
  popularity: "Popularity", item_knn_cf: "Item-kNN", matrix_factorization: "BPR-MF", lightgcn: "LightGCN", sasrec: "SASRec",
  A_hgt_only: "A: HGT only", B_hgt_sbert: "B: + SBERT", C_hgt_sbert_trust: "C: B + trust",
  D_hgt_sbert_recency: "D: B + recency", E_hgt_sbert_trust_recency: "E: B + trust + recency",
  F_dytrustrec_full: "F: DyTrustRec (full)", G_dytrustrec_cola_trust: "G: F with CoLA trust",
  H_hgt_sbert_unmasked: "H: B, no edge masking",
  I_dytrustrec_cosine: "I: E + cosine decay",
};
function pretty(model) {
  const [base, ens] = model.split("+ens");
  const name = NAMES[base] || base;
  return ens ? `${name}, ${ens}-seed ensemble` : name;
}

function byModel(results) { return Object.fromEntries(results.map((r) => [r.model, r])); }

// Table: model comparison across all accuracy and beyond-accuracy metrics.
function comparisonTable(results) {
  const rows = results.map((r) => {
    const m = r.metrics, s = r.metrics_std;
    return [pretty(r.model), r.n_seeds,
      L.pm(m["ndcg@10"], s["ndcg@10"]), L.pm(m["recall@10"], s["recall@10"]),
      L.fmt(m["map@10"]), L.fmt(m["hit_rate@10"]), L.fmt(m["coverage@10"], 3), L.fmt(m["novelty@10"], 2)];
  });
  return L.table(["Model", "Seeds", "NDCG@10", "Recall@10", "MAP@10", "Hit@10", "Cov.@10", "Nov.@10"], rows,
    [2950, 520, 1500, 1500, 780, 780, 780, 780], { size: 15, stripe: true });
}

// Table: step-by-step ablation deltas with a paired-bootstrap interval where one exists.
const STEPS = [
  ["A", "B", "A_hgt_only", "B_hgt_sbert", "add SBERT semantics"],
  ["B", "H", "B_hgt_sbert", "H_hgt_sbert_unmasked", "remove edge masking"],
  ["B", "C", "B_hgt_sbert", "C_hgt_sbert_trust", "add behavioural trust weighting"],
  ["B", "D", "B_hgt_sbert", "D_hgt_sbert_recency", "add recency weighting"],
  ["D", "E", "D_hgt_sbert_recency", "E_hgt_sbert_trust_recency", "then add trust weighting"],
  ["E", "F", "E_hgt_sbert_trust_recency", "F_dytrustrec_full", "then add the learned fusion layer"],
  ["E", "I", "E_hgt_sbert_trust_recency", "I_dytrustrec_cosine", "add cosine learning-rate decay"],
  ["F", "G", "F_dytrustrec_full", "G_dytrustrec_cola_trust", "swap behavioural trust for CoLA trust"],
];
function ablationRows(results) {
  const R = byModel(results);
  return STEPS.filter(([, , a, b]) => R[a] && R[b]).map(([x, y, a, b, what]) => {
    const ma = R[a].metrics["ndcg@10"], mb = R[b].metrics["ndcg@10"];
    const sa = R[a].metrics_std["ndcg@10"], sb = R[b].metrics_std["ndcg@10"];
    const delta = (mb - ma) / ma;
    // a change is flagged only when it exceeds the larger of the two seed standard deviations
    const clear = Math.abs(mb - ma) > Math.max(sa, sb);
    return [`${x}→${y}`, what, L.fmt(ma), L.fmt(mb), L.signedPct(delta), clear ? "yes" : "within noise"];
  });
}
function ablationTable(results) {
  return L.table(["Step", "Change", "NDCG@10 before", "after", "Δ", "> seed SD?"], ablationRows(results),
    [650, 2750, 1000, 700, 750, 1050], { size: 15, align: [A.LEFT, A.LEFT, A.RIGHT, A.RIGHT, A.RIGHT, A.CENTER] });
}

// Table: significance of the headline model against the strongest baseline.
function significanceRows(S) {
  if (!S || !S.comparisons) return [];
  const rows = [];
  for (const [model, tests] of Object.entries(S.comparisons)) {
    for (const [vs, t] of Object.entries(tests)) {
      rows.push([pretty(model), vs.replace("vs_", ""), L.fmt(t.mean_diff, 4), `[${L.fmt(t.ci_low, 4)}, ${L.fmt(t.ci_high, 4)}]`,
        L.fmt(t.p_a_better, 3)]);
    }
  }
  return rows;
}

// Table: synthetic detection AUCs.
function detectionTable(F) {
  const p = path.join(F.root, "reports", "tables", "synthetic_anomaly_report.json");
  if (!fs.existsSync(p)) return null;
  const d = JSON.parse(fs.readFileSync(p, "utf8")).detection;
  const rows = Object.entries(d).sort((a, b) => b[1].roc_auc - a[1].roc_auc)
    .map(([name, v]) => [name.replace("trivial rule: ", "Rule: "), L.fmt(v.roc_auc), L.fmt(v.average_precision), L.fmt(v.precision_at_100)]);
  return L.table(["Detector", "ROC-AUC", "Avg. prec.", "P@100"], rows, [2400, 800, 800, 800], { size: 15 });
}

// Table: HGT robustness to shills.
function robustnessTable(F, key = "robustness") {
  if (!F[key]) return null;
  const order = ["none", "cola", "trust", "trust+gate", "gate_v2", "oracle", "oracle_gate", "inverted"];
  const rows = [];
  for (const [rate, conds] of Object.entries(F[key].summary)) {
    for (const c of order) {
      if (!conds[c]) continue;
      const v = conds[c];
      rows.push([`${(parseFloat(rate) * 100).toFixed(0)}%`, c, v.n_seeds, L.pm(v["ndcg@10"].mean, v["ndcg@10"].std),
        `${(100 * v.users_with_target_in_top10.mean).toFixed(2)}%`]);
    }
  }
  return L.table(["Shills", "Defence", "n", "NDCG@10", "Shown promoted"], rows,
    [650, 1000, 400, 1550, 1300], { size: 15, align: [A.LEFT, A.LEFT, A.CENTER, A.RIGHT, A.RIGHT] });
}

// Table: hyper-parameter search, validation only.
function tuningTable(F) {
  const rows = F.tuned.records.filter((r) => r.model === "hgt" || r.model === "hgt_recency" || r.model === "hgt_trust")
    .sort((a, b) => b["val_ndcg@10"] - a["val_ndcg@10"]).slice(0, 14)
    .map((r) => [r.model === "hgt" ? "structure" : r.model.replace("hgt_", "weight: "), r.name, L.fmt(r["val_ndcg@10"]), r.best_epoch]);
  return L.table(["Search", "Setting", "Val NDCG@10", "Best epoch"], rows, [1500, 1400, 1100, 900],
    { size: 15, align: [A.LEFT, A.LEFT, A.RIGHT, A.RIGHT] });
}

// Table: detection AUC against the standard and the adaptive attack.
function adaptiveDetectionTable(F) {
  const d = F.adaptiveDetection;
  if (!d) return null;
  const attacks = Object.keys(d.attacks);
  const names = Object.keys(d.attacks[attacks[0]].auc);
  const rows = names.map((n) => [n.replace("indicator: ", "  ").replace("version 1", "Detector v1").replace("version 2", "Detector v2"),
    ...attacks.map((a) => L.fmt(d.attacks[a].auc[n], 3))]);
  return L.table(["Detector or indicator", "Standard", "Adaptive"], rows, [2900, 1000, 1000],
    { size: 15, align: [A.LEFT, A.RIGHT, A.RIGHT] });
}

// Table: cold-start reserved slots.
function coldStartTable(F) {
  const c = F.coldStart;
  if (!c) return null;
  const rows = Object.entries(c.reserved_slots.test).map(([r, v]) => [
    `${r}${Number(r) === c.reserved_slots.chosen_on_validation ? " (chosen on validation)" : ""}`, L.fmt(v["ndcg@10"]), L.fmt(v["recall@10"])]);
  return L.table(["New-arrival slots r of 10", "NDCG@10", "Recall@10"], rows, [2400, 1100, 1100],
    { size: 15, align: [A.LEFT, A.RIGHT, A.RIGHT] });
}

// Table: explanation fidelity.
function fidelityTable(F) {
  const f = F.fidelity;
  if (!f) return null;
  const rows = Object.entries(f.tests).filter(([, t]) => t.users).map(([k, t]) => [
    k.replace("shared_", ""), t.users, t.mean_rows_removed.toFixed(1), L.fmt(t.mean_score_drop_cited, 3),
    L.fmt(t.mean_score_drop_random, 3), `${(100 * t.share_cited_hurts_more).toFixed(0)}%`,
    `${t.paired_gap.mean_diff >= 0 ? "+" : "\u2212"}${Math.abs(t.paired_gap.mean_diff).toFixed(3)} [${L.fmt(t.paired_gap.ci_low, 3)}, ${L.fmt(t.paired_gap.ci_high, 3)}]`]);
  return L.table(["Cited", "n", "Rows", "Drop, cited", "Drop, random", "Worse", "Difference (95% CI)"], rows,
    [620, 380, 480, 680, 720, 520, 1560], { size: 13 });
}

module.exports = { adaptiveDetectionTable, coldStartTable, fidelityTable, pretty, byModel, comparisonTable, ablationRows, ablationTable, significanceRows,
  detectionTable, robustnessTable, tuningTable };
