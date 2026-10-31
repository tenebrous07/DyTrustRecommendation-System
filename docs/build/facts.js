// Reads the numbers the documents quote from the project's own result files,
// so a re-run of the experiments changes the documents, not a hand-edited copy.

const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..", "..");
const readJson = (p) => (fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : null);

function bestOf(records, family) {
  const rows = records.filter((r) => r.model === family);
  return rows.length ? rows.reduce((a, b) => (b["val_ndcg@10"] > a["val_ndcg@10"] ? b : a)) : null;
}

// ``slug`` selects another dataset's result directory (reports/results_<slug>); empty = Video_Games.
function loadFacts(slug = "") {
  const dir = slug ? `results_${slug}` : "results";
  const tuningDir = path.join(ROOT, "reports", dir, "tuning");
  const records = fs.existsSync(tuningDir)
    ? fs.readdirSync(tuningDir).filter((f) => f.endsWith(".json")).map((f) => readJson(path.join(tuningDir, f)))
    : [];
  const hgt = bestOf(records, "hgt");
  const rec = bestOf(records, "hgt_recency");
  const trust = bestOf(records, "hgt_trust");
  const o = (hgt && hgt.overrides) || {};
  const keep = o["training.edge_mask.keep_prob"] ?? 0.8;

  const results = readJson(path.join(ROOT, "reports", dir, "experiment_results.json"));
  const significance = readJson(path.join(ROOT, "reports", dir, "significance.json"));
  const refit = readJson(path.join(ROOT, "reports", dir, "experiment_results_refit.json"));
  const significanceRefit = readJson(path.join(ROOT, "reports", dir, "significance_refit.json"));
  const robustness = readJson(path.join(ROOT, "reports", dir, "robustness_hgt.json"));
  const confidence = readJson(path.join(ROOT, "reports", dir, "confidence_eval.json"));
  const live = slug ? null : readJson(path.join(ROOT, "docs", "diagrams", "live_example.json"));
  const R = (name) => readJson(path.join(ROOT, "reports", dir, name));
  const coldStart = R("cold_start_eval.json");
  const adaptiveDetection = R("adaptive_attack_detection.json");
  const robustnessAdaptive = R("robustness_hgt_adaptive.json");
  const fidelity = R("explanation_fidelity.json");
  const serving = R("serving_benchmark.json");

  return {
    root: ROOT,
    graph: { nodes: "138,480", edges: "718,506" },
    tuned: {
      keep, hiddenPct: Math.round((1 - keep) * 100),
      dropout: o["hgt.dropout"] ?? 0.2,
      lr: o["optimizer.learning_rate"] ?? 0.005,
      lambda: rec ? rec.recency_lambda : 0.5,
      alpha: trust ? trust.trust_alpha : 1.0,
      hgtVal: hgt ? hgt["val_ndcg@10"] : null,
      records,
    },
    results, significance, refit, significanceRefit, robustness, confidence, live,
    coldStart, adaptiveDetection, robustnessAdaptive, fidelity, serving,
  };
}

module.exports = { loadFacts, readJson, ROOT };
