#!/usr/bin/env python
"""One real, un-curated recommendation with its explanation, counterfactual and confidence.

The user is chosen by a rule fixed in advance (the evaluable user with the longest
training history), not by looking at what the model recommends, and the result
reports whether the top item was actually a hit in the held-out test split. The
output feeds the live example printed on the architecture diagram and quoted in
the paper, so those cannot drift from what the model really does.

    python scripts/export_model.py && python scripts/explain_example.py
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import active_dirs  # noqa: E402

DIRS = active_dirs()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.experiments.context import ExperimentContext  # noqa: E402
from src.explainability.counterfactual import CounterfactualAnalyzer  # noqa: E402
from src.explainability.explanations import Explainer  # noqa: E402
from src.explainability.graph_paths import EvidenceRetriever  # noqa: E402
from src.features.product_features import extract_category_levels  # noqa: E402
from src.recommendation.confidence import ConfidenceEstimator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = DIRS["results"] / "runs"
TOP_K = 5


def member_embeddings(arm: str) -> list[tuple[np.ndarray, np.ndarray]]:
    members = []
    for path in sorted(RUNS_DIR.glob(f"{arm}__seed*.emb.npz")):
        with np.load(path) as data:
            members.append((data["user"].astype(np.float32), data["item"].astype(np.float32)))
    return members


def zscore(scores: np.ndarray) -> np.ndarray:
    spread = scores.std()
    return (scores - scores.mean()) / (spread if spread > 0 else 1.0)


def main() -> int:
    ctx = ExperimentContext(device="cpu")
    exported = np.load(DIRS["checkpoints"] / "dytrustrec_embeddings.npz", allow_pickle=True)
    user_emb, item_emb = exported["user_emb"], exported["item_emb"]
    if [str(u) for u in exported["user_ids"]] != ctx.vocab.user_ids:
        raise SystemExit("Exported embeddings do not match the training vocabulary; re-run export_model.py.")

    trust_table = ctx.tables_dir / "trust_scores_behavioural.csv"
    trust_map = (dict(zip(pd.read_csv(trust_table)["entity_id"].astype(str),
                          pd.read_csv(trust_table)["trust_score"])) if trust_table.is_file() else {})
    counts = ctx.train.groupby("user_id").size().to_dict()

    evaluable = [ctx.vocab.user_ids[int(u)] for u in ctx.test_eval.eval_users]
    user_id = max(evaluable, key=lambda u: (counts[u], u))
    user_index = ctx.vocab.users[user_id]

    scores = item_emb @ user_emb[user_index]
    seen = ctx.test_eval.train_positives.get(user_index, np.array([], dtype=int))
    scores[seen] = -np.inf
    order = np.argsort(-scores)
    top = order[:TOP_K]
    truth = ctx.test_eval.ground_truth[user_index]

    products = extract_category_levels(
        ctx.products, node_level=str(ctx.model_config.get("graph.category_level", "l2")))
    retriever = EvidenceRetriever(ctx.train, products, ctx.vocab, trust_map=trust_map)
    titles = retriever.item_title
    best = ctx.vocab.item_ids[int(top[0])]
    explanation = Explainer(retriever).explain(user_id, best)

    arm = str(ctx.experiment_config.get("serve_arm", "E_hgt_sbert_trust_recency"))
    members = member_embeddings(arm)
    stability = None
    if len(members) >= 3:
        stack = np.stack([zscore(ue[user_index] @ ie.T)[top] for ue, ie in members])
        stability = ConfidenceEstimator.ensemble_stability(stack)
    estimator = ConfidenceEstimator(ctx.model_config, user_interaction_counts=counts, trust_map=trust_map)
    confidence = estimator.estimate(None, user_id, user_index, top,
                                    [ctx.vocab.item_ids[int(i)] for i in top], stability=stability)[0]

    analyzer = CounterfactualAnalyzer(user_emb, item_emb, ctx.vocab, item_categories=ctx.item_categories,
                                      masked_items=lambda _u, idx=np.asarray(seen): idx)
    near_miss = ctx.vocab.item_ids[int(order[TOP_K])]
    counterfactual = analyzer.analyse(user_id, near_miss, k=TOP_K)

    title = titles.get(best, best)
    text = (f"Live example (real run, un-curated: the evaluable user with the longest history): "
            f"'{title}' recommended to a user with {counts[user_id]} training interactions and trust "
            f"{trust_map.get(user_id, float('nan')):.3f}. {explanation.text.split('because', 1)[-1].strip().capitalize()} "
            f"Held-out outcome: the top item {'was' if int(top[0]) in truth else 'was not'} a hit in the test split.")
    lines = textwrap.wrap(text, 235)[:2]

    summary = (f"the user with the longest training history among evaluable users ({counts[user_id]} interactions, "
               f"behavioural trust {trust_map.get(user_id, float('nan')):.3f}) is recommended '{title}'. "
               f"{explanation.text} Confidence {confidence.confidence:.3f} ({confidence.reason}). "
               f"For the first item outside the top {TOP_K} ('{titles.get(near_miss, near_miss)}'), the counterfactual "
               f"module reports: {counterfactual.text} The top item was "
               f"{'a hit' if int(top[0]) in truth else 'not a hit'} in the held-out test split.")

    result = {
        "user_id": user_id, "n_train": int(counts[user_id]),
        "trust": float(trust_map.get(user_id, float("nan"))),
        "top": [{"item_id": ctx.vocab.item_ids[int(i)], "title": titles.get(ctx.vocab.item_ids[int(i)], ""),
                 "score": float(scores[i]), "hit": bool(int(i) in truth)} for i in top],
        "explanation": explanation.to_dict(), "confidence": confidence.to_dict(),
        "counterfactual": counterfactual.to_dict(), "near_miss": near_miss,
        "ensemble_members": len(members), "lines": lines, "summary": summary,
    }
    out = ROOT / "docs" / "diagrams" / "live_example.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
