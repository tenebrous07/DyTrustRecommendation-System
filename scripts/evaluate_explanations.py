#!/usr/bin/env python
"""Are the explanations faithful to the model?

An explanation such as "recommended because 96 of the user's 117 past items are in
'legacy systems'" is built from retrieved facts, so it is true of the data. Whether the
MODEL actually used those facts is a separate question, and the only way to answer it is to
take the cited interactions away and see what the model does.

For each sampled user the model's top recommendation is explained; the user's own past
interactions that the explanation cites (their items in the cited category, or from the
cited brand) are removed from the graph, the trained model is run again, and the drop in that
recommendation's score is measured. The control removes the same number of the user's OTHER
interactions at random. If the explanation points at what the model relies on, removing the
cited interactions should hurt the recommendation more than removing as many arbitrary ones.

Needs the checkpoint of the arm in ``export_arm`` (written by ``run_experiments.py``).

    python scripts/evaluate_explanations.py --users 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import active_dirs  # noqa: E402

DIRS = active_dirs()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.experiments.context import ExperimentContext, paired_bootstrap  # noqa: E402
from src.explainability.graph_paths import EvidenceRetriever  # noqa: E402
from src.features.product_features import extract_category_levels  # noqa: E402
from src.models.hgt import HGTRecommender, configure_backend_for_device  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_model(ctx: ExperimentContext, flags: dict, state: dict) -> HGTRecommender:
    """Rebuild the trained HGT and load its weights; no training happens here."""
    rec = HGTRecommender(ctx.vocab.n_users, ctx.vocab.n_items, bundle=ctx.bundle(sbert=bool(flags["sbert"])),
                         model_config=ctx.model_config, training_config=ctx.training_config,
                         device=str(ctx.device), use_fusion=bool(flags["fusion"]), epochs=0)
    configure_backend_for_device(rec.device)
    rec.model = rec._build()
    rec.model.load_state_dict(state)
    rec.model.eval()
    data = rec.bundle.data.to(rec.device)
    rec.x_dict = {t: data[t].x for t in data.node_types if "x" in data[t]}
    rec._full_edges = {e: data[e].edge_index for e in data.edge_types}
    rec._interaction_edges = [e for e in rec._full_edges
                              if e[1].replace("rev_", "") in ("purchased", "wrote", "about")]
    return rec


@torch.no_grad()
def embeddings_without(rec: HGTRecommender, n_rows: int, removed: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    keep = torch.ones(n_rows, dtype=torch.bool, device=rec.device)
    if len(removed):
        keep[torch.as_tensor(removed, device=rec.device)] = False
    out = rec.model(rec.x_dict, rec._select_edges(keep), rec.sbert_dim)
    return out["user"], out["product"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Explanation fidelity by interaction removal.")
    parser.add_argument("--users", type=int, default=300)
    args = parser.parse_args(argv)

    ctx = ExperimentContext()
    state = torch.load(DIRS["checkpoints"] / "best_model.pt", map_location=ctx.device, weights_only=False)
    rec = load_model(ctx, state["flags"], state["state_dict"])
    n_rows = len(ctx.train)

    products = extract_category_levels(
        ctx.products, node_level=str(ctx.model_config.get("graph.category_level", "l2")))
    retriever = EvidenceRetriever(ctx.train, products, ctx.vocab)
    rows_of_user = ctx.train_idx.groupby("user_idx").indices                      # user -> row positions
    item_ids = ctx.train_idx["item_id"].astype(str).to_numpy()

    user_emb, item_emb = embeddings_without(rec, n_rows, np.array([], dtype=int))
    rng = np.random.default_rng(0)
    candidates = rng.permutation(ctx.test_eval.eval_users)

    results = {"checkpoint": f"{state['ablation']} seed {state['seed']}", "tests": {}}
    for kind, lookup, key in (("shared_category", retriever.item_category, "category"),
                              ("shared_brand", retriever.item_brand, "brand")):
        records = []
        for user in candidates:
            if len(records) >= args.users:
                break
            user = int(user)
            history = rows_of_user.get(user)
            if history is None or len(history) < 6:
                continue
            seen = ctx.test_eval.train_positives.get(user, np.array([], dtype=int))
            scores = (item_emb @ user_emb[user]).cpu().numpy()
            scores[seen] = -np.inf
            best = int(np.argmax(scores))
            item_id = ctx.vocab.item_ids[best]
            cited = [e for e in retriever.retrieve(ctx.vocab.user_ids[user], item_id) if e.kind == kind]
            if not cited:
                continue
            value = cited[0].support[key]
            mine = np.array([r for r in history if lookup.get(item_ids[r]) == value], dtype=int)
            others = np.array([r for r in history if lookup.get(item_ids[r]) != value], dtype=int)
            # Only users where the cited rows are a minority, so a same-sized random control exists.
            if len(mine) < 2 or len(mine) > len(others):
                continue

            def fall(removed: np.ndarray) -> tuple[float, int]:
                u_emb, i_emb = embeddings_without(rec, n_rows, removed)
                new = (i_emb @ u_emb[user]).cpu().numpy()
                new[seen] = -np.inf
                spread = float(np.std(scores[np.isfinite(scores)]))
                return (float(scores[best]) - float(new[best])) / spread, int((new > new[best]).sum())

            drop_cited, rank_cited = fall(mine)
            controls = [fall(rng.choice(others, size=len(mine), replace=False)) for _ in range(3)]
            records.append({"user": user, "removed": int(len(mine)), "history": int(len(history)),
                            "drop_cited": drop_cited, "rank_cited": rank_cited,
                            "drop_random": float(np.mean([c[0] for c in controls])),
                            "rank_random": float(np.mean([c[1] for c in controls]))})
        frame = pd.DataFrame(records)
        if frame.empty:
            results["tests"][kind] = {"users": 0}
            continue
        gap = paired_bootstrap(frame["drop_cited"].to_numpy(), frame["drop_random"].to_numpy())
        results["tests"][kind] = {
            "users": int(len(frame)), "mean_rows_removed": float(frame["removed"].mean()),
            "mean_score_drop_cited": float(frame["drop_cited"].mean()),
            "mean_score_drop_random": float(frame["drop_random"].mean()),
            "mean_rank_after_cited": float(frame["rank_cited"].mean()),
            "mean_rank_after_random": float(frame["rank_random"].mean()),
            "share_cited_hurts_more": float((frame["drop_cited"] > frame["drop_random"]).mean()),
            "paired_gap": gap}

    (ctx.results_dir / "explanation_fidelity.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = ["# Explanation fidelity by interaction removal (test users, top-1 recommendation)", "",
             f"Model: `{results['checkpoint']}`. Score drop is in units of the user's own score spread; a larger drop means "
             "the recommendation depended more on what was removed.", "",
             "| Cited evidence | users | rows removed | score drop, cited | score drop, same number of random rows | "
             "cited hurts more | difference, 95% CI |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for kind, t in results["tests"].items():
        if not t.get("users"):
            lines.append(f"| {kind} | 0 | | | | | |")
            continue
        g = t["paired_gap"]
        lines.append(f"| {kind} | {t['users']} | {t['mean_rows_removed']:.1f} | {t['mean_score_drop_cited']:.3f} | "
                     f"{t['mean_score_drop_random']:.3f} | {100 * t['share_cited_hurts_more']:.0f}% | "
                     f"{g['mean_diff']:+.3f} [{g['ci_low']:+.3f}, {g['ci_high']:+.3f}] |")
    (ctx.tables_dir / "explanation_fidelity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
