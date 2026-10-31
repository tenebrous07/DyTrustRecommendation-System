#!/usr/bin/env python
"""Can the model rank products it has never seen?

The main evaluation scores only users and items that training saw, which leaves out
about 60% of test rows. This script measures the part that was left out for items:
test interactions of known users with products that did not exist in training.

Two settings:
  * mixed catalogue  - rank every product, new ones included, against all test items.
                       Compared with the same model without the content path, which
                       cannot score a new product at all.
  * new products only - rank only the products that are new, against the new products
                       the user actually interacted with. The "launch day" question.

The content path (``ContentMapper``) is fitted on training items only. Nothing from the
validation or test interactions is used to build it.

    python scripts/evaluate_cold_start.py --arm E_hgt_sbert_trust_recency
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

from src.evaluation.metrics import ndcg_at_k, recall_at_k  # noqa: E402
from src.experiments.context import ExperimentContext, paired_bootstrap  # noqa: E402
from src.features.sbert import SBERTEncoder  # noqa: E402
from src.recommendation.cold_items import ContentMapper  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = DIRS["results"] / "runs"
K = 10


def zscore(scores: np.ndarray) -> np.ndarray:
    spread = scores.std(axis=1, keepdims=True)
    return (scores - scores.mean(axis=1, keepdims=True)) / np.where(spread > 0, spread, 1.0)


def rank_metrics(scores: np.ndarray, truth: list[set[int]]) -> tuple[np.ndarray, np.ndarray]:
    top = np.argpartition(-scores, K - 1, axis=1)[:, :K]
    order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
    ranked = np.take_along_axis(top, order, axis=1)
    ndcg = np.array([ndcg_at_k(ranked[i], truth[i], K) for i in range(len(truth))])
    recall = np.array([recall_at_k(ranked[i], truth[i], K) for i in range(len(truth))])
    return ndcg, recall


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Cold-start item evaluation.")
    parser.add_argument("--arm", default="E_hgt_sbert_trust_recency")
    args = parser.parse_args(argv)

    ctx = ExperimentContext(device="cpu")
    vocab = ctx.vocab
    warm_n = vocab.n_items

    # Universe = every product the cleaned catalogue knows; new ones follow the warm ones.
    cold_ids = [i for i in ctx.products["item_id"].astype(str) if i not in vocab.items]
    universe = vocab.item_ids + cold_ids
    index = {i: n for n, i in enumerate(universe)}
    products = ctx.products.assign(item_id=ctx.products["item_id"].astype(str)).set_index("item_id")
    texts = []
    for item in universe:
        row = products.loc[item]
        description = row.get("product_description")
        description = description if isinstance(description, str) else ""
        title = row.get("product_title")
        texts.append(f"{title if isinstance(title, str) else ''}. {description[:1200]}")
    content = SBERTEncoder(ctx.model_config).encode(pd.Series(texts))
    content_warm, content_cold = content[:warm_n], content[warm_n:]

    members = []
    for path in sorted(RUNS_DIR.glob(f"{args.arm}__seed*.emb.npz")):
        with np.load(path) as saved:
            members.append((saved["user"].astype(np.float32), saved["item"].astype(np.float32)))
    if not members:
        raise SystemExit(f"No saved embeddings for {args.arm}.")

    norm = content / (np.linalg.norm(content, axis=1, keepdims=True) + 1e-12)

    def prepare(frame: pd.DataFrame):
        """Users, their truth in the universe index space, and the training items to mask."""
        frame = frame.assign(user_id=frame["user_id"].astype(str), item_id=frame["item_id"].astype(str))
        frame = frame[frame["user_id"].isin(vocab.users) & frame["item_id"].isin(index)]
        grouped = frame.groupby("user_id")["item_id"].apply(lambda s: {index[i] for i in s})
        users = list(grouped.index)
        user_idx = np.array([vocab.users[u] for u in users])
        truth = [grouped[u] for u in users]
        seen = [np.asarray(ctx.test_eval.train_positives.get(int(u), np.array([], dtype=int)), dtype=int)
                for u in user_idx]
        return user_idx, truth, seen

    def masked(matrix: np.ndarray, seen) -> np.ndarray:
        out = matrix.copy()
        for row, positives in enumerate(seen):
            out[row, positives] = -np.inf
        return out

    def components(user_idx: np.ndarray, seen):
        """z-scored model scores on warm items, and text similarity on every item."""
        warm = np.zeros((len(user_idx), warm_n), dtype=np.float32)
        for user_emb, item_emb in members:
            warm += zscore(user_emb[user_idx] @ item_emb.T) / len(members)
        profiles = np.stack([norm[s].mean(axis=0) if len(s) else np.zeros(norm.shape[1]) for s in seen])
        return warm, zscore((profiles @ norm.T).astype(np.float32))

    def blend(warm: np.ndarray, text: np.ndarray, beta: float, gamma: float) -> np.ndarray:
        scores = np.empty_like(text)
        scores[:, :warm_n] = warm + beta * text[:, :warm_n]
        scores[:, warm_n:] = gamma + beta * text[:, warm_n:]      # new products have no model score
        return scores

    # ---- choose the blend on VALIDATION, then score test once
    v_users, v_truth, v_seen = prepare(ctx.val)
    v_warm, v_text = components(v_users, v_seen)
    grid = []
    for beta in (0.25, 0.5, 1.0, 2.0):
        for gamma in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
            ndcg, _ = rank_metrics(masked(blend(v_warm, v_text, beta, gamma), v_seen), v_truth)
            grid.append((float(ndcg.mean()), beta, gamma))
    val_ndcg, beta, gamma = max(grid)

    def reserved(warm: np.ndarray, text: np.ndarray, seen, slots: int) -> np.ndarray:
        """Top-10 lists whose last ``slots`` places are the best NEW products by text similarity.

        A separate strip for new arrivals, the way shops run a "new for you" shelf, instead of
        asking a single score to rank a product with no history against one with thousands.
        """
        main = np.argsort(-masked(warm, seen), axis=1)[:, :K - slots]
        fresh = np.argsort(-text[:, warm_n:], axis=1)[:, :slots] + warm_n
        return np.concatenate([main, fresh], axis=1)

    def list_metrics(lists: np.ndarray, truth):
        ndcg = np.array([ndcg_at_k(lists[i], truth[i], K) for i in range(len(truth))])
        recall = np.array([recall_at_k(lists[i], truth[i], K) for i in range(len(truth))])
        return ndcg, recall

    slot_val = {r: float(list_metrics(reserved(v_warm, v_text, v_seen, r), v_truth)[0].mean()) for r in range(0, 6)}
    slots = max(slot_val, key=slot_val.get)

    users_idx, truth_all, seen = prepare(ctx.test)
    truth_cold = [{i for i in t if i >= warm_n} for t in truth_all]
    warm, text = components(users_idx, seen)

    no_cold = np.concatenate([warm, np.full((len(users_idx), len(cold_ids)), -np.inf, np.float32)], axis=1)
    blended = blend(warm, text, beta, gamma)
    popularity = np.tile(np.concatenate([ctx.test_eval.item_popularity, np.full(len(cold_ids), -np.inf)]),
                         (len(users_idx), 1)).astype(np.float32)
    random_scores = np.random.default_rng(0).random(text.shape).astype(np.float32)

    # the earlier idea, kept as a measured negative result: map text into the embedding space
    mapped_cold = np.zeros((len(users_idx), len(cold_ids)), dtype=np.float32)
    mapper_report = []
    for user_emb, item_emb in members:
        mapper = ContentMapper().fit(content_warm, item_emb)
        mapper.calibrate(user_emb, item_emb, mapper.out_of_fold(content_warm, item_emb))
        mapped_cold += zscore(mapper.rescale(user_emb[users_idx] @ mapper.embed(content_cold).T)) / len(members)
        mapper_report.append({"alpha": mapper.alpha_, "cv_r2": mapper.cv_r2_})

    results: dict = {
        "arm": args.arm, "members": len(members), "users": len(users_idx),
        "warm_items": warm_n, "new_items": len(cold_ids),
        "test_rows": int(sum(len(t) for t in truth_all)),
        "test_rows_on_new_items": int(sum(len(t) for t in truth_cold)),
        "blend": {"beta": beta, "gamma": gamma, "val_ndcg@10": val_ndcg,
                  "note": "chosen on validation only"},
        "embedding_mapping": mapper_report,
    }

    mixed = {}
    for name, matrix in (("blend: model + text (chosen on validation)", blended),
                         ("model only (cannot score new products)", no_cold),
                         ("text similarity only", masked_text := np.concatenate([text[:, :warm_n], text[:, warm_n:]], axis=1)),
                         ("popularity", popularity), ("random", random_scores)):
        ndcg, recall = rank_metrics(masked(matrix, seen), truth_all)
        mixed[name] = {"ndcg@10": float(ndcg.mean()), "recall@10": float(recall.mean()), "_u": ndcg}
    results["mixed_catalogue"] = {k: {a: b for a, b in v.items() if a != "_u"} for k, v in mixed.items()}
    key_blend, key_model = "blend: model + text (chosen on validation)", "model only (cannot score new products)"
    results["mixed_gain_over_model_only"] = paired_bootstrap(mixed[key_blend]["_u"], mixed[key_model]["_u"])

    # ---- reserved new-arrival slots (r chosen on validation)
    slot_rows = {}
    for r in range(0, 6):
        ndcg, recall = list_metrics(reserved(warm, text, seen, r), truth_all)
        slot_rows[r] = {"ndcg@10": float(ndcg.mean()), "recall@10": float(recall.mean()), "_u": ndcg}
    results["reserved_slots"] = {"chosen_on_validation": slots, "val_ndcg@10": slot_val,
                                 "test": {str(r): {a: b for a, b in v.items() if a != "_u"} for r, v in slot_rows.items()}}
    results["reserved_slots_gain_over_none"] = paired_bootstrap(slot_rows[slots]["_u"], slot_rows[0]["_u"])

    # ---- new products only
    keep = [i for i, t in enumerate(truth_cold) if t]
    cold_truth = [{c - warm_n for c in truth_cold[i]} for i in keep]
    only = {}
    for name, sub in (("text similarity", text[np.ix_(keep, np.arange(warm_n, len(universe)))]),
                      ("model embedding mapped from text", mapped_cold[keep]),
                      ("random", random_scores[np.ix_(keep, np.arange(warm_n, len(universe)))])):
        ndcg, recall = rank_metrics(sub, cold_truth)
        only[name] = {"ndcg@10": float(ndcg.mean()), "recall@10": float(recall.mean())}
    results["new_products_only"] = {"users": len(keep), "candidates": len(cold_ids), **only}

    (ctx.results_dir / "cold_start_eval.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    lines = ["# Cold-start items (test split)", "",
             f"Embeddings: {len(members)}-seed ensemble of `{args.arm}`. {results['users']:,} known users with a test "
             f"interaction, on a catalogue of {warm_n:,} warm and {len(cold_ids):,} new products; "
             f"{results['test_rows_on_new_items']:,} of {results['test_rows']:,} test rows are on new products.", "",
             f"Blend weights (text weight {beta}, new-product bonus {gamma}) were chosen on validation (NDCG@10 {val_ndcg:.4f}).", "",
             "## Reserved new-arrival slots (last r of the 10 places are the best new products by text)", "",
             f"r was chosen on validation: r = {slots}.", "",
             "| r | NDCG@10 | Recall@10 |", "| ---: | ---: | ---: |"]
    for r, v in results["reserved_slots"]["test"].items():
        lines.append(f"| {r}{' (chosen)' if int(r) == slots else ''} | {v['ndcg@10']:.4f} | {v['recall@10']:.4f} |")
    g2 = results["reserved_slots_gain_over_none"]
    lines += ["", f"Chosen r against r = 0, paired bootstrap over users: {g2['mean_diff']:+.4f} NDCG@10, "
                  f"95% CI [{g2['ci_low']:+.4f}, {g2['ci_high']:+.4f}].", "",
             "## Mixed catalogue, one score for everything (a single blended score; kept as a negative result)", "",
             "| Model | NDCG@10 | Recall@10 |", "| --- | ---: | ---: |"]
    for name, v in results["mixed_catalogue"].items():
        lines.append(f"| {name} | {v['ndcg@10']:.4f} | {v['recall@10']:.4f} |")
    g = results["mixed_gain_over_model_only"]
    lines += ["", f"Gain of the blend over the model alone, paired bootstrap over users: {g['mean_diff']:+.4f} NDCG@10, "
                  f"95% CI [{g['ci_low']:+.4f}, {g['ci_high']:+.4f}].", "",
              f"## New products only ({len(keep)} users, {len(cold_ids)} candidates)", "",
              "| Model | NDCG@10 | Recall@10 |", "| --- | ---: | ---: |"]
    for name, v in only.items():
        lines.append(f"| {name} | {v['ndcg@10']:.4f} | {v['recall@10']:.4f} |")
    lines += ["", f"Mapping text into the embedding space (the first idea) has a 5-fold CV R^2 of "
                  f"{np.mean([m['cv_r2'] for m in mapper_report]):.3f} and ranks new products worse than plain text similarity."]
    (ctx.tables_dir / "cold_start_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
