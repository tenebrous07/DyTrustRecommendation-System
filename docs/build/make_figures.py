#!/usr/bin/env python
"""Figures for the paper, drawn from the saved run histories and tuning records.

    python docs/build/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / "reports" / "results" / "runs"
TUNING = ROOT / "reports" / "results" / "tuning"

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "figure.dpi": 200})
BLUE, ORANGE, GREY = "#2F5F9E", "#C77D2E", "#777777"


def histories(prefix: str, directory: Path = RUNS) -> list[list[dict]]:
    out = []
    for path in sorted(directory.glob(f"{prefix}__seed*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("history"):
            out.append(record["history"])
    return out


def curve(runs: list[list[dict]], key: str):
    """Mean and standard deviation across runs at the epochs every run has measured."""
    epochs = sorted({h["epoch"] for run in runs for h in run if key in h})
    table = {e: [h[key] for run in runs for h in run if h["epoch"] == e and key in h] for e in epochs}
    xs = [e for e in epochs if len(table[e]) == len(runs)]
    mean = np.array([np.mean(table[e]) for e in xs])
    std = np.array([np.std(table[e]) for e in xs])
    return np.array(xs), mean, std


def dynamics() -> None:
    # Arm B's curves come from a re-run made for this figure (runs_curves/); the
    # arm-B runs behind the tables predate curve logging.
    masked = histories("B_hgt_sbert", ROOT / "reports" / "results" / "runs_curves") or histories("B_hgt_sbert")
    unmasked = histories("H_hgt_sbert_unmasked")
    if not masked or not unmasked:
        print("dynamics: run histories missing, skipped")
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.3))
    for runs, label, colour in ((masked, "edge-masked (B)", BLUE), (unmasked, "no masking (H)", ORANGE)):
        x, m, s = curve(runs, "val_ndcg@10")
        axes[0].plot(x, m, color=colour, label=label, lw=1.2)
        axes[0].fill_between(x, m - s, m + s, color=colour, alpha=0.18, lw=0)
        x, m, s = curve(runs, "train_loss")
        axes[1].plot(x, m, color=colour, label=label, lw=1.2)
        axes[1].fill_between(x, m - s, m + s, color=colour, alpha=0.18, lw=0)
    axes[0].set(xlabel="epoch", ylabel="validation NDCG@10", title="Held-out ranking quality")
    axes[1].set(xlabel="epoch", ylabel="training BPR loss", title="Training loss")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "fig_dynamics.png")
    plt.close(fig)


def tuning() -> None:
    records = [json.loads(p.read_text(encoding="utf-8")) for p in TUNING.glob("*.json")]
    lam = sorted((r["recency_lambda"], r["val_ndcg@10"]) for r in records if r["model"] == "hgt_recency")
    keep = sorted((r["overrides"].get("training.edge_mask.keep_prob", 0.8), r["val_ndcg@10"])
                  for r in records if r["model"] == "hgt" and set(r["overrides"]) <= {"training.edge_mask.keep_prob"})
    drop = sorted((r["overrides"].get("hgt.dropout", 0.2), r["val_ndcg@10"])
                  for r in records if r["model"] == "hgt" and set(r["overrides"]) <= {"hgt.dropout"})
    if not (lam and keep):
        print("tuning: records missing, skipped")
        return
    base = next((r["val_ndcg@10"] for r in records if r["model"] == "hgt" and r["name"] == "combined"), None)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.0))
    for ax, data, xlabel, title in ((axes[0], keep, "share of interaction edges kept visible", "Edge masking"),
                                    (axes[1], drop, "dropout", "Dropout"),
                                    (axes[2], lam, "recency decay λ (per year)", "Recency weighting")):
        x, y = zip(*data)
        ax.plot(x, y, "o-", color=BLUE, ms=3, lw=1)
        ax.set(xlabel=xlabel, title=title)
        if ax is axes[2]:
            ax.set_xscale("log")
            if base:
                ax.axhline(base, color=GREY, ls="--", lw=0.8)
                ax.text(x[0], base, " unweighted", va="bottom", color=GREY, fontsize=7)
    axes[0].set_ylabel("validation NDCG@10")
    fig.tight_layout()
    fig.savefig(HERE / "fig_tuning.png")
    plt.close(fig)


if __name__ == "__main__":
    dynamics()
    tuning()
    print("figures written to", HERE)
