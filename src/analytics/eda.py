"""Exploratory Data Analysis (Module 3).

Produces the descriptive picture of the corpus that every later modelling
decision is justified against: who the users are, how popularity is
distributed, how ratings and text behave, and how all of that moves over time.

Two rules shape the code:

* **Figures are artefacts, not side effects.** Every figure is written to
  ``reports/figures/`` by a function that also returns the table behind it, so
  a number in the report can always be traced to the frame that produced it.
* **Absent data is reported, never imputed.** A panel whose source column is
  missing is skipped with a note in :attr:`EDAResult.notes`, rather than being
  drawn from a default that would read as a measurement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..features.product_features import (  # noqa: E402
    build_product_profiles,
    extract_category_levels,
    long_tail_summary,
)
from ..features.user_features import build_user_profiles  # noqa: E402
from ..utils.config import Config, resolve_path  # noqa: E402
from ..utils.logging import get_logger, log_stage  # noqa: E402

__all__ = ["EDAResult", "ExploratoryAnalysis"]

logger = get_logger(__name__)

_FALLBACK_STYLES = ("seaborn-v0_8-darkgrid", "seaborn-darkgrid", "ggplot", "default")


def _resolve_style(preferred: str) -> str:
    """Pick a matplotlib style that exists in this matplotlib version."""
    for candidate in (preferred, *_FALLBACK_STYLES):
        if candidate in plt.style.available or candidate == "default":
            return candidate
    return "default"


@dataclass
class EDAResult:
    """Everything Phase 2's EDA produced."""

    summary: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: dict[str, Path] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = ["# Exploratory Data Analysis", "", "## Headline figures", "",
                 "| Metric | Value |", "| --- | ---: |"]
        for key, value in self.summary.items():
            if isinstance(value, float):
                rendered = f"{value:,.4f}" if abs(value) < 1000 else f"{value:,.1f}"
            elif isinstance(value, int):
                rendered = f"{value:,}"
            else:
                rendered = str(value)
            lines.append(f"| {key.replace('_', ' ')} | {rendered} |")

        if self.figures:
            lines += ["", "## Figures", "", "| Figure | Path |", "| --- | --- |"]
            for name, path in sorted(self.figures.items()):
                lines.append(f"| {name.replace('_', ' ')} | `{path}` |")

        if self.notes:
            lines += ["", "## Notes and caveats", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


class ExploratoryAnalysis:
    """Descriptive analysis of a cleaned interaction corpus."""

    def __init__(
        self,
        interactions: pd.DataFrame,
        products: pd.DataFrame | None,
        config: Config,
    ) -> None:
        if interactions.empty:
            raise ValueError("EDA requires a non-empty interactions frame.")
        self.interactions = interactions
        self.config = config
        self.cfg: Config = config.require("eda")

        self.products = None
        if products is not None and not products.empty:
            level = str(self.cfg.get("category_level", "l2"))
            self.products = extract_category_levels(products, node_level=level)

        self.figure_dir = resolve_path(config.get("paths.figures", "reports/figures"))
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = int(self.cfg.get("figure_dpi", 150))
        self.style = _resolve_style(str(self.cfg.get("figure_style", "default")))
        self._interactive = set(self.cfg.get("interactive_figures", []) or [])
        self.result = EDAResult()

        self._user_profiles: pd.DataFrame | None = None
        self._product_profiles: pd.DataFrame | None = None

    # -- cached profiles ---------------------------------------------------
    @property
    def user_profiles(self) -> pd.DataFrame:
        if self._user_profiles is None:
            self._user_profiles = build_user_profiles(self.interactions)
        return self._user_profiles

    @property
    def product_profiles(self) -> pd.DataFrame:
        if self._product_profiles is None:
            self._product_profiles = build_product_profiles(self.interactions, self.products)
        return self._product_profiles

    # -- figure plumbing ---------------------------------------------------
    def _save(self, fig: plt.Figure, name: str) -> Path:
        path = self.figure_dir / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        self.result.figures[name] = path
        logger.debug("figure -> %s", path)
        return path

    def _save_interactive(self, plotly_fig, name: str) -> Path | None:
        """Write an interactive HTML twin for figures where hovering helps."""
        if name not in self._interactive:
            return None
        try:
            path = self.figure_dir / f"{name}.html"
            # "directory" writes plotly.min.js once beside the files, so the
            # reports stay viewable offline without embedding 3 MB per figure.
            plotly_fig.write_html(path, include_plotlyjs="directory")
            self.result.figures[f"{name}_interactive"] = path
            return path
        except Exception as exc:  # plotly is optional to the pipeline's success
            logger.warning("Could not write interactive figure '%s': %s", name, exc)
            return None

    def _note(self, message: str) -> None:
        self.result.notes.append(message)
        logger.info("note: %s", message)

    # =====================================================================
    # Analyses
    # =====================================================================
    def user_analytics(self) -> pd.DataFrame:
        """Distribution of user activity, generosity and consistency."""
        profiles = self.user_profiles
        cfg = self.cfg
        heavy_pct = float(cfg.get("heavy_user_percentile", 99))
        inactive_max = int(cfg.get("inactive_user_max_interactions", 5))

        heavy_cut = float(np.percentile(profiles["n_interactions"], heavy_pct))
        heavy = profiles[profiles["n_interactions"] >= heavy_cut]
        inactive = profiles[profiles["n_interactions"] <= inactive_max]

        self.result.summary.update({
            "n_users": int(len(profiles)),
            "interactions_per_user_mean": float(profiles["n_interactions"].mean()),
            "interactions_per_user_median": float(profiles["n_interactions"].median()),
            "interactions_per_user_max": int(profiles["n_interactions"].max()),
            f"heavy_users_p{heavy_pct:g}": int(len(heavy)),
            f"heavy_user_threshold_p{heavy_pct:g}": heavy_cut,
            f"users_with_le_{inactive_max}_interactions": int(len(inactive)),
        })
        if "mean_rating" in profiles.columns:
            self.result.summary["mean_rating_given"] = float(profiles["mean_rating"].mean())
            zero_var = int((profiles["rating_std"] == 0).sum())
            self.result.summary["users_with_zero_rating_variance"] = zero_var
            self._note(
                f"{zero_var:,} users ({100 * zero_var / len(profiles):.1f}%) gave the same "
                f"rating to everything. That is an anomaly indicator for Module 4, not a "
                f"fraud finding - most are simply users with very short histories."
            )

        with plt.style.context(self.style):
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
            axes[0].hist(profiles["n_interactions"], bins=60, color="#4C72B0")
            axes[0].set_yscale("log")
            axes[0].set_xlabel("interactions per user")
            axes[0].set_ylabel("users (log)")
            axes[0].set_title("User activity distribution")
            axes[0].axvline(heavy_cut, color="#C44E52", ls="--", lw=1.2,
                            label=f"p{heavy_pct:g} = {heavy_cut:.0f}")
            axes[0].legend()

            if "mean_rating" in profiles.columns:
                axes[1].hist(profiles["mean_rating"].dropna(), bins=40, color="#55A868")
                axes[1].set_xlabel("mean rating given")
                axes[1].set_ylabel("users")
                axes[1].set_title("User generosity")

                axes[2].hist(profiles["rating_std"].dropna(), bins=40, color="#8172B2")
                axes[2].set_yscale("log")
                axes[2].set_xlabel("std. dev. of ratings given")
                axes[2].set_ylabel("users (log)")
                axes[2].set_title("User rating consistency")
            self._save(fig, "user_distributions")

        self.result.tables["user_profiles"] = profiles
        return profiles

    def product_analytics(self) -> pd.DataFrame:
        """Popularity concentration and the long tail."""
        profiles = self.product_profiles
        head_fraction = float(self.cfg.get("long_tail.head_fraction", 0.20))
        tail = long_tail_summary(profiles, head_fraction)

        self.result.summary.update({
            "n_items": int(len(profiles)),
            "interactions_per_item_mean": float(profiles["n_interactions"].mean()),
            "interactions_per_item_median": float(profiles["n_interactions"].median()),
            "interactions_per_item_max": int(profiles["n_interactions"].max()),
            f"share_held_by_top_{head_fraction:.0%}_items": tail["head_share"],
            "popularity_gini": tail["gini"],
        })
        self._note(
            f"The most popular {head_fraction:.0%} of items hold "
            f"{tail['head_share']:.1%} of all interactions (Gini {tail['gini']:.3f}). "
            f"A popularity baseline will therefore be strong, and any model that fails "
            f"to beat it is not learning personalisation."
        )

        counts = profiles["n_interactions"].sort_values(ascending=False).to_numpy(float)
        ranks = np.arange(1, len(counts) + 1)
        cumulative = np.cumsum(np.sort(counts)) / counts.sum()
        item_share = np.arange(1, len(counts) + 1) / len(counts)

        with plt.style.context(self.style):
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
            axes[0].loglog(ranks, counts, color="#4C72B0", lw=1.5)
            axes[0].set_xlabel("item popularity rank (log)")
            axes[0].set_ylabel("interactions (log)")
            axes[0].set_title("Long tail")

            axes[1].plot(item_share, cumulative, color="#C44E52", lw=2)
            axes[1].plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="perfect equality")
            axes[1].set_xlabel("cumulative share of items (least popular first)")
            axes[1].set_ylabel("cumulative share of interactions")
            axes[1].set_title(f"Lorenz curve (Gini {tail['gini']:.3f})")
            axes[1].legend()

            if "mean_rating" in profiles.columns:
                axes[2].hist(profiles["mean_rating"].dropna(), bins=40, color="#55A868")
                axes[2].set_xlabel("mean rating received")
                axes[2].set_ylabel("items")
                axes[2].set_title("Item quality distribution")
            self._save(fig, "long_tail")

        self._plot_long_tail_interactive(ranks, counts)
        self.result.tables["product_profiles"] = profiles
        return profiles

    def _plot_long_tail_interactive(self, ranks: np.ndarray, counts: np.ndarray) -> None:
        if "long_tail" not in self._interactive:
            return
        try:
            import plotly.graph_objects as go
        except ImportError:
            return
        step = max(1, len(ranks) // 5000)          # keep the HTML small
        fig = go.Figure(go.Scattergl(x=ranks[::step], y=counts[::step], mode="lines"))
        fig.update_layout(title="Item long tail", xaxis_type="log", yaxis_type="log",
                          xaxis_title="popularity rank", yaxis_title="interactions")
        self._save_interactive(fig, "long_tail")

    def review_analytics(self) -> pd.DataFrame:
        """Ratings, review length, verification, and sentiment when available."""
        frame = self.interactions
        table: dict[str, Any] = {}

        if "rating" in frame.columns:
            distribution = frame["rating"].value_counts(normalize=True).sort_index()
            table["rating_distribution"] = distribution
            self.result.summary["mean_rating_overall"] = float(frame["rating"].mean())
            self.result.summary["share_five_star"] = float((frame["rating"] == 5).mean())
            self._note(
                f"{100 * float((frame['rating'] == 5).mean()):.1f}% of retained interactions "
                f"are 5-star. This skew is characteristic of e-commerce review data and is "
                f"why rating value is treated as a feature, not as a graded relevance label."
            )

        lengths = None
        if "review_text" in frame.columns:
            lengths = frame["review_text"].fillna("").str.len()
            self.result.summary["review_length_mean_chars"] = float(lengths.mean())
            self.result.summary["review_length_median_chars"] = float(lengths.median())

        has_sentiment = "sentiment_compound" in frame.columns
        if not has_sentiment:
            self._note(
                "Sentiment distribution is not plotted: sentiment is produced by Module 6 "
                "in Phase 3, which has not run. Re-running EDA afterwards adds the panel."
            )

        n_panels = 2 + int("verified_purchase" in frame.columns) + int(has_sentiment)
        with plt.style.context(self.style):
            fig, axes = plt.subplots(1, n_panels, figsize=(5.3 * n_panels, 4.5))
            axes = np.atleast_1d(axes)
            index = 0

            if "rating" in frame.columns:
                counts = frame["rating"].value_counts().sort_index()
                axes[index].bar(counts.index.astype(float), counts.to_numpy(), color="#4C72B0",
                                width=0.6)
                axes[index].set_xlabel("rating")
                axes[index].set_ylabel("interactions")
                axes[index].set_title("Rating distribution")
                index += 1

            if lengths is not None:
                positive = lengths[lengths > 0]
                axes[index].hist(positive, bins=60, color="#55A868")
                axes[index].set_xscale("log")
                axes[index].set_yscale("log")
                axes[index].set_xlabel("review length (chars, log)")
                axes[index].set_ylabel("reviews (log)")
                axes[index].set_title("Review length")
                index += 1

            if "verified_purchase" in frame.columns:
                verified = frame["verified_purchase"].fillna(False)
                share = float(verified.mean())
                self.result.summary["share_verified_purchase"] = share
                grouped = frame.groupby(verified, observed=True)["rating"].mean()
                axes[index].bar([str(k) for k in grouped.index], grouped.to_numpy(),
                                color=["#C44E52", "#4C72B0"], width=0.55)
                axes[index].set_ylim(1, 5)
                axes[index].set_ylabel("mean rating")
                axes[index].set_xlabel("verified purchase")
                axes[index].set_title(f"Verified vs unverified ({share:.0%} verified)")
                delta = float(grouped.get(True, np.nan) - grouped.get(False, np.nan))
                if np.isfinite(delta):
                    self.result.summary["verified_minus_unverified_rating"] = delta
                    self._note(
                        f"Verified purchases rate {delta:+.3f} stars differently from "
                        f"unverified ones. `verified_purchase` is therefore kept as a "
                        f"review feature and as a trust input."
                    )
                index += 1

            if has_sentiment:
                axes[index].hist(frame["sentiment_compound"].dropna(), bins=50, color="#8172B2")
                axes[index].set_xlabel("Sentiment compound (transformer)")
                axes[index].set_ylabel("reviews")
                axes[index].set_title("Sentiment distribution")
            self._save(fig, "review_distributions")

        out = pd.DataFrame(table) if table else pd.DataFrame()
        self.result.tables["review_analytics"] = out
        return out

    def temporal_analytics(self) -> pd.DataFrame:
        """Volume, rating trend and category mix over time."""
        if "event_time" not in self.interactions.columns:
            self._note("No event_time column; temporal analytics skipped.")
            return pd.DataFrame()

        granularity = str(self.cfg.get("temporal_granularity", "ME"))
        min_share = float(self.cfg.get("trend_min_period_share", 0.001))

        frame = self.interactions.set_index("event_time").sort_index()
        monthly = pd.DataFrame({
            "interactions": frame["rating"].resample(granularity).size()
            if "rating" in frame.columns else frame.resample(granularity).size(),
        })
        if "rating" in frame.columns:
            monthly["mean_rating"] = frame["rating"].resample(granularity).mean()
        if "verified_purchase" in frame.columns:
            monthly["verified_share"] = (
                frame["verified_purchase"].fillna(False).resample(granularity).mean()
            )
        monthly["active_users"] = frame["user_id"].resample(granularity).nunique()

        # Early Amazon years hold a handful of reviews and stretch every axis.
        threshold = max(1.0, min_share * len(self.interactions))
        dense = monthly[monthly["interactions"] >= threshold]
        if len(dense) < len(monthly):
            self._note(
                f"Trend panels start at {dense.index.min().date() if len(dense) else 'n/a'}: "
                f"{len(monthly) - len(dense)} early periods hold fewer than "
                f"{threshold:.0f} interactions each and would otherwise dominate the axis. "
                f"Full history is retained in the table, only the plots are trimmed."
            )

        self.result.summary.update({
            "first_interaction": str(self.interactions["event_time"].min().date()),
            "last_interaction": str(self.interactions["event_time"].max().date()),
            "n_active_months": int((monthly["interactions"] > 0).sum()),
            "peak_month": str(monthly["interactions"].idxmax().date()),
            "peak_month_interactions": int(monthly["interactions"].max()),
        })

        with plt.style.context(self.style):
            fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
            axes[0].plot(dense.index, dense["interactions"], color="#4C72B0", lw=1.6)
            axes[0].set_ylabel("interactions")
            axes[0].set_title("Interaction volume over time")

            if "mean_rating" in dense.columns:
                axes[1].plot(dense.index, dense["mean_rating"], color="#C44E52", lw=1.6)
                axes[1].set_ylabel("mean rating")
                axes[1].set_ylim(1, 5)
                axes[1].set_title("Rating trend")

            axes[2].plot(dense.index, dense["active_users"], color="#55A868", lw=1.6)
            axes[2].set_ylabel("distinct active users")
            axes[2].set_xlabel("time")
            axes[2].set_title("Active users over time")
            self._save(fig, "temporal_trends")

        self._plot_temporal_interactive(dense)
        self._category_trends(granularity, threshold)
        self.result.tables["temporal"] = monthly
        return monthly

    def _plot_temporal_interactive(self, dense: pd.DataFrame) -> None:
        if not ({"reviews_per_month", "rating_trend"} & self._interactive):
            return
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            return
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=dense.index, y=dense["interactions"],
                                 name="interactions", mode="lines"), secondary_y=False)
        if "mean_rating" in dense.columns:
            fig.add_trace(go.Scatter(x=dense.index, y=dense["mean_rating"],
                                     name="mean rating", mode="lines"), secondary_y=True)
        fig.update_layout(title="Interaction volume and rating trend",
                          hovermode="x unified")
        fig.update_yaxes(title_text="interactions", secondary_y=False)
        fig.update_yaxes(title_text="mean rating", range=[1, 5], secondary_y=True)
        self._save_interactive(fig, "reviews_per_month")

    def _category_trends(self, granularity: str, threshold: float) -> None:
        """How the category mix shifts over time - the premise for Module 13."""
        if self.products is None or "category_node" not in self.products.columns:
            self._note("No product metadata; category trends skipped.")
            return

        top_n = int(self.cfg.get("top_n_categories", 10))
        mapping = self.products.set_index("item_id")["category_node"]
        merged = self.interactions[["item_id", "event_time"]].copy()
        merged["category"] = merged["item_id"].map(mapping)
        merged = merged.dropna(subset=["category"])
        if merged.empty:
            self._note("Category could not be resolved for any interaction; trends skipped.")
            return

        top = merged["category"].value_counts().head(top_n).index
        pivot = (
            merged[merged["category"].isin(top)]
            .set_index("event_time")
            .groupby([pd.Grouper(freq=granularity), "category"], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        pivot = pivot[pivot.sum(axis=1) >= threshold]
        if pivot.empty:
            self._note("Too few interactions per period to plot category trends.")
            return

        shares = pivot.div(pivot.sum(axis=1), axis=0)
        self.result.tables["category_trends"] = pivot
        self.result.summary["n_categories"] = int(merged["category"].nunique())

        with plt.style.context(self.style):
            fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
            axes[0].stackplot(pivot.index, *[pivot[c] for c in pivot.columns],
                              labels=list(pivot.columns))
            axes[0].set_ylabel("interactions")
            axes[0].set_title(f"Top {len(pivot.columns)} categories over time (volume)")
            axes[0].legend(loc="upper left", fontsize=7, ncol=2)

            for column in shares.columns:
                axes[1].plot(shares.index, shares[column], lw=1.4, label=column)
            axes[1].set_ylabel("share of interactions")
            axes[1].set_xlabel("time")
            axes[1].set_title("Category mix (share) - the premise for temporal weighting")
            self._save(fig, "category_trends")

        try:
            import plotly.express as px
            self._save_interactive(
                px.area(shares.reset_index().melt(id_vars="event_time",
                                                  var_name="category", value_name="share"),
                        x="event_time", y="share", color="category",
                        title="Category mix over time"),
                "category_trends",
            )
        except ImportError:
            pass

        # Quantify drift rather than asserting it: how much does an average
        # user's category mix move between their first and second half?
        drift = self._preference_drift(merged)
        if drift is not None:
            self.result.summary["mean_user_category_drift"] = drift
            self._note(
                f"Mean per-user category drift between first and second half of history is "
                f"{drift:.3f} (0 = identical mix, 1 = disjoint). This is the measured "
                f"premise for recency weighting in Module 13, not an assumption."
            )

    def _preference_drift(self, merged: pd.DataFrame) -> float | None:
        """Mean total-variation distance between each user's early and late category mix."""
        frame = merged.join(self.interactions[["user_id"]])
        eligible = frame.groupby("user_id", observed=True).size()
        eligible = eligible[eligible >= 6].index
        if len(eligible) == 0:
            return None

        subset = frame[frame["user_id"].isin(eligible)].sort_values(["user_id", "event_time"])
        distances: list[float] = []
        for _, group in subset.groupby("user_id", observed=True):
            half = len(group) // 2
            early = group.iloc[:half]["category"].value_counts(normalize=True)
            late = group.iloc[half:]["category"].value_counts(normalize=True)
            categories = early.index.union(late.index)
            early = early.reindex(categories, fill_value=0.0)
            late = late.reindex(categories, fill_value=0.0)
            distances.append(float(0.5 * np.abs(early - late).sum()))
        return float(np.mean(distances)) if distances else None

    # =====================================================================
    def run(self) -> EDAResult:
        """Run every analysis and return the collected result."""
        logger.info("=" * 72)
        logger.info("Exploratory Data Analysis")
        logger.info("=" * 72)

        self.result.summary["n_interactions"] = int(len(self.interactions))
        with log_stage(logger, "User analytics"):
            self.user_analytics()
        with log_stage(logger, "Product analytics"):
            self.product_analytics()
        with log_stage(logger, "Review analytics"):
            self.review_analytics()
        with log_stage(logger, "Temporal analytics"):
            self.temporal_analytics()

        users = self.result.summary.get("n_users", 1)
        items = self.result.summary.get("n_items", 1)
        self.result.summary["density"] = float(len(self.interactions) / max(1, users * items))

        logger.info("EDA complete: %d figures, %d tables",
                    len(self.result.figures), len(self.result.tables))
        return self.result

    def save(self, tables_dir: str | Path | None = None) -> tuple[Path, Path]:
        """Write the EDA summary as JSON and Markdown, plus every table as CSV."""
        target = resolve_path(tables_dir or self.config.get("paths.tables", "reports/tables"))
        target.mkdir(parents=True, exist_ok=True)

        json_path = target / "eda_summary.json"
        md_path = target / "eda_summary.md"
        payload = {
            "summary": self.result.summary,
            "figures": {k: str(v) for k, v in self.result.figures.items()},
            "notes": self.result.notes,
        }
        json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        md_path.write_text(self.result.to_markdown(), encoding="utf-8")

        for name, table in self.result.tables.items():
            if not table.empty:
                table.to_csv(target / f"eda_{name}.csv")
        logger.info("EDA summary written to %s and %s", json_path, md_path)
        return json_path, md_path
