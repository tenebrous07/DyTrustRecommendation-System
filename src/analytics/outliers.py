"""Outlier detection (Module 4).

Produces **anomaly indicators**, never fraud labels. Amazon Reviews 2023 has no
manipulation ground truth, so nothing in this module can be validated against
the truth, and nothing in it is entitled to call a user "fake". Every output
column is named for what it actually is: a statistical flag, an isolation-forest
score, a count of agreeing indicators.

Three layers, deliberately kept separate so they can disagree:

1. **IQR** - a classical per-feature fence. Transparent and explainable, which
   makes it the thing to show a viva examiner.
2. **Z-score on log1p** - behavioural counts are roughly lognormal, so a raw
   z-score would flag the upper tail of a perfectly ordinary distribution
   simply because it is skewed. The log transform is not cosmetic.
3. **Isolation Forest / LOF** - multivariate, catches combinations no single
   feature reveals (few reviews *and* all 5-star *and* seconds apart).

A user is called ``corroborated`` when several independent layers agree. That
is a triage ranking for human review, not a verdict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..utils.config import Config, resolve_path
from ..utils.logging import get_logger, log_stage

__all__ = ["OutlierReport", "OutlierDetector"]

logger = get_logger(__name__)


@dataclass
class OutlierReport:
    """Indicator output plus the provenance needed to interpret it."""

    scores: pd.DataFrame
    features_used: list[str] = field(default_factory=list)
    features_missing: list[str] = field(default_factory=list)
    fitted_scope: str = "unknown"
    mode: str = "unknown"
    summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "features_used": self.features_used,
            "features_missing": self.features_missing,
            "fitted_scope": self.fitted_scope,
            "mode": self.mode,
            "notes": self.notes,
        }

    def to_markdown(self, top_n: int = 25) -> str:
        lines = ["# Outlier Indicators (Module 4)", "",
                 "> These are anomaly **indicators**, not fraud labels. This dataset has no",
                 "> manipulation ground truth, so none of the numbers below can be validated",
                 "> against the truth. They rank users for inspection; they do not convict them.",
                 "", "## Summary", "", "| Metric | Value |", "| --- | ---: |"]
        for key, value in self.summary.items():
            rendered = f"{value:,.4f}" if isinstance(value, float) else f"{value:,}" \
                if isinstance(value, int) else str(value)
            lines.append(f"| {key.replace('_', ' ')} | {rendered} |")

        lines += ["", f"## Features used ({len(self.features_used)})", "",
                  ", ".join(f"`{f}`" for f in self.features_used) or "_none_"]
        if self.features_missing:
            lines += ["", f"## Features unavailable ({len(self.features_missing)})", "",
                      ", ".join(f"`{f}`" for f in self.features_missing)]

        if "n_indicators" in self.scores.columns:
            top = self.scores.nlargest(top_n, ["n_indicators", "iforest_score"])
            display = [c for c in ("n_indicators", "iforest_score", "lof_score",
                                   "iqr_flags", "zscore_flags", "flag_reasons")
                       if c in top.columns]
            lines += ["", f"## Top {len(top)} users by agreeing indicators", "",
                      "| user_id | " + " | ".join(display) + " |",
                      "| --- |" + " ---: |" * len(display)]
            for user_id, row in top[display].iterrows():
                cells = [f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c])
                         for c in display]
                lines.append(f"| `{user_id}` | " + " | ".join(cells) + " |")

        if self.notes:
            lines += ["", "## Notes", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


class OutlierDetector:
    """Fits statistical fences and unsupervised detectors over user profiles."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.cfg: Config = config.require("outliers")
        self.declared_features: list[str] = list(self.cfg.require("features"))
        self.min_indicators = int(self.cfg.get("consensus.min_indicators", 2))

        # Transductive vs inductive. LOF with novelty=True is documented as
        # invalid for scoring the very data it was fitted on, so the mode is
        # chosen from the leakage setting rather than assumed:
        #   fit_on_train_split_only=False -> exploratory, score the fitted rows
        #                                    (transductive, novelty=False)
        #   fit_on_train_split_only=True  -> score unseen users later
        #                                    (inductive, novelty=True)
        self.inductive = bool(self.cfg.get("fit_on_train_split_only", False))
        self.mode = "inductive" if self.inductive else "transductive"

        self._features: list[str] = []
        self._missing: list[str] = []
        self._bounds: dict[str, tuple[float, float]] = {}
        self._log_stats: dict[str, tuple[float, float, float]] = {}
        self._iforest: Pipeline | None = None
        self._lof: Pipeline | LocalOutlierFactor | None = None
        self._lof_train_scores: np.ndarray | None = None
        self._lof_train_flags: np.ndarray | None = None
        self._fitted_scope = "not fitted"
        self._fitted_index: pd.Index | None = None

    # -- fitting -----------------------------------------------------------
    def fit(self, profiles: pd.DataFrame, scope: str = "all data") -> "OutlierDetector":
        """Fit fences and detectors on ``profiles``.

        Args:
            profiles: User profiles, one row per user.
            scope: Human-readable description of what was fitted on, recorded in
                the report so a reader can tell whether it was train-only.
        """
        self._features = [f for f in self.declared_features if f in profiles.columns]
        self._missing = [f for f in self.declared_features if f not in profiles.columns]
        if not self._features:
            raise ValueError(
                f"None of the configured outlier features exist in the profiles. "
                f"Configured: {self.declared_features}; available: {list(profiles.columns)}"
            )
        if self._missing:
            logger.warning("Outlier features unavailable in this dataset: %s", self._missing)

        self._fitted_scope = scope
        self._fitted_index = profiles.index
        matrix = self._matrix(profiles)

        with log_stage(logger, f"Fit outlier detectors ({self.mode}, on {scope})") as stage:
            self._fit_iqr(profiles)
            self._fit_zscore(profiles)
            self._fit_isolation_forest(matrix)
            self._fit_lof(matrix)
            stage["features"] = len(self._features)
            stage["rows"] = f"{len(profiles):,}"
        return self

    def _matrix(self, profiles: pd.DataFrame) -> pd.DataFrame:
        """Numeric feature matrix with infinities neutralised."""
        matrix = profiles[self._features].astype("float64").copy()
        return matrix.replace([np.inf, -np.inf], np.nan)

    def _fit_iqr(self, profiles: pd.DataFrame) -> None:
        if not bool(self.cfg.get("statistical.iqr.enabled", True)):
            return
        multiplier = float(self.cfg.get("statistical.iqr.multiplier", 1.5))
        for feature in self._features:
            series = profiles[feature].astype("float64").replace([np.inf, -np.inf], np.nan)
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            self._bounds[feature] = (q1 - multiplier * iqr, q3 + multiplier * iqr)

    def _fit_zscore(self, profiles: pd.DataFrame) -> None:
        if not bool(self.cfg.get("statistical.zscore.enabled", True)):
            return
        use_log = bool(self.cfg.get("statistical.zscore.log_transform", True))
        for feature in self._features:
            series = profiles[feature].astype("float64").replace([np.inf, -np.inf], np.nan)
            if use_log:
                # log1p needs a non-negative domain; shift if the feature can go
                # negative rather than silently producing NaN.
                minimum = series.min()
                shift = float(-minimum) if pd.notna(minimum) and minimum < 0 else 0.0
                series = np.log1p(series + shift)
            else:
                shift = 0.0
            mean, std = float(series.mean()), float(series.std())
            # (mean, std, shift): shift is replayed at score time so the same
            # transform is applied to unseen rows.
            self._log_stats[feature] = (mean, std if std > 0 else 1.0, shift)
            self._bounds.setdefault(feature, (np.nan, np.nan))

    def _fit_isolation_forest(self, matrix: pd.DataFrame) -> None:
        if not bool(self.cfg.get("isolation_forest.enabled", True)):
            return
        params = self.cfg.require("isolation_forest")
        self._iforest = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", IsolationForest(
                n_estimators=int(params.get("n_estimators", 200)),
                max_samples=params.get("max_samples", "auto"),
                contamination=params.get("contamination", "auto"),
                random_state=int(params.get("random_state", 42)),
                n_jobs=int(params.get("n_jobs", -1)),
            )),
        ]).fit(matrix)

    def _fit_lof(self, matrix: pd.DataFrame) -> None:
        if not bool(self.cfg.get("local_outlier_factor.enabled", True)):
            return
        params = self.cfg.require("local_outlier_factor")
        n_neighbors = min(int(params.get("n_neighbors", 20)), max(2, len(matrix) - 1))
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=params.get("contamination", "auto"),
            novelty=self.inductive,
        )
        pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ])
        if self.inductive:
            self._lof = pipeline.fit(matrix)
        else:
            # Transductive: fit_predict is the only valid call, and the scores
            # for the fitted rows are read off negative_outlier_factor_.
            # fit_predict is the only valid call in transductive mode. Capture
            # both the labels and the scores now: recomputing them at score time
            # would refit the model and could return different flags.
            self._lof_train_flags = pipeline.fit_predict(matrix) == -1
            self._lof = pipeline
            self._lof_train_scores = -model.negative_outlier_factor_

    # -- scoring -----------------------------------------------------------
    def score(self, profiles: pd.DataFrame) -> OutlierReport:
        """Score ``profiles`` and assemble the indicator report."""
        if not self._features:
            raise RuntimeError("fit() must be called before score().")
        if not self.inductive and self._fitted_index is not None:
            if not profiles.index.equals(self._fitted_index):
                raise RuntimeError(
                    "This detector was fitted transductively (LOF novelty=False), so it can "
                    "only score the exact rows it was fitted on. Set "
                    "outliers.fit_on_train_split_only=true for an inductive detector."
                )

        matrix = self._matrix(profiles)
        scores = pd.DataFrame(index=profiles.index)
        reasons: list[list[str]] = [[] for _ in range(len(profiles))]
        notes: list[str] = []

        # -- IQR -----------------------------------------------------------
        iqr_hits = pd.Series(0, index=profiles.index, dtype="int64")
        if bool(self.cfg.get("statistical.iqr.enabled", True)):
            for feature in self._features:
                low, high = self._bounds.get(feature, (np.nan, np.nan))
                if not np.isfinite(low) and not np.isfinite(high):
                    continue
                values = matrix[feature]
                outside = ((values < low) | (values > high)).fillna(False)
                iqr_hits += outside.astype("int64")
                for position in np.flatnonzero(outside.to_numpy()):
                    reasons[position].append(f"iqr:{feature}")
        scores["iqr_flags"] = iqr_hits

        # -- z-score -------------------------------------------------------
        z_hits = pd.Series(0, index=profiles.index, dtype="int64")
        if bool(self.cfg.get("statistical.zscore.enabled", True)):
            threshold = float(self.cfg.get("statistical.zscore.threshold", 3.0))
            use_log = bool(self.cfg.get("statistical.zscore.log_transform", True))
            for feature in self._features:
                stats = self._log_stats.get(feature)
                if stats is None:
                    continue
                mean, std, shift = stats
                values = matrix[feature]
                transformed = np.log1p(values + shift) if use_log else values
                z = ((transformed - mean) / std).abs()
                outside = (z > threshold).fillna(False)
                z_hits += outside.astype("int64")
                for position in np.flatnonzero(outside.to_numpy()):
                    reasons[position].append(f"z>{threshold:g}:{feature}")
        scores["zscore_flags"] = z_hits

        # -- Isolation Forest ------------------------------------------------
        if self._iforest is not None:
            # score_samples: higher = more normal. Negate so that in every
            # column in this frame, higher consistently means more anomalous.
            raw = -self._iforest.score_samples(matrix)
            scores["iforest_score"] = raw
            scores["iforest_flag"] = self._iforest.predict(matrix) == -1
            for position in np.flatnonzero(scores["iforest_flag"].to_numpy()):
                reasons[position].append("isolation_forest")

        # -- LOF -------------------------------------------------------------
        if self._lof is not None:
            if self.inductive:
                lof_raw = -self._lof.score_samples(matrix)
                lof_flag = self._lof.predict(matrix) == -1
            else:
                lof_raw = self._lof_train_scores
                lof_flag = self._lof_train_flags
            scores["lof_score"] = lof_raw
            scores["lof_flag"] = lof_flag
            for position in np.flatnonzero(np.asarray(lof_flag)):
                reasons[position].append("local_outlier_factor")

        # -- consensus --------------------------------------------------------
        indicators = pd.Series(0, index=profiles.index, dtype="int64")
        indicators += (scores["iqr_flags"] > 0).astype("int64")
        indicators += (scores["zscore_flags"] > 0).astype("int64")
        if "iforest_flag" in scores.columns:
            indicators += scores["iforest_flag"].astype("int64")
        if "lof_flag" in scores.columns:
            indicators += pd.Series(scores["lof_flag"], index=scores.index).astype("int64")
        scores["n_indicators"] = indicators
        scores["corroborated"] = indicators >= self.min_indicators
        scores["flag_reasons"] = [";".join(r) for r in reasons]

        if "iforest_score" in scores.columns:
            scores["anomaly_rank"] = scores["iforest_score"].rank(
                ascending=False, method="min"
            ).astype("int64")

        summary = {
            "n_users_scored": int(len(scores)),
            "n_flagged_iqr": int((scores["iqr_flags"] > 0).sum()),
            "n_flagged_zscore": int((scores["zscore_flags"] > 0).sum()),
            "n_flagged_iforest": int(scores.get("iforest_flag", pd.Series(dtype=bool)).sum()),
            "n_flagged_lof": int(pd.Series(scores.get("lof_flag", []), dtype="boolean")
                                 .fillna(False).sum()) if "lof_flag" in scores else 0,
            "n_corroborated": int(scores["corroborated"].sum()),
            "corroborated_share": float(scores["corroborated"].mean()),
            "min_indicators_for_corroboration": self.min_indicators,
        }

        notes.append(
            "These are unsupervised anomaly indicators. The dataset carries no fraud "
            "labels, so precision and recall cannot be computed here. Quantitative "
            "validation happens only under the synthetic-injection protocol in Phase 16, "
            "where labels exist by construction."
        )
        notes.append(
            f"Detectors were fitted on: {self._fitted_scope} ({self.mode}). Any use of "
            f"these scores as MODEL FEATURES must refit on the training split only, by "
            f"setting outliers.fit_on_train_split_only=true."
        )
        if not self.inductive:
            notes.append(
                "LOF ran with novelty=False because it is scoring the rows it was fitted "
                "on; scikit-learn documents novelty=True as invalid for that case."
            )
        # Multiple comparisons: IQR fires if ANY single feature is outside its
        # fence, so with k features the per-user false-positive rate compounds.
        # Quote the corroborated count, not the raw IQR count.
        iqr_share = float((scores["iqr_flags"] > 0).mean())
        if iqr_share > 0.25:
            notes.append(
                f"IQR alone flags {iqr_share:.1%} of users, but that is a multiple-comparisons "
                f"artefact rather than a finding: the rule fires when ANY of "
                f"{len(self._features)} features falls outside its fence, so the per-user "
                f"hit rate compounds with the feature count. Use `n_indicators` / "
                f"`corroborated`, not the raw IQR count."
            )

        agreement = self._agreement(scores)
        if agreement is not None:
            notes.append(
                f"Isolation Forest and LOF agree on {agreement:.1%} of their flags "
                f"(Jaccard). Low agreement is expected - they model different notions of "
                f"outlyingness - and is a reason to use the consensus count rather than "
                f"any single detector."
            )

        report = OutlierReport(
            scores=scores, features_used=list(self._features),
            features_missing=list(self._missing), fitted_scope=self._fitted_scope,
            mode=self.mode, summary=summary, notes=notes,
        )
        logger.info(
            "Outlier indicators: %s users, %s corroborated by >= %d layers (%.2f%%)",
            f"{summary['n_users_scored']:,}", f"{summary['n_corroborated']:,}",
            self.min_indicators, 100 * summary["corroborated_share"],
        )
        return report

    @staticmethod
    def _agreement(scores: pd.DataFrame) -> float | None:
        if "iforest_flag" not in scores.columns or "lof_flag" not in scores.columns:
            return None
        a = scores["iforest_flag"].to_numpy(dtype=bool)
        b = np.asarray(scores["lof_flag"], dtype=bool)
        union = int((a | b).sum())
        return float((a & b).sum() / union) if union else None

    # -- persistence -------------------------------------------------------
    def save(self, report: OutlierReport, tables_dir: str | Path | None = None
             ) -> tuple[Path, Path]:
        target = resolve_path(tables_dir or self.config.get("paths.tables", "reports/tables"))
        target.mkdir(parents=True, exist_ok=True)
        json_path = target / "outlier_report.json"
        md_path = target / "outlier_report.md"
        json_path.write_text(json.dumps(report.to_dict(), indent=2, default=str),
                             encoding="utf-8")
        md_path.write_text(
            report.to_markdown(int(self.cfg.get("consensus.report_top_n", 50))),
            encoding="utf-8",
        )
        report.scores.to_csv(target / "outlier_scores.csv")
        logger.info("Outlier report written to %s and %s", json_path, md_path)
        return json_path, md_path
