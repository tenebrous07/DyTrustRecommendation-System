"""Data cleaning (Module 2).

Governing principle: **invalid records are removed, suspicious records are
flagged.** A review is invalid when it cannot be interpreted at all - a rating
outside the instrument's range, a timestamp outside the dataset's collection
window, a missing user or item key. Everything else that merely looks odd (a
5,000-word review, a user posting identical text on six products, a burst of
reviews in one minute) is *signal*, not noise: those rows are exactly what the
anomaly detector in Module 11 is supposed to find. Deleting them here would
quietly destroy the evidence the trust component depends on, and would flatter
its results.

Every decision is counted and written to ``reports/tables/cleaning_report.*``.

Known caveat, recorded honestly: k-core filtering (:meth:`DataCleaner._k_core`)
selects users and items using their interaction counts over the *whole*
timeline, including the period that later becomes the test split. This is
standard practice in the recommender-systems literature, but it is a mild form
of selection leakage. It is applied before splitting, disclosed in the report,
and can be disabled in config.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.config import Config, resolve_path
from ..utils.logging import get_logger, log_stage
from .loader import LoadResult

__all__ = ["CleaningReport", "CleanResult", "DataCleaner", "clean_dataset"]

logger = get_logger(__name__)

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_ASIN_OK = re.compile(r"^[A-Z0-9]{10}$")


# =============================================================================
# Report
# =============================================================================
@dataclass
class CleaningReport:
    """Auditable record of every row removed or flagged, and why."""

    rows_raw: int = 0
    rows_final: int = 0
    users_raw: int = 0
    users_final: int = 0
    items_raw: int = 0
    items_final: int = 0
    steps: list[dict[str, Any]] = field(default_factory=list)
    missing_values: dict[str, dict[str, Any]] = field(default_factory=dict)
    flags: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def record(self, step: str, *, before: int, after: int, detail: str = "") -> None:
        removed = before - after
        self.steps.append({
            "step": step,
            "rows_before": before,
            "rows_after": after,
            "rows_removed": removed,
            "pct_removed": round(100 * removed / before, 4) if before else 0.0,
            "detail": detail,
        })
        level = logger.warning if before and removed / before > 0.25 else logger.info
        level("  %-32s %9s -> %9s  (-%s, %.2f%%)  %s", step, f"{before:,}", f"{after:,}",
              f"{removed:,}", 100 * removed / before if before else 0.0, detail)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "rows_raw": self.rows_raw,
                "rows_final": self.rows_final,
                "rows_removed": self.rows_raw - self.rows_final,
                "retention_pct": round(100 * self.rows_final / self.rows_raw, 3) if self.rows_raw else 0.0,
                "users_raw": self.users_raw, "users_final": self.users_final,
                "items_raw": self.items_raw, "items_final": self.items_final,
            },
            "steps": self.steps,
            "missing_values": self.missing_values,
            "flags_set": self.flags,
            "notes": self.notes,
        }

    def to_markdown(self) -> str:
        summary = self.to_dict()["summary"]
        lines = [
            "# Cleaning Report", "",
            "## Summary", "",
            "| Metric | Value |", "| --- | ---: |",
            f"| Raw interactions | {summary['rows_raw']:,} |",
            f"| Retained interactions | {summary['rows_final']:,} |",
            f"| Removed | {summary['rows_removed']:,} |",
            f"| Retention | {summary['retention_pct']:.2f}% |",
            f"| Users (raw -> final) | {summary['users_raw']:,} -> {summary['users_final']:,} |",
            f"| Items (raw -> final) | {summary['items_raw']:,} -> {summary['items_final']:,} |",
            "", "## Pipeline steps", "",
            "| Step | Before | After | Removed | % | Detail |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for step in self.steps:
            lines.append(
                f"| {step['step']} | {step['rows_before']:,} | {step['rows_after']:,} | "
                f"{step['rows_removed']:,} | {step['pct_removed']:.2f} | {step['detail']} |"
            )

        lines += ["", "## Missing values (after cleaning)", "",
                  "| Column | Null count | Null % |", "| --- | ---: | ---: |"]
        for column, stats in sorted(self.missing_values.items()):
            lines.append(f"| {column} | {stats['null_count']:,} | {stats['null_pct']:.2f} |")

        if self.flags:
            lines += ["", "## Rows flagged (retained, not deleted)", "",
                      "| Flag reason | Rows |", "| --- | ---: |"]
            for reason, count in sorted(self.flags.items(), key=lambda kv: -kv[1]):
                lines.append(f"| {reason} | {count:,} |")

        if self.notes:
            lines += ["", "## Notes and caveats", ""]
            lines += [f"- {note}" for note in self.notes]
        return "\n".join(lines) + "\n"


@dataclass(slots=True)
class CleanResult:
    """Output of Module 2."""

    interactions: pd.DataFrame
    products: pd.DataFrame | None
    report: CleaningReport


# =============================================================================
# Cleaner
# =============================================================================
class DataCleaner:
    """Applies the cleaning policy declared in ``data_config.yaml``."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.rules: Config = config.require("cleaning")
        tier = config.require("tiers")[config.require("active_tier")]
        self.k_core: Config = tier["k_core"]
        self.report = CleaningReport()

    # -- entrypoint --------------------------------------------------------
    def clean(self, loaded: LoadResult) -> CleanResult:
        frame = loaded.interactions.copy()
        products = None if loaded.products is None else loaded.products.copy()

        self.report.rows_raw = len(frame)
        self.report.users_raw = int(frame["user_id"].nunique())
        self.report.items_raw = int(frame["item_id"].nunique())

        logger.info("=" * 72)
        logger.info("Cleaning %s interactions", f"{len(frame):,}")
        logger.info("=" * 72)

        self._analyse_missing(frame, when="before")

        with log_stage(logger, "Clean interactions"):
            frame = self._normalise_ids(frame)
            frame = self._drop_null_keys(frame)
            frame = self._validate_ratings(frame)
            frame = self._validate_timestamps(frame)
            frame = self._deduplicate(frame)
            frame = self._flag_suspicious(frame)
            frame = self._k_core(frame)

        if products is not None:
            with log_stage(logger, "Clean product metadata") as stage:
                products = self._clean_products(products, keep_items=set(frame["item_id"].unique()))
                stage["rows"] = f"{len(products):,}"

        frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)

        self.report.rows_final = len(frame)
        self.report.users_final = int(frame["user_id"].nunique())
        self.report.items_final = int(frame["item_id"].nunique())
        if len(frame) and "event_time" in frame.columns:
            self.report.notes.append(
                f"Final retained time span: {frame['event_time'].min().date()} to "
                f"{frame['event_time'].max().date()}."
            )
        self._analyse_missing(frame, when="after")
        self._count_flags(frame)

        logger.info(
            "Cleaning complete: %s rows retained (%.2f%%), %s users, %s items",
            f"{len(frame):,}",
            100 * len(frame) / max(1, self.report.rows_raw),
            f"{self.report.users_final:,}", f"{self.report.items_final:,}",
        )
        return CleanResult(frame, products, self.report)

    # -- steps -------------------------------------------------------------
    def _analyse_missing(self, frame: pd.DataFrame, *, when: str) -> None:
        total = len(frame)
        stats = {
            column: {
                "null_count": int(frame[column].isna().sum()),
                "null_pct": round(100 * float(frame[column].isna().mean()), 4) if total else 0.0,
                "dtype": str(frame[column].dtype),
            }
            for column in frame.columns
        }
        if when == "after":
            self.report.missing_values = stats
        worst = sorted(stats.items(), key=lambda kv: -kv[1]["null_pct"])[:5]
        logger.info(
            "Missing values (%s): %s", when,
            ", ".join(f"{c}={s['null_pct']:.1f}%" for c, s in worst if s["null_pct"] > 0) or "none",
        )

    def _normalise_ids(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Trim and upper-case identifiers so join keys cannot drift on case."""
        before = len(frame)
        for column in ("user_id", "item_id", "variant_id"):
            if column in frame.columns:
                frame[column] = frame[column].str.strip().str.upper()

        malformed = 0
        if "item_id" in frame.columns:
            valid = frame["item_id"].str.match(_ASIN_OK, na=False)
            malformed = int((~valid & frame["item_id"].notna()).sum())
        self.report.record(
            "normalise_ids", before=before, after=len(frame),
            detail=f"{malformed:,} item_id values do not match the 10-char ASIN pattern (kept)",
        )
        return frame

    def _drop_null_keys(self, frame: pd.DataFrame) -> pd.DataFrame:
        """A row without a user or item is not an interaction."""
        before = len(frame)
        frame = frame.dropna(subset=["user_id", "item_id"])
        self.report.record("drop_null_keys", before=before, after=len(frame),
                           detail="rows missing user_id or item_id")
        return frame

    def _validate_ratings(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Remove ratings outside the instrument's range - uninterpretable."""
        if "rating" not in frame.columns:
            self.report.notes.append("No rating column; rating validation skipped.")
            return frame
        low = float(self.rules["rating"]["min"])
        high = float(self.rules["rating"]["max"])
        before = len(frame)
        valid = frame["rating"].between(low, high, inclusive="both")
        n_null = int(frame["rating"].isna().sum())
        frame = frame[valid.fillna(False)]
        self.report.record(
            "validate_ratings", before=before, after=len(frame),
            detail=f"outside [{low}, {high}] or null ({n_null:,} null)",
        )
        return frame

    def _validate_timestamps(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Remove timestamps outside the dataset's documented collection window."""
        if "event_time" not in frame.columns:
            self.report.notes.append("No timestamp column; temporal validation skipped.")
            return frame
        low = pd.Timestamp(self.rules["timestamp"]["min_iso"])
        high = pd.Timestamp(self.rules["timestamp"]["max_iso"])
        before = len(frame)
        valid = frame["event_time"].between(low, high, inclusive="both")
        frame = frame[valid.fillna(False)]
        self.report.record(
            "validate_timestamps", before=before, after=len(frame),
            detail=f"outside [{low.date()}, {high.date()}] or unparseable",
        )
        if len(frame):
            # Recorded at this step, so it precedes k-core. The final span can be
            # narrower; label it precisely rather than letting a viva question
            # find the discrepancy.
            self.report.notes.append(
                f"Time span after timestamp validation (before k-core): "
                f"{frame['event_time'].min().date()} to {frame['event_time'].max().date()}."
            )
        return frame

    def _deduplicate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Remove exact ingestion duplicates only.

        Duplicate *content* (same user, same text, different product) is not
        removed here - it is a coordinated-review signal and is flagged instead.
        """
        keys = [k for k in self.rules["duplicates"]["exact_keys"] if k in frame.columns]
        before = len(frame)
        frame = frame.drop_duplicates(subset=keys, keep="first")
        self.report.record(
            "deduplicate_exact", before=before, after=len(frame),
            detail=f"identical ({', '.join(keys)})",
        )

        # A user reviewing the same item more than once at different times is
        # legitimate in Amazon data (edits re-issued as new rows), but for an
        # implicit-feedback graph we need one edge per (user, item). Keep latest.
        before = len(frame)
        frame = frame.sort_values("timestamp", kind="stable").drop_duplicates(
            subset=["user_id", "item_id"], keep="last"
        )
        self.report.record(
            "collapse_user_item", before=before, after=len(frame),
            detail="repeat reviews of the same item by the same user; kept most recent",
        )
        return frame

    def _flag_suspicious(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach ``outlier_flag`` / ``outlier_reasons``. Removes nothing.

        These are cheap rule-based indicators. The learned detectors live in
        Module 4 (Isolation Forest) and Module 11 (CoLA); this step exists so
        that later modules have a transparent, explainable baseline to compare
        against, and so the dashboard can say *why* a row looked odd.
        """
        reasons: defaultdict[int, list[str]] = defaultdict(list)

        def mark(mask: pd.Series, label: str) -> None:
            mask = mask.fillna(False)
            for index in frame.index[mask]:
                reasons[index].append(label)
            if int(mask.sum()):
                logger.info("  flag %-28s %s rows", label, f"{int(mask.sum()):,}")

        if "review_text" in frame.columns:
            lengths = frame["review_text"].fillna("").str.len()
            mark(lengths == 0, "empty_review_text")
            if len(lengths) > 10:
                mark(lengths > lengths.quantile(0.999), "extreme_review_length")

            if self.rules["duplicates"].get("flag_repeated_text_by_user", True):
                # Same user posting byte-identical non-trivial text on >1 item.
                text_key = frame["review_text"].fillna("")
                has_text = text_key.str.len() >= 20
                grouped = frame.loc[has_text].groupby(
                    ["user_id", text_key.loc[has_text]], observed=True
                )["item_id"].transform("nunique")
                repeated = pd.Series(False, index=frame.index)
                repeated.loc[grouped.index] = grouped > 1
                mark(repeated, "repeated_text_across_items")

        if "event_time" in frame.columns and "user_id" in frame.columns:
            ordered = frame.sort_values(["user_id", "event_time"], kind="stable")
            gaps = ordered.groupby("user_id", observed=True)["event_time"].diff()
            burst = pd.Series(False, index=frame.index)
            burst.loc[ordered.index] = gaps < pd.Timedelta(minutes=1)
            mark(burst, "burst_under_60s")

        if "rating" in frame.columns:
            per_user = frame.groupby("user_id", observed=True)["rating"]
            counts = per_user.transform("count")
            spread = per_user.transform("std")
            mark((counts >= 5) & (spread.fillna(0) == 0), "zero_rating_variance")

        frame["outlier_reasons"] = pd.Series(
            {index: ";".join(labels) for index, labels in reasons.items()},
            dtype="string",
        ).reindex(frame.index).fillna("")
        flag_column = str(self.rules["outliers"].get("flag_column", "outlier_flag"))
        frame[flag_column] = frame["outlier_reasons"].str.len() > 0

        action = str(self.rules["outliers"].get("action", "flag"))
        if action != "flag":
            self.report.notes.append(
                f"cleaning.outliers.action is '{action}', but this pipeline only ever "
                f"flags. Outlier removal is deliberately not implemented."
            )
        self.report.record(
            "flag_suspicious", before=len(frame), after=len(frame),
            detail=f"{int(frame[flag_column].sum()):,} rows flagged, 0 removed",
        )
        return frame

    def _k_core(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Iteratively drop users/items below the interaction threshold.

        Repeats until a pass changes nothing, because removing a sparse item can
        push a user below the user threshold and vice versa.
        """
        min_user = int(self.k_core["min_interactions_per_user"])
        min_item = int(self.k_core["min_interactions_per_item"])
        max_passes = int(self.k_core.get("max_passes", 10))
        if min_user <= 1 and min_item <= 1:
            self.report.notes.append("k-core filtering disabled by config.")
            return frame

        start = len(frame)
        passes = 0
        for iteration in range(1, max_passes + 1):
            passes = iteration
            before = len(frame)
            item_counts = frame["item_id"].value_counts()
            frame = frame[frame["item_id"].isin(item_counts[item_counts >= min_item].index)]
            user_counts = frame["user_id"].value_counts()
            frame = frame[frame["user_id"].isin(user_counts[user_counts >= min_user].index)]
            if len(frame) == before:
                logger.info("  k-core converged after %d pass(es)", iteration)
                break
        else:
            # The cascade is monotone, so it always terminates; hitting the cap
            # means the cap was too low, and the output would silently violate
            # the very invariant this step exists to establish. Never ship that.
            raise RuntimeError(
                f"k-core did not converge in {max_passes} passes ({len(frame):,} rows "
                f"remaining). The result would violate the {min_user}-user / "
                f"{min_item}-item guarantee. Raise k_core.max_passes in data_config.yaml."
            )

        if len(frame):
            worst_user = int(frame["user_id"].value_counts().min())
            worst_item = int(frame["item_id"].value_counts().min())
            if worst_user < min_user or worst_item < min_item:
                raise RuntimeError(
                    f"k-core post-condition violated: min user degree {worst_user} "
                    f"(want >= {min_user}), min item degree {worst_item} (want >= {min_item})."
                )

        self.report.record(
            "k_core_filter", before=start, after=len(frame),
            detail=f"users >= {min_user}, items >= {min_item} interactions, "
                   f"converged in {passes} pass(es)",
        )
        self.report.notes.append(
            f"k-core ({min_user}-user / {min_item}-item) uses interaction counts over the "
            f"full timeline, including the future test period. This is standard practice "
            f"in recommender-systems evaluation but is a mild selection leakage; it is "
            f"disclosed rather than hidden. Set both thresholds to 1 to disable."
        )
        self.report.notes.append(
            "k-core removes the extreme long tail. Coverage and novelty metrics "
            "(Module 19) are therefore computed over the filtered catalogue, not the "
            "full Amazon catalogue, and must be reported as such."
        )
        return frame

    # -- products ----------------------------------------------------------
    def _clean_products(self, products: pd.DataFrame, keep_items: set[str]) -> pd.DataFrame:
        before = len(products)
        products["item_id"] = products["item_id"].astype("string").str.strip().str.upper()
        products = products[products["item_id"].isin(keep_items)]
        products = products.drop_duplicates(subset=["item_id"], keep="first")
        logger.info("  products: %s -> %s after restricting to retained items",
                    f"{before:,}", f"{len(products):,}")

        if "brand" in products.columns:
            products["brand"] = self._canonicalise(products["brand"])
            logger.info("  brands: %s distinct after canonicalisation",
                        f"{products['brand'].nunique():,}")

        if "main_category" in products.columns:
            products["main_category"] = self._canonicalise(products["main_category"])

        if "categories" in products.columns:
            products["categories"] = products["categories"].apply(
                lambda values: [self._slug(v) for v in values if self._slug(v)]
                if isinstance(values, list) else []
            )
            empty = int(products["categories"].apply(len).eq(0).sum())
            self.report.notes.append(
                f"`categories` is empty for {empty:,} of {len(products):,} products "
                f"({100 * empty / max(1, len(products)):.1f}%). Category nodes are built "
                f"primarily from `main_category`."
            )

        if "price" in products.columns:
            products["price"] = pd.to_numeric(products["price"], errors="coerce").astype("Float32")
            # A price of 0 is not a price; treat as missing but keep it visible.
            products.loc[products["price"] <= 0, "price"] = pd.NA
            products["price_missing"] = products["price"].isna()
            self.report.notes.append(
                f"`price` is missing for {int(products['price_missing'].sum()):,} of "
                f"{len(products):,} products ({100 * float(products['price_missing'].mean()):.1f}%). "
                f"It is never imputed; a `price_missing` indicator is carried instead."
            )
        return products.reset_index(drop=True)

    @staticmethod
    def _slug(value: Any) -> str:
        """Normalisation key: NFKD-folded, lower-cased, alphanumeric only."""
        if not isinstance(value, str) or not value.strip():
            return ""
        folded = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
        return _NON_ALNUM.sub(" ", folded.lower()).strip()

    def _canonicalise(self, series: pd.Series) -> pd.Series:
        """Collapse surface variants ("SONY", "sony ", "Sony") onto one label.

        The canonical label is the most frequent original spelling in the data,
        which keeps output human-readable while making the join key stable.
        """
        slugs = series.map(self._slug)
        surface: defaultdict[str, Counter] = defaultdict(Counter)
        for slug, original in zip(slugs, series):
            if slug and isinstance(original, str):
                surface[slug][original.strip()] += 1
        canonical = {slug: counts.most_common(1)[0][0] for slug, counts in surface.items()}
        merged = sum(1 for counts in surface.values() if len(counts) > 1)
        if merged:
            logger.info("  merged %s label(s) with multiple spellings", f"{merged:,}")
        return slugs.map(lambda s: canonical.get(s)).astype("string")

    # -- reporting ---------------------------------------------------------
    def _count_flags(self, frame: pd.DataFrame) -> None:
        if "outlier_reasons" not in frame.columns:
            return
        counter: Counter[str] = Counter()
        for value in frame["outlier_reasons"].dropna():
            for label in str(value).split(";"):
                if label:
                    counter[label] += 1
        self.report.flags = dict(counter)

    def save_report(self, directory: str | Path | None = None) -> tuple[Path, Path]:
        """Write the cleaning report as JSON and Markdown."""
        target = resolve_path(directory or self.config.require("paths.tables"))
        target.mkdir(parents=True, exist_ok=True)
        json_path = target / "cleaning_report.json"
        md_path = target / "cleaning_report.md"
        json_path.write_text(json.dumps(self.report.to_dict(), indent=2), encoding="utf-8")
        md_path.write_text(self.report.to_markdown(), encoding="utf-8")
        logger.info("Cleaning report written to %s and %s", json_path, md_path)
        return json_path, md_path


def clean_dataset(config: Config, loaded: LoadResult) -> tuple[CleanResult, DataCleaner]:
    """Convenience wrapper returning both the result and the cleaner."""
    cleaner = DataCleaner(config)
    return cleaner.clean(loaded), cleaner
