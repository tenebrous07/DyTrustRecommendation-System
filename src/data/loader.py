"""Dataset ingestion (Module 1).

Responsibilities, in order: acquire the raw file without mutating it, convert
it to CSV once (see :mod:`src.data.json_to_csv`), iterate that CSV in constant
memory, translate records through :mod:`src.data.validator`, apply a declared
sampling strategy, and emit an auditable manifest describing exactly what was
loaded and how it was biased.

Design constraints this module answers:

* **Raw data is immutable.** Downloads land in ``data/raw/`` and are never
  rewritten. Every derived artefact, including the converted CSV, goes to
  ``data/interim/``.
* **Work off CSV, not raw JSON.** The public corpus ships as JSONL, but
  everything past the first read - schema detection, sampling, k-core - reads
  the plain CSV produced by :func:`src.data.json_to_csv.convert_json_to_csv`.
  It is easier to inspect and quick to load with ``pandas.read_csv``.
* **Files can exceed disk.** The Electronics tier is 25.95 GB against 30 GB
  free, so ``materialise: false`` skips conversion entirely and reads JSON
  straight off an HTTP range request with a byte budget, never storing it.
* **Sampling bias must be declared, not hidden.** Each strategy records what it
  does to representativeness in the manifest.
"""

from __future__ import annotations

import gzip
import json
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import pandas as pd
import requests

from ..utils.config import Config, resolve_path
from ..utils.logging import get_logger, log_stage
from ..utils.seed import hash_file
from .json_to_csv import convert_json_to_csv
from .validator import FieldAvailability, SchemaAdapter, SchemaError

__all__ = ["LoadResult", "DatasetLoader", "load_dataset", "acquire"]

logger = get_logger(__name__)

_DOWNLOAD_CHUNK = 1 << 22          # 4 MiB
_DETECT_SAMPLE = 200               # records read for schema detection
_PROGRESS_EVERY = 500_000          # records between progress logs


# =============================================================================
# Result container
# =============================================================================
@dataclass(slots=True)
class LoadResult:
    """Everything Phase 1 hands to Phase 2.

    Attributes:
        interactions: One row per review/interaction, canonical column names.
        products: Product metadata restricted to items present in
            ``interactions``. ``None`` if no metadata source was configured.
        interaction_fields: Which canonical interaction fields are real.
        product_fields: Which canonical product fields are real.
        manifest: Provenance - source URLs, byte counts, digests, sampling
            strategy and its declared bias, and row counts at each step.
    """

    interactions: pd.DataFrame
    products: pd.DataFrame | None
    interaction_fields: FieldAvailability
    product_fields: FieldAvailability | None
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def n_interactions(self) -> int:
        return len(self.interactions)

    @property
    def n_users(self) -> int:
        return int(self.interactions["user_id"].nunique())

    @property
    def n_items(self) -> int:
        return int(self.interactions["item_id"].nunique())

    def describe(self) -> str:
        lines = [
            f"interactions : {self.n_interactions:,}",
            f"users        : {self.n_users:,}",
            f"items        : {self.n_items:,}",
            f"density      : {self.n_interactions / max(1, self.n_users * self.n_items):.3e}",
        ]
        if self.products is not None:
            lines.append(f"products meta: {len(self.products):,}")
        return "\n".join(lines)


# =============================================================================
# Acquisition
# =============================================================================
def _remote_size(url: str, *, timeout: int = 30) -> int | None:
    """Content-Length of a remote file, following redirects."""
    try:
        response = requests.head(url, allow_redirects=True, timeout=timeout)
        response.raise_for_status()
        length = response.headers.get("Content-Length")
        return int(length) if length else None
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Could not determine remote size for %s (%s)", url, exc)
        return None


def download_with_resume(url: str, destination: Path, *, timeout: int = 60) -> Path:
    """Download ``url`` to ``destination``, resuming a partial file if present.

    A ``.part`` file is used until the transfer completes, so an interrupted run
    can never leave a truncated file that looks valid.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = _remote_size(url)

    if destination.exists():
        actual = destination.stat().st_size
        if expected is None or actual == expected:
            logger.info("Using cached raw file %s (%.2f GB)", destination, actual / 2**30)
            return destination
        logger.warning(
            "Cached %s is %.2f GB but remote is %.2f GB; re-downloading.",
            destination.name, actual / 2**30, expected / 2**30,
        )
        destination.unlink()

    partial = destination.with_suffix(destination.suffix + ".part")
    already = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={already}-"} if already else {}
    if already:
        logger.info("Resuming %s from %.2f GB", destination.name, already / 2**30)

    started = time.perf_counter()
    with requests.get(url, stream=True, headers=headers, timeout=timeout) as response:
        if already and response.status_code == 200:
            # Server ignored the Range header; restart cleanly.
            logger.warning("Server ignored Range request; restarting download.")
            already = 0
            partial.unlink(missing_ok=True)
        response.raise_for_status()

        mode = "ab" if already else "wb"
        downloaded = already
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if expected and downloaded % (_DOWNLOAD_CHUNK * 64) < _DOWNLOAD_CHUNK:
                    logger.info(
                        "  %s  %.2f / %.2f GB (%.1f%%)", destination.name,
                        downloaded / 2**30, expected / 2**30, 100 * downloaded / expected,
                    )

    if expected is not None and partial.stat().st_size != expected:
        raise IOError(
            f"Download of {url} finished at {partial.stat().st_size} bytes, "
            f"expected {expected}. Left partial file at {partial}."
        )

    partial.rename(destination)
    elapsed = time.perf_counter() - started
    logger.info(
        "Downloaded %s (%.2f GB) in %.0fs", destination.name,
        destination.stat().st_size / 2**30, elapsed,
    )
    return destination


def acquire(source: str, raw_dir: Path) -> Path:
    """Resolve a configured source to a local file.

    A source may be an ``http(s)`` URL, which is downloaded into ``raw_dir``, or
    a filesystem path, which is used in place. Supporting both means the same
    code path serves the public corpus, an offline mirror, and the synthetic
    fixtures used by the tests - no network monkey-patching required.
    """
    if source.startswith(("http://", "https://")):
        return download_with_resume(source, raw_dir / Path(source).name)

    path = resolve_path(source)
    if not path.is_file():
        raise FileNotFoundError(
            f"Configured source '{source}' is neither an http(s) URL nor an existing file "
            f"(resolved to {path})."
        )
    logger.info("Using local source %s (%.2f MB)", path, path.stat().st_size / 2**20)
    return path


def _relative_to_root(path: Path) -> str:
    """Project-relative path when possible, absolute otherwise (e.g. tmp dirs)."""
    try:
        return str(path.relative_to(resolve_path(".")))
    except ValueError:
        return str(path)


# =============================================================================
# Record iteration
# =============================================================================
def _open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _decode_json_cell(value: Any) -> Any:
    """Undo the JSON-encoding :func:`src.data.json_to_csv.convert_json_to_csv`
    applies to list/dict columns before they can be written to a CSV cell.

    Anything that isn't a string starting with ``[`` or ``{`` is left alone,
    so this is safe to run on every value read back from a CSV.
    """
    if isinstance(value, str) and value[:1] in "[{":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def iter_local_records(path: Path, *, max_records: int | None = None) -> Iterator[dict[str, Any]]:
    """Stream records from a local file in constant memory.

    Supports ``.jsonl``, ``.json`` (array or lines), ``.csv``, ``.parquet``, and
    any of the text formats additionally gzipped. CSV is the common case now
    that ingestion converts the raw JSON to CSV first (see
    :mod:`src.data.json_to_csv`); list/dict fields that were JSON-encoded to
    fit in a cell are decoded back out here.
    """
    stem_suffix = Path(path.name.removesuffix(".gz")).suffix.lower()

    if stem_suffix == ".parquet":
        frame = pd.read_parquet(path)
        for count, record in enumerate(frame.to_dict("records")):
            if max_records is not None and count >= max_records:
                return
            yield record
        return

    if stem_suffix == ".csv":
        reader = pd.read_csv(path, chunksize=100_000, low_memory=False)
        emitted = 0
        for chunk in reader:
            for record in chunk.to_dict("records"):
                if max_records is not None and emitted >= max_records:
                    return
                emitted += 1
                yield {key: _decode_json_cell(value) for key, value in record.items()}
        return

    # JSON / JSONL. A leading '[' means a single JSON array document.
    with _open_maybe_gzip(path) as handle:
        first = handle.readline()
        if first.lstrip().startswith("["):
            handle.seek(0)
            payload = json.load(handle)
            for count, record in enumerate(payload):
                if max_records is not None and count >= max_records:
                    return
                yield record
            return

        emitted = 0
        for line_no, line in enumerate((first, *handle), start=1):
            line = line.strip()
            if not line:
                continue
            if max_records is not None and emitted >= max_records:
                return
            try:
                yield json.loads(line)
                emitted += 1
            except json.JSONDecodeError:
                logger.debug("Skipping malformed JSON at %s:%d", path.name, line_no)
            if emitted and emitted % _PROGRESS_EVERY == 0:
                logger.info("  read %s records from %s", f"{emitted:,}", path.name)


def iter_remote_records(
    url: str, *, byte_budget: int, max_records: int | None = None, timeout: int = 60
) -> Iterator[dict[str, Any]]:
    """Stream JSONL records over HTTP without storing the file.

    Reads at most ``byte_budget`` bytes. Used for the Electronics tier, whose
    25.95 GB exceeds available disk. The final partial line is discarded.
    """
    logger.info("Streaming %s (budget %.2f GB, file not stored)", url, byte_budget / 2**30)
    consumed = 0
    emitted = 0
    remainder = b""

    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK):
            if not chunk:
                continue
            consumed += len(chunk)
            buffer = remainder + chunk
            *lines, remainder = buffer.split(b"\n")
            for raw in lines:
                if not raw.strip():
                    continue
                try:
                    yield json.loads(raw)
                    emitted += 1
                except json.JSONDecodeError:
                    continue
                if max_records is not None and emitted >= max_records:
                    logger.info("Stream stopped at record cap (%s records).", f"{emitted:,}")
                    return
            if consumed >= byte_budget:
                logger.info(
                    "Stream stopped at byte budget (%.2f GB, %s records).",
                    consumed / 2**30, f"{emitted:,}",
                )
                return


# =============================================================================
# Sampling
# =============================================================================
_BIAS_NOTES = {
    "head": (
        "BIASED. Takes the first N records in file order. Amazon Reviews 2023 "
        "files are not shuffled, so a prefix over-represents whichever users or "
        "items appear early. Do not describe results from this strategy as "
        "representative of the category."
    ),
    "reservoir": (
        "Uniform random sample of interactions (Algorithm R). Unbiased over "
        "interactions, but it fragments user histories: a sampled user keeps "
        "only a random subset of their reviews, which distorts sequence length "
        "and recency signals used by temporal modelling."
    ),
    "user_core": (
        "Uniform random sample of USERS, keeping every interaction of each "
        "sampled user. Unbiased over users and preserves complete per-user "
        "histories, which temporal weighting and sequence analysis depend on. "
        "Item popularity within the sample remains proportional to the full "
        "category. This is the default for graph work."
    ),
}

def scan_user_counts(path: Path, source_field: str) -> Counter[str]:
    """First pass: count interactions per user.

    Just reads every record and tallies ``source_field``. This is the
    straightforward way to do it now that ingestion works off a converted CSV
    (see :mod:`src.data.json_to_csv`) rather than parsing raw JSON by hand.
    """
    counts: Counter[str] = Counter()
    for record in iter_local_records(path):
        value = record.get(source_field)
        if value is not None:
            counts[str(value)] += 1

    logger.info(
        "Scanned %s: %s interactions across %s users",
        path.name, f"{sum(counts.values()):,}", f"{len(counts):,}",
    )
    return counts


def select_users(
    counts: Mapping[str, int],
    *,
    max_users: int | None,
    min_interactions: int,
    rng: random.Random,
    max_interactions: int | None = None,
) -> set[str]:
    """Choose the user sample: filter by activity, then sample uniformly.

    Pre-filtering on ``min_interactions`` before sampling means the budget is
    spent on users who can actually support a train/val/test split, rather than
    on single-review users that k-core would discard anyway.

    ``max_interactions`` is the crucial argument. A row cap must be honoured by
    admitting *fewer users*, never by cutting a user off part-way through their
    history: truncating mid-stream would hand later users artificially short
    histories, which is precisely the signal temporal modelling reads. So the
    cap is spent user-by-user, in random order, and a user is either fully in
    or fully out.

    Returns:
        The selected user ids.
    """
    eligible = [user for user, count in counts.items() if count >= min_interactions]
    logger.info(
        "%s of %s users have >= %d interactions",
        f"{len(eligible):,}", f"{len(counts):,}", min_interactions,
    )
    if not eligible:
        raise SchemaError(
            f"No users have >= {min_interactions} interactions. "
            f"Lower k_core.min_interactions_per_user in data_config.yaml."
        )

    eligible.sort()                       # deterministic order before shuffling
    rng.shuffle(eligible)
    if max_users is not None and len(eligible) > max_users:
        eligible = eligible[:max_users]

    if max_interactions is None:
        return set(eligible)

    chosen: set[str] = set()
    budget = 0
    for user in eligible:
        cost = counts[user]
        if budget + cost > max_interactions:
            continue          # skip, do not truncate; a smaller user may still fit
        chosen.add(user)
        budget += cost
    if not chosen:
        raise SchemaError(
            f"max_reviews={max_interactions} is smaller than the history of every "
            f"eligible user (smallest is {min(counts[u] for u in eligible)}). "
            f"Raise limits.max_reviews."
        )
    logger.info(
        "Interaction budget %s: admitted %s of %s candidate users, %s interactions "
        "(complete histories)",
        f"{max_interactions:,}", f"{len(chosen):,}", f"{len(eligible):,}", f"{budget:,}",
    )
    return chosen


# =============================================================================
# Loader
# =============================================================================
class DatasetLoader:
    """Config-driven ingestion for one tier of the configured dataset."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.tier_name: str = config.require("active_tier")
        tiers = config.require("tiers")
        if self.tier_name not in tiers:
            raise KeyError(
                f"active_tier '{self.tier_name}' is not defined. Available: {sorted(tiers)}"
            )
        self.tier: Config = tiers[self.tier_name]
        self.sampling: Config = config.require("sampling")
        self.raw_dir = resolve_path(config.require("paths.raw"))
        self.interim_dir = resolve_path(config.require("paths.interim"))
        self.rng = random.Random(int(self.sampling.get("seed", 42)))

        self.interaction_adapter = SchemaAdapter.for_interactions(config.require("schema"))
        self.product_adapter = SchemaAdapter.for_products(config.require("schema"))

    # -- public ------------------------------------------------------------
    def load(self) -> LoadResult:
        """Run ingestion for the active tier."""
        category = self.tier["category"]
        logger.info("=" * 72)
        logger.info("Ingesting tier '%s' (category=%s)", self.tier_name, category)
        logger.info("=" * 72)

        manifest: dict[str, Any] = {
            "tier": self.tier_name,
            "category": category,
            "dataset": self.config.get("dataset.name", "unknown"),
            "sampling": {
                "strategy": self.sampling.get("strategy", "head"),
                "seed": self.sampling.get("seed", 42),
                "bias_declaration": _BIAS_NOTES.get(
                    str(self.sampling.get("strategy", "head")), "unknown strategy"
                ),
            },
            "limits": dict(self.tier["limits"].to_dict()),
            "k_core_declared": dict(self.tier["k_core"].to_dict()),
            "sources": {},
        }

        with log_stage(logger, "Load interactions") as stage:
            interactions, i_fields = self._load_interactions(manifest)
            stage["rows"] = f"{len(interactions):,}"

        with log_stage(logger, "Load product metadata") as stage:
            products, p_fields = self._load_products(
                needed_items=set(interactions["item_id"].dropna().unique()),
                manifest=manifest,
            )
            stage["rows"] = f"{len(products):,}" if products is not None else "none"

        manifest["counts"] = {
            "interactions": len(interactions),
            "users": int(interactions["user_id"].nunique()),
            "items": int(interactions["item_id"].nunique()),
            "products_with_metadata": 0 if products is None else len(products),
        }
        manifest["field_availability"] = {
            "interactions": i_fields.to_dict(),
            "products": p_fields.to_dict() if p_fields else None,
        }

        result = LoadResult(interactions, products, i_fields, p_fields, manifest)
        logger.info("Ingestion complete:\n%s", result.describe())
        return result

    # -- CSV conversion ------------------------------------------------------
    def _ensure_csv(self, path: Path, stem: str) -> Path:
        """Convert a raw JSON/JSONL source to CSV once; reuse it after that.

        A file that is already tabular (``.csv`` or ``.parquet``) is used as
        is. Everything downstream then reads from the CSV rather than parsing
        JSON directly - that is the "json to csv, then work on csv" change.
        """
        real_suffix = Path(path.name.removesuffix(".gz")).suffix.lower()
        if real_suffix in (".csv", ".parquet"):
            return path
        csv_path = self.interim_dir / f"{stem}_{self.tier_name}.csv"
        return convert_json_to_csv(path, csv_path)

    # -- interactions ------------------------------------------------------
    def _load_interactions(self, manifest: dict[str, Any]) -> tuple[pd.DataFrame, FieldAvailability]:
        url = self.tier["reviews_url"]
        materialise = bool(self.tier.get("materialise", True))
        limits = self.tier["limits"]
        max_reviews = limits.get("max_reviews")
        max_users = limits.get("max_users")
        min_per_user = int(self.tier["k_core"]["min_interactions_per_user"])
        strategy = str(self.sampling.get("strategy", "head"))

        if materialise:
            path = acquire(url, self.raw_dir)
            manifest["sources"]["reviews"] = {
                "url": url,
                "local_path": _relative_to_root(path),
                "bytes": path.stat().st_size,
                "sha256_prefix": hash_file(path, max_bytes=64 << 20),
                "materialised": True,
            }
            csv_path = self._ensure_csv(path, "reviews")
            manifest["sources"]["reviews"]["csv_path"] = _relative_to_root(csv_path)
            detect_records = list(iter_local_records(csv_path, max_records=_DETECT_SAMPLE))
            fields = self.interaction_adapter.detect(detect_records)
            records = self._sample_local(csv_path, strategy, max_reviews, max_users, min_per_user)
        else:
            budget = int(self.tier.get("stream_byte_budget", 1 << 31))
            manifest["sources"]["reviews"] = {
                "url": url, "local_path": None,
                "stream_byte_budget": budget, "materialised": False,
            }
            if strategy == "user_core":
                logger.warning(
                    "Strategy 'user_core' needs two passes and the tier is not "
                    "materialised. Falling back to 'head' over the streamed prefix."
                )
                manifest["sampling"]["strategy"] = "head"
                manifest["sampling"]["fallback_reason"] = (
                    "user_core requires two passes over a local file; this tier streams."
                )
                manifest["sampling"]["bias_declaration"] = _BIAS_NOTES["head"]
            detect_records = []
            stream = iter_remote_records(url, byte_budget=budget, max_records=max_reviews)
            for record in stream:
                detect_records.append(record)
                if len(detect_records) >= _DETECT_SAMPLE:
                    break
            fields = self.interaction_adapter.detect(detect_records)
            records = self._chain(detect_records, stream, max_reviews)

        rows = [self.interaction_adapter.apply(record) for record in records]
        if not rows:
            raise SchemaError("Ingestion produced zero interactions. Check tier limits and URL.")

        frame = pd.DataFrame(rows)
        frame = self._apply_dtypes(frame)
        self._check_product_budget(frame, limits.get("max_products"), manifest)
        logger.info(
            "Interactions loaded: %s rows, %s users, %s items",
            f"{len(frame):,}", f"{frame['user_id'].nunique():,}", f"{frame['item_id'].nunique():,}",
        )
        return frame, fields

    @staticmethod
    def _check_product_budget(
        frame: pd.DataFrame, max_products: int | None, manifest: dict[str, Any]
    ) -> None:
        """Report against ``limits.max_products``. Deliberately does not truncate.

        This is a guard rail, not a cap, and the distinction is a design decision
        rather than an omission. The only ways to force the item count down after
        users are chosen are to drop the least popular items - which destroys the
        long tail that the coverage and novelty metrics exist to measure - or to
        drop users, which would undo the complete-history guarantee that
        :func:`select_users` just established. Both cures are worse than the
        disease, so the overrun is reported and left to k-core, which reduces the
        catalogue on a principled criterion instead.
        """
        if max_products is None:
            return
        observed = int(frame["item_id"].nunique())
        manifest["product_budget"] = {
            "max_products": max_products,
            "observed_before_k_core": observed,
            "exceeded": observed > max_products,
            "policy": "report-only; no truncation (see _check_product_budget)",
        }
        if observed > max_products:
            logger.warning(
                "Sampled users span %s items, above limits.max_products=%s. Not truncating: "
                "dropping items by popularity would bias coverage/novelty, and dropping users "
                "would break complete histories. k-core will reduce the catalogue instead.",
                f"{observed:,}", f"{max_products:,}",
            )

    def _sample_local(
        self, path: Path, strategy: str, max_reviews: int | None,
        max_users: int | None, min_per_user: int,
    ) -> Iterable[Mapping[str, Any]]:
        """Dispatch to the configured sampling strategy over a local file."""
        if strategy == "head":
            logger.warning("Sampling strategy 'head' is biased. %s", _BIAS_NOTES["head"])
            return iter_local_records(path, max_records=max_reviews)

        if strategy == "reservoir":
            return self._reservoir(path, max_reviews)

        if strategy == "user_core":
            user_field = self.interaction_adapter.availability.mapped["user_id"]
            with log_stage(logger, "Pass 1/2: scan user activity"):
                counts = scan_user_counts(path, user_field)
            keep = select_users(
                counts, max_users=max_users, min_interactions=min_per_user,
                rng=self.rng, max_interactions=max_reviews,
            )
            logger.info(
                "Selected %s users covering %s interactions",
                f"{len(keep):,}", f"{sum(counts[u] for u in keep):,}",
            )
            # No row cap here on purpose: select_users already sized the sample,
            # and capping again would truncate the last user's history.
            return self._filter_by_user(path, user_field, keep, None)

        raise ValueError(
            f"Unknown sampling.strategy '{strategy}'. Use head|reservoir|user_core."
        )

    def _reservoir(self, path: Path, size: int | None) -> list[Mapping[str, Any]]:
        """Algorithm R: uniform sample of ``size`` records in one pass."""
        if size is None:
            return list(iter_local_records(path))
        reservoir: list[Mapping[str, Any]] = []
        for index, record in enumerate(iter_local_records(path)):
            if index < size:
                reservoir.append(record)
            else:
                slot = self.rng.randint(0, index)
                if slot < size:
                    reservoir[slot] = record
        logger.info("Reservoir sample: %s records", f"{len(reservoir):,}")
        return reservoir

    @staticmethod
    def _filter_by_user(
        path: Path, user_field: str, keep: set[str], max_records: int | None
    ) -> Iterator[Mapping[str, Any]]:
        """Second pass: emit every record belonging to a selected user."""
        emitted = 0
        for record in iter_local_records(path):
            if record.get(user_field) not in keep:
                continue
            yield record
            emitted += 1
            if max_records is not None and emitted >= max_records:
                logger.info("Reached max_reviews cap at %s records.", f"{emitted:,}")
                return

    @staticmethod
    def _chain(
        head: list[Mapping[str, Any]], rest: Iterator[Mapping[str, Any]], cap: int | None
    ) -> Iterator[Mapping[str, Any]]:
        emitted = 0
        for record in head:
            if cap is not None and emitted >= cap:
                return
            yield record
            emitted += 1
        for record in rest:
            if cap is not None and emitted >= cap:
                return
            yield record
            emitted += 1

    # -- products ----------------------------------------------------------
    def _load_products(
        self, needed_items: set[str], manifest: dict[str, Any]
    ) -> tuple[pd.DataFrame | None, FieldAvailability | None]:
        url = self.tier.get("meta_url")
        if not url:
            logger.warning("No meta_url configured for tier '%s'; skipping metadata.", self.tier_name)
            return None, None

        path = acquire(url, self.raw_dir)
        manifest["sources"]["meta"] = {
            "url": url,
            "local_path": _relative_to_root(path),
            "bytes": path.stat().st_size,
            "sha256_prefix": hash_file(path, max_bytes=64 << 20),
            "materialised": True,
        }
        csv_path = self._ensure_csv(path, "meta")
        manifest["sources"]["meta"]["csv_path"] = _relative_to_root(csv_path)

        detect_records = list(iter_local_records(csv_path, max_records=_DETECT_SAMPLE))
        fields = self.product_adapter.detect(detect_records)
        item_field = fields.mapped["item_id"]

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in iter_local_records(csv_path):
            item_id = record.get(item_field)
            if item_id not in needed_items or item_id in seen:
                continue
            seen.add(item_id)
            rows.append(self.product_adapter.apply(record))
            if len(seen) == len(needed_items):
                break

        coverage = len(rows) / max(1, len(needed_items))
        logger.info(
            "Product metadata: %s of %s items matched (%.1f%% coverage)",
            f"{len(rows):,}", f"{len(needed_items):,}", 100 * coverage,
        )
        if coverage < 0.90:
            logger.warning(
                "Only %.1f%% of interacted items have metadata. Content features "
                "and Brand/Category nodes will be missing for the remainder.", 100 * coverage,
            )
        manifest["metadata_coverage"] = round(coverage, 4)

        if not rows:
            return None, fields
        return pd.DataFrame(rows), fields

    # -- typing ------------------------------------------------------------
    @staticmethod
    def _apply_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
        """Coerce to compact, explicit dtypes and derive ``event_time``.

        Nullable extension dtypes are used throughout so that "missing" stays
        distinguishable from 0 / False - which matters because helpful_votes=0
        and verified_purchase=False are meaningful values, not absences.
        """
        casts = {
            "rating": "Float32",
            "helpful_votes": "Int32",
            "verified_purchase": "boolean",
            "timestamp": "Int64",
        }
        for column, dtype in casts.items():
            if column in frame.columns:
                frame[column] = frame[column].astype(dtype)

        for column in ("user_id", "item_id", "variant_id"):
            if column in frame.columns:
                frame[column] = frame[column].astype("string")

        if "timestamp" in frame.columns:
            frame["event_time"] = pd.to_datetime(
                frame["timestamp"], unit="ms", utc=True, errors="coerce"
            )
        return frame


def load_dataset(config: Config) -> LoadResult:
    """Convenience wrapper: build a :class:`DatasetLoader` and run it."""
    return DatasetLoader(config).load()
