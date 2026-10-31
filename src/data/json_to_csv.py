"""Convert a raw Amazon Reviews JSON/JSONL file to CSV (Module 1a).

The public dataset ships as JSONL (one JSON object per line, optionally
gzipped). Earlier versions of this project read that JSONL straight through,
but a plain CSV is much easier to open, inspect with a spreadsheet, or hand
to ``pandas.read_csv`` quickly, so ingestion now converts once and everything
downstream works off the CSV instead.

A few fields in the product metadata (``details``, ``description``,
``features``, ``categories``) are lists or dicts in the source JSON, which a
CSV cell cannot hold directly. Those get JSON-encoded into a single string on
the way out and decoded back into a real list/dict on the way back in (see
``decode_json_cells`` in :mod:`src.data.loader`), so nothing is lost in the
round trip - it is just stored as text in between.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from ..utils.logging import get_logger

__all__ = ["convert_json_to_csv"]

logger = get_logger(__name__)


def _open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _read_json_records(path: Path) -> list[dict]:
    """Read a .json (array) or .jsonl (one object per line) file."""
    with _open_maybe_gzip(path) as handle:
        first = handle.readline()
        if first.lstrip().startswith("["):
            handle.seek(0)
            return json.load(handle)

        records = []
        for line in (first, *handle):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records


def convert_json_to_csv(json_path: Path, csv_path: Path, *, force: bool = False) -> Path:
    """Convert ``json_path`` to a CSV at ``csv_path``, once.

    If ``csv_path`` already exists and is not older than ``json_path``, the
    existing CSV is reused instead of converting again - re-running the
    pipeline should not redo this every time.
    """
    if csv_path.exists() and not force and csv_path.stat().st_mtime >= json_path.stat().st_mtime:
        logger.info("Using cached CSV %s (already converted from %s)", csv_path.name, json_path.name)
        return csv_path

    logger.info("Converting %s to CSV ...", json_path.name)
    records = _read_json_records(json_path)
    frame = pd.DataFrame(records)

    # Any column holding lists/dicts (product "details", "description", ...)
    # gets JSON-encoded so a CSV cell can hold it without losing structure.
    for column in frame.columns:
        if frame[column].map(lambda value: isinstance(value, (list, dict))).any():
            frame[column] = frame[column].map(
                lambda value: json.dumps(value) if isinstance(value, (list, dict)) else value
            )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    logger.info("Wrote %s rows to %s", f"{len(frame):,}", csv_path.name)
    return csv_path
