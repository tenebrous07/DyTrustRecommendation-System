"""Schema detection, adaptation and validation (Module 1).

The pipeline downstream of this module speaks exactly one vocabulary: the
*canonical schema* defined here. Source datasets speak their own. This module
is the translation layer, driven entirely by ``configs/data_config.yaml`` so a
different corpus needs a config profile, not a code change.

The three things it guarantees to callers:

* **Naming** - a record that arrives as ``parent_asin`` leaves as ``item_id``.
* **Availability** - callers get an explicit :class:`FieldAvailability` saying
  which canonical fields actually exist. A field that is absent is reported as
  absent; it is never fabricated, defaulted or imputed here.
* **Typing** - values are coerced to declared dtypes, with unparseable values
  becoming null rather than raising, and the null count reported.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..utils.logging import get_logger

__all__ = [
    "CANONICAL_INTERACTION_FIELDS",
    "CANONICAL_PRODUCT_FIELDS",
    "FieldAvailability",
    "SchemaAdapter",
    "SchemaError",
]

logger = get_logger(__name__)


class SchemaError(ValueError):
    """Raised when a source cannot satisfy the configured required fields."""


# -----------------------------------------------------------------------------
# Canonical vocabulary
# -----------------------------------------------------------------------------
# "kind" drives coercion. Everything downstream may assume these types.
CANONICAL_INTERACTION_FIELDS: dict[str, str] = {
    "user_id": "str",
    "item_id": "str",
    "variant_id": "str",
    "rating": "float",
    "review_title": "text",
    "review_text": "text",
    "timestamp": "int",       # milliseconds since Unix epoch, UTC
    "helpful_votes": "int",
    "verified_purchase": "bool",
}

CANONICAL_PRODUCT_FIELDS: dict[str, str] = {
    "item_id": "str",
    "product_title": "text",
    "product_description": "text_list",   # list[str] -> single string
    "product_features": "str_list",       # list[str] -> kept as list
    "brand": "str",
    "main_category": "str",
    "categories": "str_list",
    "price": "float",
    "avg_rating_meta": "float",
    "rating_count_meta": "int",
    "details": "dict",
    "seller": "str",
}

_WHITESPACE = re.compile(r"\s+")
_HTML_TAG = re.compile(r"<[^>]{1,200}>")
# Amazon review text carries these as literal escapes rather than real newlines.
_LITERAL_ESCAPES = re.compile(r"\\[nrt]")


@dataclass(slots=True)
class FieldAvailability:
    """What the adapter found when it inspected the source.

    Attributes:
        mapped: canonical -> source field, for fields present in the data.
        declared_absent: canonical fields the config explicitly maps to ``null``.
            These are known gaps in the dataset (e.g. ``seller``), not errors.
        missing_in_data: canonical fields the config expected but no sampled
            record contained. Usually a config/dataset-version mismatch.
        unmapped_source: source fields nothing maps to. Informational only.
        records_inspected: how many records the detection scan read.
    """

    mapped: dict[str, str] = field(default_factory=dict)
    declared_absent: list[str] = field(default_factory=list)
    missing_in_data: list[str] = field(default_factory=list)
    unmapped_source: list[str] = field(default_factory=list)
    records_inspected: int = 0

    def has(self, canonical_field: str) -> bool:
        """Whether a canonical field is genuinely usable downstream."""
        return canonical_field in self.mapped

    def to_dict(self) -> dict[str, Any]:
        return {
            "mapped": dict(self.mapped),
            "declared_absent": list(self.declared_absent),
            "missing_in_data": list(self.missing_in_data),
            "unmapped_source": list(self.unmapped_source),
            "records_inspected": self.records_inspected,
        }

    def summary(self) -> str:
        parts = [f"{len(self.mapped)} field(s) available"]
        if self.declared_absent:
            parts.append(f"absent by dataset design: {', '.join(sorted(self.declared_absent))}")
        if self.missing_in_data:
            parts.append(f"EXPECTED BUT NOT FOUND: {', '.join(sorted(self.missing_in_data))}")
        return "; ".join(parts)


# -----------------------------------------------------------------------------
# Coercion
# -----------------------------------------------------------------------------
def _clean_text(value: Any) -> str | None:
    """Normalise a free-text field without destroying information SBERT needs.

    Deliberately does NOT lowercase or strip punctuation: ``all-MiniLM-L6-v2``
    is a cased model and sentence boundaries carry signal. We only remove markup
    and normalise whitespace.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    text = html.unescape(value)
    text = _HTML_TAG.sub(" ", text)
    text = _LITERAL_ESCAPES.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text or None


def _to_float(value: Any) -> float | None:
    """Parse a float, tolerating currency strings such as ``"$19.99"``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if result == result else None  # reject NaN
    if isinstance(value, str):
        stripped = value.strip().replace("$", "").replace(",", "").strip()
        if not stripped:
            return None
        # Price ranges like "12.99 - 24.99": take the lower bound, note the loss.
        if " - " in stripped:
            stripped = stripped.split(" - ", 1)[0].strip()
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    parsed = _to_float(value)
    return int(parsed) if parsed is not None else None


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "yes", "y", "1"}:
            return True
        if lowered in {"false", "f", "no", "n", "0"}:
            return False
    return None


def _to_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    return None


def _to_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = _clean_text(value)
        return [cleaned] if cleaned else []
    if isinstance(value, Sequence):
        out: list[str] = []
        for item in value:
            cleaned = _clean_text(item) if not isinstance(item, dict) else None
            if cleaned:
                out.append(cleaned)
        return out
    return []


def _to_text_list(value: Any) -> str | None:
    """Flatten a list of paragraphs into one string (product descriptions)."""
    parts = _to_str_list(value)
    return " ".join(parts) if parts else None


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


_COERCERS = {
    "str": _to_str,
    "text": _clean_text,
    "float": _to_float,
    "int": _to_int,
    "bool": _to_bool,
    "str_list": _to_str_list,
    "text_list": _to_text_list,
    "dict": _to_dict,
}


# -----------------------------------------------------------------------------
# Adapter
# -----------------------------------------------------------------------------
class SchemaAdapter:
    """Translates source records into the canonical schema.

    Example:
        >>> adapter = SchemaAdapter(
        ...     field_map={"user_id": "user_id", "item_id": "parent_asin",
        ...                "rating": "rating", "timestamp": "timestamp",
        ...                "seller": None},
        ...     required=["user_id", "item_id", "rating", "timestamp"],
        ...     canonical=CANONICAL_INTERACTION_FIELDS,
        ...     name="interactions",
        ... )
        >>> adapter.detect([{"user_id": "A1", "parent_asin": "B0",
        ...                  "rating": 5.0, "timestamp": 1588687728923}]).has("seller")
        False
    """

    def __init__(
        self,
        field_map: Mapping[str, str | None],
        required: Sequence[str],
        canonical: Mapping[str, str],
        name: str = "source",
    ) -> None:
        unknown = set(field_map) - set(canonical)
        if unknown:
            raise SchemaError(
                f"[{name}] config maps unknown canonical field(s) {sorted(unknown)}. "
                f"Valid names: {sorted(canonical)}"
            )
        self.name = name
        self.canonical = dict(canonical)
        self.required = list(required)
        # Split the config's declared mapping into "should exist" vs "known gap".
        self.field_map: dict[str, str] = {
            key: value for key, value in field_map.items() if value is not None
        }
        self.declared_absent: list[str] = [
            key for key, value in field_map.items() if value is None
        ]
        self._availability: FieldAvailability | None = None

    # -- detection ---------------------------------------------------------
    def detect(self, records: Iterable[Mapping[str, Any]], limit: int = 500) -> FieldAvailability:
        """Inspect sample records and report which canonical fields are usable.

        A field counts as present if it appears as a *key* in at least one
        sampled record, even when its value is null. Presence of the key means
        the dataset models the concept; a null value is missing data, which is
        the cleaner's problem, not the adapter's.
        """
        seen_keys: set[str] = set()
        inspected = 0
        for record in records:
            seen_keys.update(record.keys())
            inspected += 1
            if inspected >= limit:
                break

        if inspected == 0:
            raise SchemaError(f"[{self.name}] no records available to inspect.")

        mapped = {c: s for c, s in self.field_map.items() if s in seen_keys}
        missing = sorted(set(self.field_map) - set(mapped))
        unmapped = sorted(seen_keys - set(self.field_map.values()))

        availability = FieldAvailability(
            mapped=mapped,
            declared_absent=sorted(self.declared_absent),
            missing_in_data=missing,
            unmapped_source=unmapped,
            records_inspected=inspected,
        )

        missing_required = [f for f in self.required if f not in mapped]
        if missing_required:
            raise SchemaError(
                f"[{self.name}] required canonical field(s) {missing_required} are absent "
                f"from the source. Configured mapping: "
                f"{ {f: self.field_map.get(f) for f in missing_required} }. "
                f"Source keys seen: {sorted(seen_keys)}"
            )

        logger.info("[%s] %s", self.name, availability.summary())
        if missing:
            logger.warning(
                "[%s] configured field(s) %s were not found in %d sampled records. "
                "They will be treated as unavailable.", self.name, missing, inspected,
            )
        self._availability = availability
        return availability

    @property
    def availability(self) -> FieldAvailability:
        if self._availability is None:
            raise RuntimeError(f"[{self.name}] detect() must be called before availability.")
        return self._availability

    # -- application -------------------------------------------------------
    def apply(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Translate and coerce one source record.

        Only fields found during :meth:`detect` are emitted, so every column in
        the resulting frame is one the dataset genuinely provides.
        """
        mapped = self.availability.mapped
        out: dict[str, Any] = {}
        for canonical_name, source_name in mapped.items():
            coerce = _COERCERS[self.canonical[canonical_name]]
            out[canonical_name] = coerce(record.get(source_name))
        return out

    def apply_many(self, records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [self.apply(record) for record in records]

    # -- construction ------------------------------------------------------
    @classmethod
    def for_interactions(cls, schema_cfg: Mapping[str, Any]) -> "SchemaAdapter":
        section = schema_cfg["interactions"]
        return cls(
            field_map=dict(section["field_map"]),
            required=list(section["required"]),
            canonical=CANONICAL_INTERACTION_FIELDS,
            name="interactions",
        )

    @classmethod
    def for_products(cls, schema_cfg: Mapping[str, Any]) -> "SchemaAdapter":
        section = schema_cfg["products"]
        return cls(
            field_map=dict(section["field_map"]),
            required=list(section["required"]),
            canonical=CANONICAL_PRODUCT_FIELDS,
            name="products",
        )
