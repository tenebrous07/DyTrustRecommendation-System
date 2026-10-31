"""Shared fixtures.

Tests never touch the network or the real corpus. They build a synthetic file
that mirrors the *exact* field names of Amazon Reviews 2023, so the schema
adapter is exercised against the real contract rather than a convenient one.
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import Config  # noqa: E402


def _ms(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp() * 1000)


@pytest.fixture(scope="session")
def review_records() -> list[dict]:
    """Synthetic reviews using the real Amazon Reviews 2023 review schema."""
    rng = random.Random(7)
    records: list[dict] = []

    # 12 well-behaved users x 6 reviews each over 2019-2022.
    for user in range(12):
        for index in range(6):
            records.append({
                "rating": float(rng.choice([1, 2, 3, 4, 5])),
                "title": f"Review title {user}-{index}",
                "text": f"This is review number {index} written by user {user}. It is fine.",
                "images": [],
                "asin": f"B{user:04d}{index:05d}",
                "parent_asin": f"P{index:09d}",
                "user_id": f"USER{user:04d}",
                "timestamp": _ms(2019 + index // 2, 1 + (index * 3) % 12, 1 + index),
                "helpful_vote": rng.randint(0, 20),
                "verified_purchase": bool(rng.getrandbits(1)),
            })

    # One low-activity user, to be removed by k-core.
    records.append({
        "rating": 5.0, "title": "One off", "text": "Only review.", "images": [],
        "asin": "B9999999999", "parent_asin": "P000000000",
        "user_id": "USERSPARSE", "timestamp": _ms(2021, 5, 5),
        "helpful_vote": 0, "verified_purchase": True,
    })

    # A suspicious user: identical text across items, all 5 stars, seconds apart.
    base = _ms(2022, 6, 1)
    for index in range(6):
        records.append({
            "rating": 5.0, "title": "Amazing", "images": [],
            "text": "Absolutely perfect product, best purchase I have ever made here.",
            "asin": f"S{index:09d}", "parent_asin": f"P{index:09d}",
            "user_id": "USERSHILL", "timestamp": base + index * 20_000,
            "helpful_vote": 0, "verified_purchase": False,
        })

    # Invalid rows the cleaner must remove.
    records += [
        {"rating": 7.0, "title": "Bad rating", "text": "Out of range.", "images": [],
         "asin": "BAD0000001", "parent_asin": "P000000000", "user_id": "USER0000",
         "timestamp": _ms(2020, 1, 1), "helpful_vote": 0, "verified_purchase": True},
        {"rating": 4.0, "title": "Bad time", "text": "Impossible timestamp.", "images": [],
         "asin": "BAD0000002", "parent_asin": "P000000001", "user_id": "USER0001",
         "timestamp": _ms(2031, 1, 1), "helpful_vote": 0, "verified_purchase": True},
        {"rating": 4.0, "title": "No user", "text": "Missing key.", "images": [],
         "asin": "BAD0000003", "parent_asin": "P000000002", "user_id": None,
         "timestamp": _ms(2020, 1, 1), "helpful_vote": 0, "verified_purchase": True},
    ]

    # Exact duplicate of the first record (same user, item and timestamp).
    records.append(dict(records[0]))
    return records


@pytest.fixture(scope="session")
def meta_records() -> list[dict]:
    """Synthetic product metadata using the real meta schema.

    Deliberately includes the awkward real-world cases: a null price, a price
    as a currency string, empty `categories`, and brand spelling variants.
    """
    brands = ["Sony", "SONY", "sony ", "Acme Corp", "acme corp"]
    records = []
    for index in range(6):
        records.append({
            "main_category": "Video Games",
            "title": f"Product {index}",
            "average_rating": 4.0 + (index % 5) / 10,
            "rating_number": 10 * (index + 1),
            "features": [f"feature {index}a", f"feature {index}b"],
            "description": [f"Description paragraph for product {index}."],
            "price": [None, "$19.99", 24.5, "", "12.99 - 24.99", 0][index % 6],
            "images": [], "videos": [],
            "store": brands[index % len(brands)],
            "categories": [] if index % 2 else ["Video Games", "Accessories"],
            "details": {"Weight": f"{index} lb"},
            "parent_asin": f"P{index:09d}",
            "bought_together": None,
        })
    return records


@pytest.fixture()
def reviews_file(tmp_path: Path, review_records: list[dict]) -> Path:
    path = tmp_path / "reviews.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in review_records:
            handle.write(json.dumps(record) + "\n")
    return path


@pytest.fixture()
def meta_file(tmp_path: Path, meta_records: list[dict]) -> Path:
    path = tmp_path / "meta.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in meta_records:
            handle.write(json.dumps(record) + "\n")
    return path


@pytest.fixture()
def data_config(tmp_path: Path, reviews_file: Path, meta_file: Path) -> Config:
    """A full data config pointing at the synthetic files instead of the network."""
    return Config({
        "dataset": {"name": "synthetic_test"},
        "active_tier": "test",
        "tiers": {
            "test": {
                "category": "Test",
                "reviews_url": str(reviews_file),
                "meta_url": str(meta_file),
                "materialise": True,
                "limits": {"max_reviews": None, "max_users": None, "max_products": None},
                "k_core": {"min_interactions_per_user": 3,
                           "min_interactions_per_item": 2, "max_passes": 10},
            }
        },
        "sampling": {"strategy": "user_core", "seed": 42, "declare_bias": True},
        "schema": {
            "interactions": {
                "field_map": {
                    "user_id": "user_id", "item_id": "parent_asin", "variant_id": "asin",
                    "rating": "rating", "review_title": "title", "review_text": "text",
                    "timestamp": "timestamp", "helpful_votes": "helpful_vote",
                    "verified_purchase": "verified_purchase",
                },
                "required": ["user_id", "item_id", "rating", "timestamp"],
            },
            "products": {
                "field_map": {
                    "item_id": "parent_asin", "product_title": "title",
                    "product_description": "description", "product_features": "features",
                    "brand": "store", "main_category": "main_category",
                    "categories": "categories", "price": "price",
                    "avg_rating_meta": "average_rating",
                    "rating_count_meta": "rating_number", "details": "details",
                    "seller": None,
                },
                "required": ["item_id"],
            },
        },
        "cleaning": {
            "rating": {"min": 1.0, "max": 5.0},
            "timestamp": {"min_iso": "1996-01-01T00:00:00Z",
                          "max_iso": "2023-12-31T23:59:59Z", "unit": "ms"},
            "text": {"strip_html": True, "normalise_whitespace": True,
                     "unescape_entities": True, "lowercase": False, "min_length_chars": 0},
            "duplicates": {"exact_keys": ["user_id", "item_id", "timestamp"],
                           "flag_repeated_text_by_user": True},
            "outliers": {"action": "flag", "flag_column": "outlier_flag"},
        },
        "paths": {
            "raw": str(tmp_path / "raw"), "interim": str(tmp_path / "interim"),
            "processed": str(tmp_path / "processed"), "embeddings": str(tmp_path / "emb"),
            "reports": str(tmp_path / "reports"), "figures": str(tmp_path / "reports/figures"),
            "tables": str(tmp_path / "reports/tables"), "results": str(tmp_path / "reports/results"),
        },
        "split": {"strategy": "global_temporal", "train_frac": 0.8, "val_frac": 0.1,
                  "test_frac": 0.1, "enforce_no_future_edges": True},
    }, name="test_data_config")
