"""Phase 1 tests: config, schema adaptation, ingestion, cleaning.

These assert the invariants the later phases are allowed to rely on. In
particular they pin the behaviours that are easy to regress into dishonesty:
absent fields staying absent, suspicious rows being flagged rather than
deleted, and sampling bias being declared in the manifest.
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

import pandas as pd
import pytest

from src.data.cleaner import DataCleaner
from src.data.json_to_csv import convert_json_to_csv
from src.data.loader import (
    DatasetLoader,
    acquire,
    iter_local_records,
    scan_user_counts,
    select_users,
)
from src.data.validator import (
    CANONICAL_INTERACTION_FIELDS,
    CANONICAL_PRODUCT_FIELDS,
    SchemaAdapter,
    SchemaError,
)
from src.utils.config import Config, load_config
from src.utils.seed import set_seed


# =============================================================================
# Config
# =============================================================================
class TestConfig:
    def test_real_data_config_loads(self):
        config = load_config("data_config")
        assert config.require("active_tier") in config.require("tiers")

    def test_every_tier_is_well_formed(self):
        config = load_config("data_config")
        for name, tier in config.require("tiers").items():
            assert tier["reviews_url"].startswith("http"), name
            assert "limits" in tier and "k_core" in tier, name

    def test_dotted_get_and_default(self, data_config: Config):
        assert data_config.get("cleaning.rating.max") == 5.0
        assert data_config.get("cleaning.nope.missing", "fallback") == "fallback"

    def test_immutable(self, data_config: Config):
        with pytest.raises(TypeError):
            data_config.active_tier = "other"

    def test_missing_key_names_alternatives(self, data_config: Config):
        with pytest.raises(KeyError, match="Available"):
            _ = data_config["does_not_exist"]

    def test_overrides_do_not_mutate_original(self, data_config: Config):
        updated = data_config.with_overrides({"cleaning.rating.max": 10.0})
        assert updated.get("cleaning.rating.max") == 10.0
        assert data_config.get("cleaning.rating.max") == 5.0


# =============================================================================
# Schema adapter
# =============================================================================
class TestSchemaAdapter:
    def _adapter(self, data_config: Config) -> SchemaAdapter:
        return SchemaAdapter.for_interactions(data_config.require("schema"))

    def test_detects_real_amazon_fields(self, data_config, review_records):
        availability = self._adapter(data_config).detect(review_records)
        assert availability.has("user_id")
        assert availability.mapped["item_id"] == "parent_asin"
        assert availability.mapped["helpful_votes"] == "helpful_vote"

    def test_seller_reported_absent_not_invented(self, data_config, meta_records):
        adapter = SchemaAdapter.for_products(data_config.require("schema"))
        availability = adapter.detect(meta_records)
        assert not availability.has("seller")
        assert "seller" in availability.declared_absent
        assert "seller" not in adapter.apply(meta_records[0])

    def test_missing_required_field_raises(self, review_records):
        adapter = SchemaAdapter(
            field_map={"user_id": "user_id", "item_id": "nonexistent_column",
                       "rating": "rating", "timestamp": "timestamp"},
            required=["user_id", "item_id", "rating", "timestamp"],
            canonical=CANONICAL_INTERACTION_FIELDS, name="broken",
        )
        with pytest.raises(SchemaError, match="required canonical field"):
            adapter.detect(review_records)

    def test_unknown_canonical_name_rejected(self):
        with pytest.raises(SchemaError, match="unknown canonical field"):
            SchemaAdapter(field_map={"not_a_field": "x"}, required=[],
                          canonical=CANONICAL_PRODUCT_FIELDS, name="bad")

    @pytest.mark.parametrize(
        "raw,expected",
        [("$19.99", 19.99), (24.5, 24.5), ("", None), (None, None),
         ("12.99 - 24.99", 12.99), ("1,299.00", 1299.0), ("free", None)],
    )
    def test_price_coercion_handles_real_world_junk(self, data_config, meta_records, raw, expected):
        adapter = SchemaAdapter.for_products(data_config.require("schema"))
        adapter.detect(meta_records)
        record = dict(meta_records[0]) | {"price": raw}
        assert adapter.apply(record)["price"] == expected

    def test_description_list_is_flattened(self, data_config, meta_records):
        adapter = SchemaAdapter.for_products(data_config.require("schema"))
        adapter.detect(meta_records)
        assert isinstance(adapter.apply(meta_records[0])["product_description"], str)

    def test_text_normalised_but_case_preserved_for_sbert(self, data_config, review_records):
        adapter = self._adapter(data_config)
        adapter.detect(review_records)
        record = dict(review_records[0]) | {"text": "<b>Great</b>  &amp;  Loud\\nSecond line"}
        cleaned = adapter.apply(record)["review_text"]
        assert "<b>" not in cleaned and "&amp;" not in cleaned
        assert "  " not in cleaned
        assert "Great" in cleaned, "SBERT is a cased model; casing must survive"

    def test_bool_coercion(self, data_config, review_records):
        adapter = self._adapter(data_config)
        adapter.detect(review_records)
        for raw, expected in [("true", True), ("N", False), (1, True), (None, None)]:
            record = dict(review_records[0]) | {"verified_purchase": raw}
            assert adapter.apply(record)["verified_purchase"] is expected

    def test_availability_before_detect_raises(self, data_config):
        with pytest.raises(RuntimeError, match="detect"):
            _ = self._adapter(data_config).availability


# =============================================================================
# Record iteration
# =============================================================================
class TestRecordIteration:
    def test_jsonl(self, reviews_file, review_records):
        assert len(list(iter_local_records(reviews_file))) == len(review_records)

    def test_max_records_is_respected(self, reviews_file):
        assert len(list(iter_local_records(reviews_file, max_records=5))) == 5

    def test_json_array(self, tmp_path, review_records):
        path = tmp_path / "reviews.json"
        path.write_text(json.dumps(review_records), encoding="utf-8")
        assert len(list(iter_local_records(path))) == len(review_records)

    def test_gzipped_jsonl(self, tmp_path, review_records):
        path = tmp_path / "reviews.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for record in review_records:
                handle.write(json.dumps(record) + "\n")
        assert len(list(iter_local_records(path))) == len(review_records)

    def test_csv(self, tmp_path, review_records):
        path = tmp_path / "reviews.csv"
        pd.DataFrame(review_records).to_csv(path, index=False)
        assert len(list(iter_local_records(path))) == len(review_records)

    def test_malformed_lines_skipped_not_fatal(self, tmp_path):
        path = tmp_path / "broken.jsonl"
        path.write_text('{"a": 1}\nNOT JSON\n{"a": 2}\n\n', encoding="utf-8")
        assert len(list(iter_local_records(path))) == 2

    def test_acquire_rejects_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            acquire(str(tmp_path / "nope.jsonl"), tmp_path)


# =============================================================================
# JSON -> CSV conversion
# =============================================================================
class TestJsonToCsv:
    def test_row_count_matches(self, reviews_file, review_records, tmp_path):
        csv_path = convert_json_to_csv(reviews_file, tmp_path / "reviews.csv")
        assert len(list(iter_local_records(csv_path))) == len(review_records)

    def test_list_and_dict_fields_round_trip(self, meta_file, meta_records, tmp_path):
        """Product metadata has real list/dict fields (features, details, ...);
        those have to survive being written into and read back out of a CSV
        cell rather than getting flattened into a single string."""
        csv_path = convert_json_to_csv(meta_file, tmp_path / "meta.csv")
        recovered = list(iter_local_records(csv_path))
        assert recovered[0]["features"] == meta_records[0]["features"]
        assert recovered[0]["details"] == meta_records[0]["details"]

    def test_cached_csv_is_reused(self, reviews_file, tmp_path):
        csv_path = tmp_path / "reviews.csv"
        first = convert_json_to_csv(reviews_file, csv_path)
        written_at = first.stat().st_mtime
        again = convert_json_to_csv(reviews_file, csv_path)
        assert again.stat().st_mtime == written_at

    def test_force_reconverts(self, reviews_file, tmp_path):
        csv_path = tmp_path / "reviews.csv"
        convert_json_to_csv(reviews_file, csv_path)
        before = csv_path.read_bytes()
        convert_json_to_csv(reviews_file, csv_path, force=True)
        assert csv_path.read_bytes() == before  # same source -> same output, just rewritten


# =============================================================================
# Sampling
# =============================================================================
class TestSampling:
    def test_scan_user_counts_matches_manual_count(self, reviews_file):
        scanned = scan_user_counts(reviews_file, "user_id")
        manual: dict[str, int] = {}
        for record in iter_local_records(reviews_file):
            user = record.get("user_id")
            if user is not None:
                manual[str(user)] = manual.get(str(user), 0) + 1
        assert dict(scanned) == manual

    def test_select_users_filters_then_samples(self):
        counts = {f"U{i}": i for i in range(1, 21)}
        chosen = select_users(counts, max_users=5, min_interactions=10,
                              rng=random.Random(0))
        assert len(chosen) == 5
        assert all(counts[user] >= 10 for user in chosen)

    def test_select_users_is_deterministic_for_a_seed(self):
        counts = {f"U{i}": 10 for i in range(100)}
        a = select_users(counts, max_users=10, min_interactions=5, rng=random.Random(42))
        b = select_users(counts, max_users=10, min_interactions=5, rng=random.Random(42))
        assert a == b

    def test_select_users_raises_when_threshold_impossible(self):
        with pytest.raises(SchemaError, match="Lower k_core"):
            select_users({"U1": 1}, max_users=5, min_interactions=99, rng=random.Random(0))

    def test_budget_admits_fewer_users_rather_than_truncating(self):
        """A row cap must cost whole users, never partial histories."""
        counts = {f"U{i}": 6 for i in range(50)}
        chosen = select_users(counts, max_users=None, min_interactions=3,
                              rng=random.Random(1), max_interactions=20)
        assert sum(counts[u] for u in chosen) <= 20
        assert len(chosen) == 3          # 3 x 6 = 18 fits; a 4th would overflow

    def test_budget_raises_when_no_user_fits(self):
        with pytest.raises(SchemaError, match="max_reviews"):
            select_users({"U1": 100}, max_users=None, min_interactions=1,
                         rng=random.Random(0), max_interactions=5)

    def test_user_core_keeps_complete_histories_under_a_row_cap(self, data_config):
        """Regression: the row cap used to truncate the stream mid-user."""
        capped = data_config.with_overrides({"tiers.test.limits.max_reviews": 20})
        frame = DatasetLoader(capped).load().interactions

        full = DatasetLoader(data_config).load().interactions
        true_history = full.groupby("user_id").size()
        sampled_history = frame.groupby("user_id").size()
        for user, length in sampled_history.items():
            assert length == true_history[user], (
                f"user {user} kept {length} of {true_history[user]} interactions - "
                f"history was truncated by the row cap"
            )

    def test_manifest_declares_sampling_bias(self, data_config):
        manifest = DatasetLoader(data_config).load().manifest
        assert manifest["sampling"]["strategy"] == "user_core"
        assert len(manifest["sampling"]["bias_declaration"]) > 40

    def test_head_strategy_is_labelled_biased(self, data_config):
        config = data_config.with_overrides({"sampling.strategy": "head"})
        manifest = DatasetLoader(config).load().manifest
        assert "BIASED" in manifest["sampling"]["bias_declaration"]

    def test_unknown_strategy_rejected(self, data_config):
        config = data_config.with_overrides({"sampling.strategy": "magic"})
        with pytest.raises(ValueError, match="head|reservoir|user_core"):
            DatasetLoader(config).load()


# =============================================================================
# Loader
# =============================================================================
class TestLoader:
    def test_load_produces_canonical_columns(self, data_config):
        result = DatasetLoader(data_config).load()
        for column in ("user_id", "item_id", "rating", "timestamp", "event_time"):
            assert column in result.interactions.columns

    def test_raw_file_is_not_modified(self, data_config, reviews_file):
        before = reviews_file.read_bytes()
        DatasetLoader(data_config).load()
        assert reviews_file.read_bytes() == before

    def test_event_time_derived_from_ms_timestamp(self, data_config):
        frame = DatasetLoader(data_config).load().interactions
        assert str(frame["event_time"].dtype).startswith("datetime64")
        assert frame["event_time"].dt.year.between(1996, 2031).all()

    def test_nullable_dtypes_keep_zero_distinct_from_missing(self, data_config):
        frame = DatasetLoader(data_config).load().interactions
        assert str(frame["helpful_votes"].dtype) == "Int32"
        assert str(frame["verified_purchase"].dtype) == "boolean"

    def test_products_restricted_to_interacted_items(self, data_config):
        result = DatasetLoader(data_config).load()
        assert result.products is not None
        assert set(result.products["item_id"]) <= set(result.interactions["item_id"])

    def test_manifest_records_counts_and_provenance(self, data_config):
        manifest = DatasetLoader(data_config).load().manifest
        assert manifest["counts"]["interactions"] > 0
        assert manifest["sources"]["reviews"]["local_path"]
        assert manifest["field_availability"]["products"]["declared_absent"] == ["seller"]

    def test_converted_csv_is_written_to_interim(self, data_config):
        """Ingestion converts JSON to CSV once and works off that CSV from
        then on - the converted file should actually land in data/interim/."""
        manifest = DatasetLoader(data_config).load().manifest
        csv_path = Path(manifest["sources"]["reviews"]["csv_path"])
        assert csv_path.suffix == ".csv"
        assert csv_path.parent == Path(data_config.require("paths.interim"))
        assert csv_path.exists()

    def test_max_products_is_reported_not_silently_ignored(self, data_config):
        """A config key that does nothing is a lie in the config file."""
        config = data_config.with_overrides({"tiers.test.limits.max_products": 2})
        manifest = DatasetLoader(config).load().manifest
        budget = manifest["product_budget"]
        assert budget["max_products"] == 2
        assert budget["exceeded"] is True
        assert "no truncation" in budget["policy"]

    def test_max_products_guard_does_not_drop_items(self, data_config):
        tight = data_config.with_overrides({"tiers.test.limits.max_products": 1})
        loose = data_config.with_overrides({"tiers.test.limits.max_products": None})
        assert (DatasetLoader(tight).load().n_items
                == DatasetLoader(loose).load().n_items)

    def test_unknown_tier_names_alternatives(self, data_config):
        config = data_config.with_overrides({"active_tier": "nope"})
        with pytest.raises(KeyError, match="Available"):
            DatasetLoader(config)


# =============================================================================
# Cleaner
# =============================================================================
@pytest.fixture()
def cleaned(data_config):
    loaded = DatasetLoader(data_config).load()
    cleaner = DataCleaner(data_config)
    return cleaner.clean(loaded), cleaner


class TestCleaner:
    def test_invalid_ratings_removed(self, cleaned):
        result, _ = cleaned
        assert result.interactions["rating"].between(1.0, 5.0).all()

    def test_impossible_timestamps_removed(self, cleaned):
        result, _ = cleaned
        assert result.interactions["event_time"].max() <= pd.Timestamp("2023-12-31T23:59:59Z")

    def test_null_keys_removed(self, cleaned):
        result, _ = cleaned
        assert result.interactions["user_id"].notna().all()
        assert result.interactions["item_id"].notna().all()

    def test_one_edge_per_user_item_pair(self, cleaned):
        result, _ = cleaned
        assert not result.interactions.duplicated(subset=["user_id", "item_id"]).any()

    def test_suspicious_rows_are_flagged_not_deleted(self, cleaned):
        result, _ = cleaned
        assert "outlier_flag" in result.interactions.columns
        assert "outlier_reasons" in result.interactions.columns
        flagged = result.interactions[result.interactions["outlier_flag"]]
        assert len(flagged) > 0, "the shill user should trip at least one rule"
        assert (flagged["outlier_reasons"].str.len() > 0).all(), "every flag needs a reason"

    def test_shill_user_survives_cleaning(self, cleaned):
        """The anomaly detector cannot find what cleaning already deleted."""
        result, _ = cleaned
        assert "USERSHILL" in set(result.interactions["user_id"])

    def test_k_core_thresholds_hold(self, cleaned, data_config):
        result, _ = cleaned
        k = data_config["tiers"]["test"]["k_core"]
        counts = result.interactions["user_id"].value_counts()
        assert counts.min() >= int(k["min_interactions_per_user"])
        item_counts = result.interactions["item_id"].value_counts()
        assert item_counts.min() >= int(k["min_interactions_per_item"])

    @staticmethod
    def _cascade_result(n: int = 20):
        """A graph engineered to need ~n/2 k-core passes to settle.

        User j touches items j and j-1, so item n-1 and user 0 both start at
        degree 1. Removing them drops their neighbours to degree 1, which
        cascades inward from both ends one layer per pass.
        """
        from src.data.loader import LoadResult
        from src.data.validator import FieldAvailability

        rows = []
        for j in range(n):
            rows.append((f"U{j:04d}", f"P{j:09d}"))
            if j >= 1:
                rows.append((f"U{j:04d}", f"P{j - 1:09d}"))
        frame = pd.DataFrame(rows, columns=["user_id", "item_id"])
        frame["rating"] = 5.0
        frame["timestamp"] = range(1_600_000_000_000, 1_600_000_000_000 + len(frame))
        frame["event_time"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame["review_text"] = "text"
        frame["user_id"] = frame["user_id"].astype("string")
        frame["item_id"] = frame["item_id"].astype("string")
        return LoadResult(frame, None, FieldAvailability(), None, {})

    def test_k_core_refuses_to_ship_a_violated_invariant(self, data_config):
        """Regression: non-convergence used to warn and return violating data."""
        strict = data_config.with_overrides({
            "tiers.test.k_core.max_passes": 3,
            "tiers.test.k_core.min_interactions_per_user": 2,
            "tiers.test.k_core.min_interactions_per_item": 2,
        })
        with pytest.raises(RuntimeError, match="k-core did not converge"):
            DataCleaner(strict).clean(self._cascade_result())

    def test_k_core_converges_when_given_enough_passes(self, data_config):
        generous = data_config.with_overrides({
            "tiers.test.k_core.max_passes": 100,
            "tiers.test.k_core.min_interactions_per_user": 2,
            "tiers.test.k_core.min_interactions_per_item": 2,
        })
        result = DataCleaner(generous).clean(self._cascade_result())
        frame = result.interactions
        if len(frame):
            assert frame["user_id"].value_counts().min() >= 2
            assert frame["item_id"].value_counts().min() >= 2

    def test_sparse_user_dropped_by_k_core(self, cleaned):
        result, _ = cleaned
        assert "USERSPARSE" not in set(result.interactions["user_id"])

    def test_output_sorted_chronologically(self, cleaned):
        result, _ = cleaned
        assert result.interactions["timestamp"].is_monotonic_increasing

    def test_ids_uppercased_for_stable_joins(self, cleaned):
        result, _ = cleaned
        ids = result.interactions["item_id"].dropna()
        assert (ids == ids.str.upper()).all()

    def test_brand_spelling_variants_merged(self, cleaned):
        result, _ = cleaned
        brands = set(result.products["brand"].dropna())
        lowered = [b.lower().strip() for b in brands]
        assert len(lowered) == len(set(lowered)), f"unmerged brand variants: {brands}"

    def test_zero_price_treated_as_missing_never_imputed(self, cleaned):
        result, _ = cleaned
        prices = result.products["price"].dropna()
        assert (prices > 0).all()
        assert "price_missing" in result.products.columns

    def test_report_accounts_for_every_row(self, cleaned):
        result, _ = cleaned
        report = result.report.to_dict()
        assert report["summary"]["rows_raw"] >= report["summary"]["rows_final"]
        # Each step's "after" must be the next step's "before": no silent losses.
        steps = report["steps"]
        for earlier, later in zip(steps, steps[1:]):
            assert earlier["rows_after"] == later["rows_before"], (
                f"unaccounted rows between '{earlier['step']}' and '{later['step']}'"
            )

    def test_report_distinguishes_pre_and_post_kcore_time_span(self, cleaned):
        result, _ = cleaned
        notes = " ".join(result.report.notes)
        assert "before k-core" in notes
        assert "Final retained time span" in notes

    def test_report_discloses_kcore_leakage_caveat(self, cleaned):
        result, _ = cleaned
        notes = " ".join(result.report.notes).lower()
        assert "leakage" in notes and "k-core" in notes

    def test_report_writes_json_and_markdown(self, cleaned, tmp_path):
        _, cleaner = cleaned
        json_path, md_path = cleaner.save_report()
        assert json.loads(json_path.read_text())["summary"]["rows_final"] >= 0
        assert md_path.read_text().startswith("# Cleaning Report")


# =============================================================================
# Reproducibility
# =============================================================================
class TestReproducibility:
    def test_same_seed_same_sample(self, data_config):
        first = DatasetLoader(data_config).load().interactions
        second = DatasetLoader(data_config).load().interactions
        pd.testing.assert_frame_equal(first, second)

    def test_set_seed_makes_python_and_numpy_deterministic(self):
        import numpy as np
        set_seed(123)
        a = (random.random(), float(np.random.rand()))
        set_seed(123)
        assert a == (random.random(), float(np.random.rand()))
