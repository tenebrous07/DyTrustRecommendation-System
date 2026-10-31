"""Synthetic anomaly injection and detection-metric invariants (Module 45)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.anomaly.synthetic import SyntheticAnomalyInjector, detection_metrics
from src.utils.config import Config


@pytest.fixture()
def genuine() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    base = pd.Timestamp("2020-01-01", tz="UTC")
    rows = []
    for user in range(60):
        for _ in range(rng.integers(5, 12)):
            when = base + pd.Timedelta(days=int(rng.integers(0, 900)))
            rows.append({
                "user_id": f"U{user:04d}", "item_id": f"P{rng.integers(0, 40):04d}",
                "rating": float(rng.integers(1, 6)),
                "review_title": "title", "review_text": f"genuine opinion {rng.integers(0, 999)}",
                "timestamp": int(when.timestamp() * 1000), "event_time": when,
                "helpful_votes": int(rng.integers(0, 5)), "verified_purchase": True,
            })
    frame = pd.DataFrame(rows).drop_duplicates(["user_id", "item_id"], keep="last")
    return frame.sort_values("timestamp").reset_index(drop=True)


@pytest.fixture()
def config() -> Config:
    return Config({"synthetic_anomaly": {
        "enabled": True, "output_suffix": "_synthetic", "injection_rate": 0.10,
        "patterns": {
            "burst_reviews": {"enabled": True, "reviews_per_burst": 8, "window_minutes": 30},
            "extreme_rating": {"enabled": True, "forced_rating": 5.0},
            "duplicated_text": {"enabled": True, "template_pool_size": 3},
            "coordinated_targeting": {"enabled": True, "shills_per_target": 3},
        },
    }}, name="syn")


class TestInjection:
    def test_original_rows_are_untouched(self, genuine, config):
        before = genuine.copy()
        result = SyntheticAnomalyInjector(config, seed=1).inject(genuine)
        pd.testing.assert_frame_equal(genuine, before)
        kept = result.interactions[~result.interactions.user_id.str.startswith("SYNTH_")]
        assert len(kept) == len(before)

    def test_injected_users_are_labelled(self, genuine, config):
        result = SyntheticAnomalyInjector(config, seed=1).inject(genuine)
        assert result.n_shills > 0
        assert all(u.startswith("SYNTH_SHILL_") for u in result.shill_users)
        labels = result.label_array(sorted(result.interactions.user_id.unique()))
        assert labels.sum() == result.n_shills

    def test_shills_exhibit_every_configured_pattern(self, genuine, config):
        result = SyntheticAnomalyInjector(config, seed=1).inject(genuine)
        shills = result.interactions[result.interactions.user_id.isin(result.shill_users)]
        assert (shills["rating"] == 5.0).all()                        # extreme rating
        assert shills["review_text"].nunique() <= 3                   # duplicated text
        per_user = shills.groupby("user_id")["timestamp"]
        span_minutes = (per_user.max() - per_user.min()) / 60000.0
        assert span_minutes.max() <= 30.0                             # burst window

    def test_shills_are_not_single_item_accounts(self, genuine, config):
        """Camouflage matters: degree-1 accounts would be trivially detectable."""
        result = SyntheticAnomalyInjector(config, seed=1).inject(genuine)
        shills = result.interactions[result.interactions.user_id.isin(result.shill_users)]
        assert shills.groupby("user_id")["item_id"].nunique().min() > 1

    def test_stats_carry_a_synthetic_warning(self, genuine, config):
        result = SyntheticAnomalyInjector(config, seed=1).inject(genuine)
        assert "SYNTHETIC" in result.stats["WARNING"]
        assert "real fraud" in result.stats["WARNING"]

    def test_reproducible_for_a_seed(self, genuine, config):
        a = SyntheticAnomalyInjector(config, seed=5).inject(genuine)
        b = SyntheticAnomalyInjector(config, seed=5).inject(genuine)
        assert a.shill_users == b.shill_users
        assert len(a.interactions) == len(b.interactions)

    def test_injection_rate_is_respected(self, genuine, config):
        result = SyntheticAnomalyInjector(config, seed=3).inject(genuine)
        expected = round(0.10 * genuine.user_id.nunique())
        assert abs(result.n_shills - expected) <= 1

    def test_schema_is_preserved(self, genuine, config):
        result = SyntheticAnomalyInjector(config, seed=1).inject(genuine)
        assert list(result.interactions.columns) == list(genuine.columns)
        assert result.interactions["timestamp"].is_monotonic_increasing


class TestDetectionMetrics:
    def test_perfect_ranking_scores_one(self):
        labels = np.array([0, 0, 1, 1])
        metrics = detection_metrics(np.array([0.1, 0.2, 0.9, 0.8]), labels, k=2)
        assert metrics["roc_auc"] == pytest.approx(1.0)
        assert metrics["precision_at_2"] == pytest.approx(1.0)

    def test_inverted_ranking_scores_zero(self):
        labels = np.array([0, 0, 1, 1])
        assert detection_metrics(np.array([0.9, 0.8, 0.1, 0.2]), labels, k=2)["roc_auc"] == \
            pytest.approx(0.0)

    def test_random_ranking_is_near_chance(self):
        rng = np.random.default_rng(0)
        labels = (rng.random(2000) < 0.05).astype(np.int8)
        auc = detection_metrics(rng.random(2000), labels, k=100)["roc_auc"]
        assert 0.4 < auc < 0.6

    def test_single_class_returns_nan_not_a_crash(self):
        metrics = detection_metrics(np.array([0.5, 0.2]), np.array([0, 0]), k=1)
        assert np.isnan(metrics["roc_auc"])

    def test_prevalence_reported(self):
        labels = np.array([0, 0, 0, 1])
        assert detection_metrics(np.array([1.0, 2, 3, 4]), labels, k=2)["prevalence"] == 0.25
