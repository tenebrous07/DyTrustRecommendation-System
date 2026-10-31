"""Phase 3-4 tests: splitting, sentiment, review features, transforms, fusion.

The invariants that matter here are leakage invariants. Most of these tests
fail loudly if a statistic is ever fitted on data the model should not have
seen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.splitter import assert_no_leakage, chronological_split
from src.features.fusion import SemanticBehaviouralFusion
from src.features.review_features import ReviewFeatureBuilder
from src.features.sentiment import SENTIMENT_COLUMNS, add_sentiment_features
from src.features.transforms import LeakageSafeScaler
from src.utils.config import Config


@pytest.fixture()
def interactions() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    base = pd.Timestamp("2019-01-01", tz="UTC")
    rows = []
    for user in range(30):
        for index in range(8):
            when = base + pd.Timedelta(days=int(rng.integers(0, 1400)))
            rows.append({
                "user_id": f"U{user:03d}", "item_id": f"P{rng.integers(0, 20):03d}",
                "rating": float(rng.integers(1, 6)),
                "review_text": "A perfectly reasonable review of this product.",
                "review_title": "Fine", "event_time": when,
                "timestamp": int(when.timestamp() * 1000),
                "helpful_votes": int(rng.integers(0, 10)),
                "verified_purchase": True,
            })
    frame = pd.DataFrame(rows).drop_duplicates(["user_id", "item_id"], keep="last")
    return frame.sort_values("timestamp").reset_index(drop=True)


@pytest.fixture()
def split_config() -> Config:
    return Config({"split": {"strategy": "global_temporal", "train_frac": 0.8,
                             "val_frac": 0.1, "test_frac": 0.1}}, name="split")


class TestSplitter:
    def test_partitions_are_chronological(self, interactions, split_config):
        split = chronological_split(interactions, split_config)
        assert split.train["timestamp"].max() <= split.val["timestamp"].min()
        assert split.val["timestamp"].max() <= split.test["timestamp"].min()

    def test_leakage_assertion_passes(self, interactions, split_config):
        assert_no_leakage(chronological_split(interactions, split_config))

    def test_no_row_appears_twice(self, interactions, split_config):
        split = chronological_split(interactions, split_config)
        assert len(split.train) + len(split.val) + len(split.test) == len(interactions)

    def test_leakage_assertion_catches_a_bad_split(self, interactions, split_config):
        split = chronological_split(interactions, split_config)
        # Move a late row into train: exactly the mistake the check exists for.
        split.train = pd.concat([split.train, split.test.tail(1)], ignore_index=True)
        with pytest.raises(AssertionError, match="Leakage"):
            assert_no_leakage(split)

    def test_reference_timestamp_is_from_train(self, interactions, split_config):
        split = chronological_split(interactions, split_config)
        assert split.train_max_timestamp == int(split.train["timestamp"].max())
        assert split.train_max_timestamp < int(split.test["timestamp"].max())

    def test_all_positives_spans_every_split(self, interactions, split_config):
        split = chronological_split(interactions, split_config)
        positives = split.all_positives()
        total = sum(len(v) for v in positives.values())
        assert total == len(interactions.drop_duplicates(["user_id", "item_id"]))

    def test_bad_fractions_rejected(self, interactions):
        bad = Config({"split": {"train_frac": 0.9, "val_frac": 0.2}}, name="bad")
        with pytest.raises(ValueError, match="Invalid split fractions"):
            chronological_split(interactions, bad)


class TestSentiment:
    def test_adds_all_four_columns(self, interactions):
        out = add_sentiment_features(interactions)
        for column in SENTIMENT_COLUMNS:
            assert column in out.columns

    def test_compound_in_range(self, interactions):
        out = add_sentiment_features(interactions)
        assert out["sentiment_compound"].between(-1, 1).all()

    def test_polarity_direction_is_right(self):
        frame = pd.DataFrame({"review_text": ["This is wonderful, I love it!",
                                              "Terrible. Broke immediately. Awful."]})
        out = add_sentiment_features(frame)
        assert out["sentiment_compound"].iloc[0] > 0
        assert out["sentiment_compound"].iloc[1] < 0

    def test_empty_text_is_zero_not_neutral_guess(self):
        out = add_sentiment_features(pd.DataFrame({"review_text": ["", None]}))
        assert (out["sentiment_compound"] == 0.0).all()

    def test_missing_column_is_skipped(self):
        frame = pd.DataFrame({"other": [1, 2]})
        assert "sentiment_compound" not in add_sentiment_features(frame).columns


class TestReviewFeatures:
    def _builder(self, train):
        return ReviewFeatureBuilder(lambda_per_year=0.5).fit(train)

    def test_transform_before_fit_raises(self, interactions):
        with pytest.raises(RuntimeError, match="before fit"):
            ReviewFeatureBuilder().transform(interactions)

    def test_produces_expected_columns(self, interactions, split_config):
        split = chronological_split(interactions, split_config)
        features = self._builder(split.train).transform(split.train)
        for column in ("review_length_log", "helpful_votes_log", "verified_purchase_f",
                       "rating_centred", "recency_weight"):
            assert column in features.columns

    def test_item_means_come_from_train_only(self, interactions, split_config):
        """The core leakage guard: a test item's own mean must not be used."""
        split = chronological_split(interactions, split_config)
        builder = self._builder(split.train)
        train_items = set(split.train["item_id"].astype(str))
        assert set(builder.item_means) <= train_items

    def test_unseen_item_falls_back_to_global_train_mean(self, interactions, split_config):
        split = chronological_split(interactions, split_config)
        builder = self._builder(split.train)
        unseen = pd.DataFrame({
            "item_id": ["BRAND_NEW"], "rating": [5.0],
            "timestamp": [builder.reference_timestamp],
        })
        centred = builder.transform(unseen)["rating_centred"].iloc[0]
        assert centred == pytest.approx(5.0 - builder.global_mean, abs=1e-4)

    def test_recency_weight_bounds(self, interactions, split_config):
        split = chronological_split(interactions, split_config)
        builder = self._builder(split.train)
        weights = builder.transform(split.train)["recency_weight"]
        assert weights.between(builder.min_weight, 1.0).all()

    def test_future_rows_are_never_upweighted(self, interactions, split_config):
        """Val/test rows postdate the reference; weight must clamp at 1, not exceed it."""
        split = chronological_split(interactions, split_config)
        builder = self._builder(split.train)
        weights = builder.transform(split.test)["recency_weight"]
        assert (weights <= 1.0).all()

    def test_older_interactions_weigh_less(self, interactions, split_config):
        split = chronological_split(interactions, split_config)
        builder = self._builder(split.train)
        reference = builder.reference_timestamp
        year_ms = int(365.25 * 24 * 3600 * 1000)
        weights = builder.recency_weight(
            pd.Series([reference, reference - year_ms, reference - 5 * year_ms])
        )
        assert weights.iloc[0] > weights.iloc[1] > weights.iloc[2]


class TestLeakageSafeScaler:
    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="before fit"):
            LeakageSafeScaler().transform(pd.DataFrame({"a": [1.0]}))

    def test_records_fitted_scope_for_audit(self):
        scaler = LeakageSafeScaler().fit(pd.DataFrame({"a": [1.0, 2.0, 3.0]}),
                                         scope="TRAIN SPLIT ONLY")
        assert scaler.to_dict()["fitted_scope"] == "TRAIN SPLIT ONLY"

    def test_standardises_using_train_statistics_only(self):
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        test = pd.DataFrame({"a": [1000.0]})
        scaler = LeakageSafeScaler(clip_sigma=100).fit(train, log_skewed=False)
        assert scaler.means["a"] == pytest.approx(3.0)
        # The extreme test value must not shift the fitted mean.
        scaler.transform(test)
        assert scaler.means["a"] == pytest.approx(3.0)

    def test_clipping_bounds_outliers(self):
        train = pd.DataFrame({"a": list(range(100))})
        scaler = LeakageSafeScaler(clip_sigma=2.0).fit(train, log_skewed=False)
        out = scaler.transform(pd.DataFrame({"a": [10**9]}))
        assert out["a"].iloc[0] == pytest.approx(2.0)

    def test_missing_column_becomes_zero_not_an_error(self):
        scaler = LeakageSafeScaler().fit(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))
        out = scaler.transform(pd.DataFrame({"a": [1.5]}))
        assert out["b"].iloc[0] == 0.0

    def test_nan_imputed_with_train_median(self):
        scaler = LeakageSafeScaler().fit(pd.DataFrame({"a": [1.0, 2.0, 3.0]}),
                                         log_skewed=False)
        out = scaler.transform(pd.DataFrame({"a": [np.nan]}))
        assert out["a"].iloc[0] == pytest.approx(0.0, abs=1e-5)


class TestFusion:
    def test_output_shape(self):
        fusion = SemanticBehaviouralFusion(384, 6, output_dim=128)
        out = fusion(torch.randn(32, 384), torch.randn(32, 6))
        assert out.shape == (32, 128)

    def test_concat_mode_width_is_the_sum(self):
        fusion = SemanticBehaviouralFusion(384, 6, mode="concat")
        assert fusion.output_dim == 390
        assert fusion(torch.randn(4, 384), torch.randn(4, 6)).shape == (4, 390)

    def test_gradients_flow(self):
        fusion = SemanticBehaviouralFusion(384, 6)
        fusion(torch.randn(8, 384), torch.randn(8, 6)).sum().backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in fusion.parameters())

    def test_output_is_finite(self):
        fusion = SemanticBehaviouralFusion(384, 6)
        assert torch.isfinite(fusion(torch.randn(16, 384), torch.randn(16, 6))).all()

    def test_batch_mismatch_rejected(self):
        fusion = SemanticBehaviouralFusion(384, 6)
        with pytest.raises(ValueError, match="Batch mismatch"):
            fusion(torch.randn(8, 384), torch.randn(4, 6))

    def test_width_mismatch_rejected(self):
        fusion = SemanticBehaviouralFusion(384, 6)
        with pytest.raises(ValueError, match="Expected sbert width"):
            fusion(torch.randn(8, 300), torch.randn(8, 6))

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="mode must be"):
            SemanticBehaviouralFusion(384, 6, mode="magic")
