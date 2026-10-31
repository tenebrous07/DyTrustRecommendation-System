"""Metric correctness against hand-computed values."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    average_precision_at_k, coverage, hit_rate_at_k, intra_list_diversity,
    ndcg_at_k, novelty, precision_at_k, recall_at_k, reciprocal_rank,
)


class TestAccuracy:
    def test_precision_perfect_and_zero(self):
        assert precision_at_k(np.array([1, 2, 3]), {1, 2, 3}, 3) == 1.0
        assert precision_at_k(np.array([1, 2, 3]), {9}, 3) == 0.0

    def test_precision_partial(self):
        assert precision_at_k(np.array([1, 5, 2, 7]), {1, 2}, 4) == pytest.approx(0.5)

    def test_recall_uses_relevant_set_size(self):
        assert recall_at_k(np.array([1, 2]), {1, 2, 3, 4}, 2) == pytest.approx(0.5)

    def test_ndcg_is_one_for_perfect_ranking(self):
        assert ndcg_at_k(np.array([1, 2, 3]), {1, 2, 3}, 3) == pytest.approx(1.0)

    def test_ndcg_rewards_earlier_hits(self):
        early = ndcg_at_k(np.array([1, 9, 9]), {1}, 3)
        late = ndcg_at_k(np.array([9, 9, 1]), {1}, 3)
        assert early > late

    def test_ndcg_hand_computed(self):
        # hit at ranks 1 and 3: DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5
        # IDCG (2 relevant) = 1/log2(2) + 1/log2(3) = 1 + 0.63093
        value = ndcg_at_k(np.array([1, 9, 2]), {1, 2}, 3)
        assert value == pytest.approx(1.5 / (1 + 1 / np.log2(3)), rel=1e-6)

    def test_map_hand_computed(self):
        # hits at 1 and 3 -> (1/1 + 2/3) / 2
        assert average_precision_at_k(np.array([1, 9, 2]), {1, 2}, 3) == \
            pytest.approx((1.0 + 2 / 3) / 2)

    def test_mrr(self):
        assert reciprocal_rank(np.array([9, 9, 4]), {4}) == pytest.approx(1 / 3)
        assert reciprocal_rank(np.array([9, 9]), {4}) == 0.0

    def test_hit_rate_is_binary(self):
        assert hit_rate_at_k(np.array([1, 2, 3]), {3}, 3) == 1.0
        assert hit_rate_at_k(np.array([1, 2, 3]), {3}, 2) == 0.0

    @pytest.mark.parametrize("fn", [precision_at_k, recall_at_k, ndcg_at_k,
                                    average_precision_at_k, hit_rate_at_k])
    def test_empty_relevant_set_is_zero_not_a_crash(self, fn):
        assert fn(np.array([1, 2, 3]), set(), 3) == 0.0

    @pytest.mark.parametrize("fn", [precision_at_k, recall_at_k, ndcg_at_k,
                                    average_precision_at_k, hit_rate_at_k])
    def test_bounded_in_unit_interval(self, fn):
        rng = np.random.default_rng(0)
        for _ in range(30):
            ranked = rng.permutation(50)[:10]
            relevant = set(rng.choice(50, size=5, replace=False).tolist())
            assert 0.0 <= fn(ranked, relevant, 10) <= 1.0


class TestBeyondAccuracy:
    def test_coverage_counts_distinct_items(self):
        assert coverage([np.array([0, 1]), np.array([1, 2])], n_items=10, k=2) == 0.3

    def test_coverage_full_and_empty(self):
        assert coverage([np.arange(5)], n_items=5, k=5) == 1.0
        assert coverage([], n_items=5, k=5) == 0.0

    def test_diversity_all_same_category_is_minimal(self):
        categories = np.zeros(10, dtype=int)
        assert intra_list_diversity([np.array([0, 1, 2])], categories, 3) == pytest.approx(1 / 3)

    def test_diversity_all_distinct_is_one(self):
        categories = np.arange(10)
        assert intra_list_diversity([np.array([0, 1, 2])], categories, 3) == 1.0

    def test_novelty_higher_for_rare_items(self):
        popularity = np.array([100.0, 100.0, 1.0])
        popular = novelty([np.array([0, 1])], popularity, 2)
        rare = novelty([np.array([2])], popularity, 1)
        assert rare > popular

    def test_novelty_zero_popularity_is_safe(self):
        assert novelty([np.array([0])], np.zeros(3), 1) == 0.0
