"""Graph construction, weighting and trust-conversion invariants."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.graph.hetero_graph import GraphBundle, NodeIndex
from src.graph.temporal import exponential_decay
from src.graph.weighting import anomaly_to_trust, compute_edge_weights
from src.utils.config import Config


class TestNodeIndex:
    def test_roundtrip(self):
        index = NodeIndex.from_ids(["a", "b", "c", "a"])
        assert len(index) == 3
        assert index.decode(index.encode(["c", "a"])) == ["c", "a"]

    def test_unknown_id_is_minus_one_not_an_exception(self):
        assert NodeIndex.from_ids(["a"]).encode(["zzz"])[0] == -1


class TestTrustConversion:
    @pytest.mark.parametrize("norm", ["rank", "minmax", "zscore_sigmoid"])
    def test_trust_is_bounded(self, norm):
        scores = np.random.default_rng(0).normal(size=500) * 10
        trust = anomaly_to_trust(scores, normalisation=norm, floor=0.05)
        assert trust.min() >= 0.05 and trust.max() <= 1.0

    def test_trust_is_monotonically_decreasing_in_anomaly(self):
        trust = anomaly_to_trust(np.array([0.0, 1.0, 2.0, 3.0]), "rank")
        assert all(trust[i] > trust[i + 1] for i in range(len(trust) - 1))

    def test_floor_prevents_zero_trust(self):
        """A trust of 0 would silently delete an edge from a weighted loss."""
        trust = anomaly_to_trust(np.arange(100.0), "rank", floor=0.05)
        assert trust.min() == pytest.approx(0.05)
        assert (trust > 0).all()

    def test_unknown_normalisation_rejected(self):
        with pytest.raises(ValueError, match="Unknown trust.normalisation"):
            anomaly_to_trust(np.array([1.0]), "magic")

    def test_empty_input_is_safe(self):
        assert anomaly_to_trust(np.array([])).size == 0


class TestEdgeWeighting:
    def _config(self, trust=True, temporal=True):
        return Config({"edge_weighting": {"base": 1.0, "trust_exponent": 1.0,
                                          "temporal_exponent": 1.0,
                                          "use_trust": trust, "use_temporal": temporal}},
                      name="w")

    def test_disabled_components_leave_weights_at_base(self):
        frame = pd.DataFrame({"a": range(5)})
        weights = compute_edge_weights(frame, self._config(False, False))
        assert np.allclose(weights, 1.0)

    def test_product_of_factors(self):
        frame = pd.DataFrame({"a": range(3)})
        weights = compute_edge_weights(
            frame, self._config(), trust=np.array([0.5, 1.0, 0.2]),
            temporal=np.array([1.0, 0.5, 0.5]),
        )
        assert np.allclose(weights, [0.5, 0.5, 0.1], atol=1e-6)

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="trust has"):
            compute_edge_weights(pd.DataFrame({"a": range(5)}), self._config(),
                                 trust=np.array([1.0, 1.0]))

    def test_weights_always_finite(self):
        frame = pd.DataFrame({"a": range(100)})
        weights = compute_edge_weights(frame, self._config(),
                                       trust=np.zeros(100), temporal=np.zeros(100))
        assert np.isfinite(weights).all()


class TestTemporalDecay:
    def test_reference_row_has_weight_one(self):
        assert exponential_decay(np.array([1_000_000]), 1_000_000)[0] == pytest.approx(1.0)

    def test_older_is_smaller(self):
        year = int(365.25 * 24 * 3600 * 1000)
        weights = exponential_decay(np.array([0, -year, -5 * year]), 0, lambda_per_year=0.5)
        assert weights[0] > weights[1] > weights[2]

    def test_two_years_matches_the_formula(self):
        year = int(365.25 * 24 * 3600 * 1000)
        weight = exponential_decay(np.array([-2 * year]), 0, lambda_per_year=0.5)[0]
        assert weight == pytest.approx(np.exp(-1.0), rel=1e-3)

    def test_future_rows_never_exceed_one(self):
        year = int(365.25 * 24 * 3600 * 1000)
        assert exponential_decay(np.array([5 * year]), 0)[0] <= 1.0

    def test_floor_applied(self):
        year = int(365.25 * 24 * 3600 * 1000)
        weight = exponential_decay(np.array([-100 * year]), 0, min_weight=0.01)[0]
        assert weight == pytest.approx(0.01)


class TestGraphBundleValidation:
    def _bundle(self):
        from torch_geometric.data import HeteroData
        data = HeteroData()
        data["user"].x = torch.randn(5, 3); data["user"].num_nodes = 5
        data["product"].x = torch.randn(4, 3); data["product"].num_nodes = 4
        data["user", "buys", "product"].edge_index = torch.tensor([[0, 1], [0, 1]])
        return GraphBundle(data=data, indices={}, metadata={})

    def test_valid_graph_passes(self):
        self._bundle().validate()

    def test_out_of_range_edge_is_caught(self):
        bundle = self._bundle()
        bundle.data["user", "buys", "product"].edge_index = torch.tensor([[0, 99], [0, 1]])
        with pytest.raises(ValueError, match="source index exceeds"):
            bundle.validate()

    def test_non_finite_features_are_caught(self):
        bundle = self._bundle()
        bundle.data["user"].x[0, 0] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            bundle.validate()
