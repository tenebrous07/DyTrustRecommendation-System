"""Phase 2 tests: profiles, EDA, outlier indicators, graph statistics.

The invariants pinned here are mostly about honesty rather than arithmetic:
absent columns must be skipped rather than defaulted, approximated metrics must
be labelled approximate, and nothing may emit a fraud verdict.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.analytics.eda import ExploratoryAnalysis
from src.analytics.graph_stats import GraphAnalyzer, build_bipartite_graph
from src.analytics.outliers import OutlierDetector
from src.features.product_features import (
    build_product_profiles,
    extract_category_levels,
    long_tail_summary,
)
from src.features.user_features import build_user_profiles
from src.utils.config import Config, load_config


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture()
def analytics_config(tmp_path) -> Config:
    """Real analytics config with paths redirected into a temp directory."""
    config = load_config("analytics_config")
    return config.with_overrides({
        "paths": {
            "raw": str(tmp_path / "raw"), "interim": str(tmp_path / "interim"),
            "processed": str(tmp_path / "processed"), "embeddings": str(tmp_path / "emb"),
            "reports": str(tmp_path / "reports"),
            "figures": str(tmp_path / "reports/figures"),
            "tables": str(tmp_path / "reports/tables"),
            "results": str(tmp_path / "reports/results"),
        },
        # Keep the expensive estimators tiny; their correctness is tested by
        # their labelling, not by their sample size.
        "graph.betweenness.k_samples": 20,
        "graph.closeness.sample_nodes": 20,
        "graph.max_nodes_for_exact_metrics": 10,
    })


@pytest.fixture()
def interactions() -> pd.DataFrame:
    """A synthetic corpus with one deliberately anomalous user."""
    rng = np.random.default_rng(7)
    rows = []
    base = pd.Timestamp("2020-01-01", tz="UTC")

    for user in range(40):
        for index in range(rng.integers(5, 15)):
            rows.append({
                "user_id": f"U{user:03d}",
                "item_id": f"P{rng.integers(0, 25):03d}",
                "rating": float(rng.integers(1, 6)),
                "review_text": f"Review {user}-{index} " + "word " * int(rng.integers(3, 40)),
                "event_time": base + pd.Timedelta(days=int(rng.integers(0, 900))),
                "helpful_votes": int(rng.integers(0, 15)),
                "verified_purchase": bool(rng.integers(0, 2)),
                "outlier_flag": False,
            })

    # A shill: many reviews, all 5 stars, seconds apart, identical text.
    for index in range(30):
        rows.append({
            "user_id": "USHILL",
            "item_id": f"P{index % 25:03d}",
            "rating": 5.0,
            "review_text": "Absolutely perfect product best purchase ever made here truly",
            "event_time": base + pd.Timedelta(seconds=20 * index),
            "helpful_votes": 0,
            "verified_purchase": False,
            "outlier_flag": True,
        })

    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(subset=["user_id", "item_id"], keep="last")
    frame["timestamp"] = (frame["event_time"].astype("int64") // 10**6)
    return frame.sort_values("timestamp").reset_index(drop=True)


@pytest.fixture()
def products() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    paths = [
        ["video games", "playstation 4", "games"],
        ["video games", "xbox one", "accessories"],
        ["video games", "legacy systems", "nintendo systems", "games"],
        ["video games", "pc", "games"],
        [],
    ]
    return pd.DataFrame([{
        "item_id": f"P{i:03d}",
        "product_title": f"Product {i}",
        "product_description": f"Description {i}",
        "brand": ["Nintendo", "Sony", "Microsoft"][i % 3],
        "main_category": "Video Games",
        "categories": paths[i % len(paths)],
        "price": float(rng.integers(5, 90)),
        "price_missing": False,
        "avg_rating_meta": 4.0,
        "rating_count_meta": 100,
    } for i in range(25)])


# =============================================================================
# User profiles
# =============================================================================
class TestUserProfiles:
    def test_one_row_per_user(self, interactions):
        profiles = build_user_profiles(interactions)
        assert len(profiles) == interactions["user_id"].nunique()
        assert profiles.index.name == "user_id"

    def test_counts_match_source(self, interactions):
        profiles = build_user_profiles(interactions)
        expected = interactions.groupby("user_id").size()
        pd.testing.assert_series_equal(
            profiles["n_interactions"].sort_index(),
            expected.sort_index(), check_names=False, check_dtype=False,
        )

    def test_shill_signature_is_visible(self, interactions):
        profiles = build_user_profiles(interactions)
        shill = profiles.loc["USHILL"]
        assert shill["rating_std"] == 0.0
        assert shill["frac_five_star"] == 1.0
        assert shill["rating_entropy"] == 0.0
        assert shill["min_gap_hours"] < 1.0
        assert shill["frac_duplicate_text"] > 0.9

    def test_rating_entropy_separates_uniform_from_bimodal(self):
        """std alone conflates 'always 5' with 'alternates 1 and 5'."""
        base = pd.Timestamp("2021-01-01", tz="UTC")
        rows = []
        for index in range(10):
            rows.append({"user_id": "CONST", "item_id": f"P{index}", "rating": 5.0,
                         "event_time": base + pd.Timedelta(days=index)})
            rows.append({"user_id": "SPLIT", "item_id": f"P{index}",
                         "rating": 1.0 if index % 2 else 5.0,
                         "event_time": base + pd.Timedelta(days=index)})
        profiles = build_user_profiles(pd.DataFrame(rows))
        assert profiles.loc["CONST", "rating_entropy"] == 0.0
        assert profiles.loc["SPLIT", "rating_entropy"] == pytest.approx(1.0)
        assert profiles.loc["SPLIT", "rating_std"] > profiles.loc["CONST", "rating_std"]

    def test_absent_columns_are_skipped_not_defaulted(self, interactions):
        minimal = interactions[["user_id", "item_id", "rating", "event_time"]]
        profiles = build_user_profiles(minimal)
        assert "mean_review_length" not in profiles.columns
        assert "frac_verified" not in profiles.columns
        assert "mean_rating" in profiles.columns

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            build_user_profiles(pd.DataFrame(columns=["user_id", "item_id"]))


# =============================================================================
# Product profiles and categories
# =============================================================================
class TestProductProfiles:
    def test_one_row_per_item(self, interactions):
        profiles = build_product_profiles(interactions)
        assert len(profiles) == interactions["item_id"].nunique()

    def test_popularity_rank_is_dense_and_ordered(self, interactions):
        profiles = build_product_profiles(interactions)
        top = profiles.nsmallest(1, "popularity_rank")
        assert int(top["n_interactions"].iloc[0]) == int(profiles["n_interactions"].max())
        assert profiles["popularity_rank"].min() == 1

    def test_metadata_join_does_not_clobber_observed_stats(self, interactions, products):
        observed = build_product_profiles(interactions)["mean_rating"]
        joined = build_product_profiles(interactions, products)
        # Amazon's own catalogue-wide average must stay under its own name.
        assert "avg_rating_meta" in joined.columns
        pd.testing.assert_series_equal(joined["mean_rating"], observed)

    def test_category_levels_extracted(self, products):
        out = extract_category_levels(products, node_level="l2")
        assert set(out["category_l2"].dropna()) >= {"playstation 4", "xbox one", "pc"}
        assert out["category_depth"].max() == 4

    def test_category_node_level_changes_granularity(self, products):
        coarse = extract_category_levels(products, "l1")["category_node"].nunique()
        fine = extract_category_levels(products, "leaf")["category_node"].nunique()
        assert fine > coarse

    def test_category_falls_back_when_path_empty(self, products):
        out = extract_category_levels(products, node_level="l2")
        empty_path = out[out["categories"].apply(len) == 0]
        assert len(empty_path) > 0
        # No usable path, so it must fall back to main_category, not go null.
        assert empty_path["category_node"].notna().all()

    def test_invalid_node_level_rejected(self, products):
        with pytest.raises(ValueError, match="node_level"):
            extract_category_levels(products, node_level="l7")

    def test_gini_bounds(self, interactions):
        profiles = build_product_profiles(interactions)
        summary = long_tail_summary(profiles)
        assert 0.0 <= summary["gini"] <= 1.0
        assert 0.0 <= summary["head_share"] <= 1.0

    def test_gini_zero_for_uniform_popularity(self):
        profiles = pd.DataFrame({"n_interactions": [10] * 50})
        assert long_tail_summary(profiles)["gini"] == pytest.approx(0.0, abs=1e-9)


# =============================================================================
# EDA
# =============================================================================
class TestEDA:
    def test_run_produces_figures_and_summary(self, interactions, products, analytics_config):
        result = ExploratoryAnalysis(interactions, products, analytics_config).run()
        assert result.summary["n_interactions"] == len(interactions)
        assert result.summary["n_users"] == interactions["user_id"].nunique()
        assert len(result.figures) >= 4
        for path in result.figures.values():
            assert path.exists() and path.stat().st_size > 0

    def test_density_is_consistent(self, interactions, products, analytics_config):
        result = ExploratoryAnalysis(interactions, products, analytics_config).run()
        expected = len(interactions) / (result.summary["n_users"] * result.summary["n_items"])
        assert result.summary["density"] == pytest.approx(expected)

    def test_missing_sentiment_is_declared_not_faked(self, interactions, products,
                                                     analytics_config):
        result = ExploratoryAnalysis(interactions, products, analytics_config).run()
        assert "sentiment_compound" not in result.summary
        assert any("sentiment" in note.lower() for note in result.notes)

    def test_runs_without_product_metadata(self, interactions, analytics_config):
        result = ExploratoryAnalysis(interactions, None, analytics_config).run()
        assert any("category" in note.lower() for note in result.notes)
        assert result.summary["n_items"] > 0

    def test_save_writes_json_and_markdown(self, interactions, products, analytics_config):
        eda = ExploratoryAnalysis(interactions, products, analytics_config)
        eda.run()
        json_path, md_path = eda.save()
        assert json_path.exists() and md_path.read_text().startswith("# Exploratory")

    def test_empty_input_rejected(self, analytics_config):
        with pytest.raises(ValueError, match="non-empty"):
            ExploratoryAnalysis(pd.DataFrame(), None, analytics_config)


# =============================================================================
# Outlier indicators
# =============================================================================
class TestOutliers:
    def _report(self, interactions, config):
        profiles = build_user_profiles(interactions)
        detector = OutlierDetector(config)
        detector.fit(profiles, scope="test fixture")
        return detector, detector.score(profiles)

    def test_scores_cover_every_user(self, interactions, analytics_config):
        _, report = self._report(interactions, analytics_config)
        assert len(report.scores) == interactions["user_id"].nunique()

    def test_shill_ranks_among_the_most_anomalous(self, interactions, analytics_config):
        _, report = self._report(interactions, analytics_config)
        rank = int(report.scores.loc["USHILL", "anomaly_rank"])
        assert rank <= max(3, int(0.05 * len(report.scores))), (
            f"the engineered shill ranked {rank} of {len(report.scores)}"
        )

    def test_every_flag_carries_a_reason(self, interactions, analytics_config):
        _, report = self._report(interactions, analytics_config)
        flagged = report.scores[report.scores["n_indicators"] > 0]
        assert (flagged["flag_reasons"].str.len() > 0).all()

    def test_no_output_column_claims_fraud(self, interactions, analytics_config):
        _, report = self._report(interactions, analytics_config)
        banned = {"is_fake", "fake", "fraud", "is_fraud", "fraudulent", "is_bot"}
        assert not banned & {c.lower() for c in report.scores.columns}

    def test_report_states_there_is_no_ground_truth(self, interactions, analytics_config):
        _, report = self._report(interactions, analytics_config)
        text = " ".join(report.notes).lower()
        assert "no fraud labels" in text or "ground truth" in text

    def test_report_records_fitted_scope_for_leakage_audit(self, interactions,
                                                           analytics_config):
        _, report = self._report(interactions, analytics_config)
        assert report.fitted_scope == "test fixture"
        assert any("training split" in note for note in report.notes)

    def test_higher_score_always_means_more_anomalous(self, interactions, analytics_config):
        _, report = self._report(interactions, analytics_config)
        flagged = report.scores[report.scores["iforest_flag"]]["iforest_score"]
        clean = report.scores[~report.scores["iforest_flag"]]["iforest_score"]
        assert flagged.min() > clean.max()

    def test_transductive_detector_refuses_unseen_rows(self, interactions, analytics_config):
        detector, _ = self._report(interactions, analytics_config)
        profiles = build_user_profiles(interactions).iloc[:5]
        with pytest.raises(RuntimeError, match="transductive"):
            detector.score(profiles)

    def test_inductive_mode_scores_unseen_users(self, interactions, analytics_config):
        config = analytics_config.with_overrides({"outliers.fit_on_train_split_only": True})
        profiles = build_user_profiles(interactions)
        detector = OutlierDetector(config)
        assert detector.mode == "inductive"
        detector.fit(profiles.iloc[:30], scope="train split only")
        report = detector.score(profiles.iloc[30:])
        assert len(report.scores) == len(profiles) - 30

    def test_missing_features_are_reported(self, interactions, analytics_config):
        minimal = interactions[["user_id", "item_id", "rating", "event_time"]]
        profiles = build_user_profiles(minimal)
        detector = OutlierDetector(analytics_config)
        detector.fit(profiles)
        report = detector.score(profiles)
        assert "mean_review_length" in report.features_missing
        assert "mean_review_length" not in report.features_used

    def test_high_iqr_rate_is_explained_as_multiple_comparisons(self, interactions,
                                                                analytics_config):
        _, report = self._report(interactions, analytics_config)
        if float((report.scores["iqr_flags"] > 0).mean()) > 0.25:
            assert any("multiple-comparisons" in note for note in report.notes)

    def test_save_writes_report(self, interactions, analytics_config):
        detector, report = self._report(interactions, analytics_config)
        json_path, md_path = detector.save(report)
        assert json_path.exists()
        assert "not fraud labels" in md_path.read_text()


# =============================================================================
# Graph analytics
# =============================================================================
class TestGraphStats:
    def test_bipartite_graph_shape(self, interactions):
        graph = build_bipartite_graph(interactions)
        assert graph.number_of_nodes() == (
            interactions["user_id"].nunique() + interactions["item_id"].nunique()
        )
        assert graph.number_of_edges() == len(
            interactions.drop_duplicates(["user_id", "item_id"])
        )

    def test_node_ids_cannot_collide(self):
        frame = pd.DataFrame({"user_id": ["X"], "item_id": ["X"]})
        graph = build_bipartite_graph(frame)
        assert graph.number_of_nodes() == 2

    def test_graph_is_actually_bipartite(self, interactions):
        assert nx.is_bipartite(build_bipartite_graph(interactions))

    def test_run_produces_core_metrics(self, interactions, analytics_config):
        report = GraphAnalyzer(build_bipartite_graph(interactions), analytics_config).run()
        for key in ("n_nodes", "n_edges", "density", "bipartite_density",
                    "n_connected_components", "n_communities", "modularity"):
            assert key in report.metrics, key

    def test_bipartite_density_uses_the_right_denominator(self, interactions,
                                                          analytics_config):
        graph = build_bipartite_graph(interactions)
        report = GraphAnalyzer(graph, analytics_config).run()
        n_users = interactions["user_id"].nunique()
        n_items = interactions["item_id"].nunique()
        assert report.metrics["bipartite_density"] == pytest.approx(
            graph.number_of_edges() / (n_users * n_items)
        )
        assert report.metrics["bipartite_density"] != report.metrics["density"]

    def test_triangle_clustering_zero_is_explained_not_reported_as_a_finding(
        self, interactions, analytics_config
    ):
        report = GraphAnalyzer(build_bipartite_graph(interactions), analytics_config).run()
        assert report.metrics["triangle_clustering_mean"] == 0.0
        assert "bipartite_clustering_mean" in report.metrics
        assert any("odd cycles" in note for note in report.notes)

    def test_approximated_metrics_are_labelled(self, interactions, analytics_config):
        report = GraphAnalyzer(build_bipartite_graph(interactions), analytics_config).run()
        # max_nodes_for_exact_metrics is 10 in the fixture, so both must sample.
        assert "betweenness" in report.approximations
        assert "closeness" in report.approximations
        assert "sample" in report.approximations["betweenness"]

    def test_exact_metrics_are_not_labelled_approximate(self, interactions):
        config = load_config("analytics_config").with_overrides({
            "graph.max_nodes_for_exact_metrics": 100000,
            "paths": {"figures": "reports/figures", "tables": "reports/tables"},
        })
        small = interactions[interactions["user_id"].isin(
            interactions["user_id"].unique()[:5]
        )]
        report = GraphAnalyzer(build_bipartite_graph(small), config).run()
        assert "betweenness" not in report.approximations

    def test_timings_recorded_for_every_stage(self, interactions, analytics_config):
        report = GraphAnalyzer(build_bipartite_graph(interactions), analytics_config).run()
        assert report.timings and all(v >= 0 for v in report.timings.values())

    def test_save_writes_report(self, interactions, analytics_config):
        analyzer = GraphAnalyzer(build_bipartite_graph(interactions), analytics_config)
        analyzer.run()
        json_path, md_path = analyzer.save()
        assert json_path.exists()
        assert md_path.read_text().startswith("# Graph Analytics")
