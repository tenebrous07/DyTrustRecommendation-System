"""Recommender, negative-sampling, loss, explanation and confidence invariants."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.evaluator import Evaluator, Vocabulary
from src.explainability.counterfactual import CounterfactualAnalyzer
from src.explainability.explanations import Explainer
from src.explainability.graph_paths import EvidenceRetriever
from src.models.collaborative_filtering import ItemKNNRecommender
from src.models.matrix_factorization import MatrixFactorizationRecommender
from src.models.popularity import PopularityRecommender
from src.recommendation.confidence import ConfidenceEstimator
from src.training.losses import bpr_loss, normalise_weights
from src.training.negative_sampling import NegativeSampler
from src.utils.config import Config


@pytest.fixture()
def frames():
    rng = np.random.default_rng(5)
    base = pd.Timestamp("2020-01-01", tz="UTC")
    rows = []
    for user in range(25):
        for _ in range(8):
            item = int(rng.integers(0, 15))
            when = base + pd.Timedelta(days=int(rng.integers(0, 700)))
            rows.append({"user_id": f"U{user:03d}", "item_id": f"P{item:03d}",
                         "rating": float(rng.integers(1, 6)),
                         "review_text": "solid product overall", "event_time": when,
                         "timestamp": int(when.timestamp() * 1000)})
    frame = pd.DataFrame(rows).drop_duplicates(["user_id", "item_id"], keep="last")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    cut = int(len(frame) * 0.8)
    return frame.iloc[:cut].reset_index(drop=True), frame.iloc[cut:].reset_index(drop=True)


@pytest.fixture()
def products():
    return pd.DataFrame([{
        "item_id": f"P{i:03d}", "product_title": f"Product {i}",
        "brand": ["Nintendo", "Sony"][i % 2],
        "category_node": ["playstation 4", "xbox one", "pc"][i % 3],
    } for i in range(15)])


@pytest.fixture()
def setup(frames):
    train, test = frames
    vocab = Vocabulary.from_train(train)
    indexed = vocab.index(train)
    positives = indexed.groupby("user_idx")["item_idx"].apply(set).to_dict()
    sampler = NegativeSampler(vocab.n_items, positives, seed=42)
    return train, test, vocab, indexed, sampler


class TestNegativeSampling:
    def test_shape(self, setup):
        _, _, vocab, indexed, sampler = setup
        users = indexed["user_idx"].to_numpy()[:20]
        assert sampler.sample(users, 3).shape == (20, 3)

    def test_negatives_are_not_known_positives(self, setup):
        _, _, vocab, indexed, sampler = setup
        users = indexed["user_idx"].to_numpy()
        negatives = sampler.sample(users, 1).ravel()
        positives = indexed.groupby("user_idx")["item_idx"].apply(set).to_dict()
        collisions = sum(1 for u, n in zip(users, negatives) if n in positives.get(u, set()))
        assert collisions == 0

    def test_in_range(self, setup):
        _, _, vocab, indexed, sampler = setup
        negatives = sampler.sample(indexed["user_idx"].to_numpy(), 2)
        assert negatives.min() >= 0 and negatives.max() < vocab.n_items

    def test_reproducible_for_a_seed(self, setup):
        _, _, vocab, indexed, _ = setup
        positives = indexed.groupby("user_idx")["item_idx"].apply(set).to_dict()
        users = indexed["user_idx"].to_numpy()[:50]
        a = NegativeSampler(vocab.n_items, positives, seed=7).sample(users, 1)
        b = NegativeSampler(vocab.n_items, positives, seed=7).sample(users, 1)
        assert np.array_equal(a, b)


class TestLosses:
    def test_bpr_is_lower_when_positive_wins(self):
        good = bpr_loss(torch.tensor([3.0]), torch.tensor([0.0]))
        bad = bpr_loss(torch.tensor([0.0]), torch.tensor([3.0]))
        assert good < bad

    def test_bpr_at_ties_is_ln2(self):
        assert bpr_loss(torch.zeros(4), torch.zeros(4)).item() == pytest.approx(np.log(2), 1e-5)

    def test_bpr_is_non_negative_and_finite(self):
        loss = bpr_loss(torch.randn(64), torch.randn(64))
        assert loss.item() >= 0 and torch.isfinite(loss)

    def test_weight_normalisation_gives_mean_one(self):
        weights = normalise_weights(torch.rand(500) * 0.4 + 0.05)
        assert weights.mean().item() == pytest.approx(1.0, abs=1e-5)

    def test_trust_weights_change_the_loss(self):
        pos, neg = torch.randn(32), torch.randn(32)
        plain = bpr_loss(pos, neg)
        weighted = bpr_loss(pos, neg, weights=torch.rand(32))
        assert not torch.isclose(plain, weighted)

    def test_gradients_flow(self):
        pos = torch.randn(16, requires_grad=True)
        bpr_loss(pos, torch.randn(16)).backward()
        assert torch.isfinite(pos.grad).all()


class TestRecommenders:
    def test_popularity_shape_and_identical_rows(self, setup):
        _, _, vocab, indexed, _ = setup
        model = PopularityRecommender(vocab.n_users, vocab.n_items).fit(indexed)
        scores = model.score_users(np.array([0, 1, 2]))
        assert scores.shape == (3, vocab.n_items)
        assert np.allclose(scores[0], scores[1])

    def test_popularity_matches_counts(self, setup):
        _, _, vocab, indexed, _ = setup
        model = PopularityRecommender(vocab.n_users, vocab.n_items).fit(indexed)
        expected = indexed["item_idx"].value_counts().idxmax()
        assert int(np.argmax(model.score_users(np.array([0]))[0])) == expected

    def test_item_knn_never_recommends_the_same_item(self, setup):
        _, _, vocab, indexed, _ = setup
        model = ItemKNNRecommender(vocab.n_users, vocab.n_items, k=5).fit(indexed)
        assert np.allclose(np.diag(model.similarity), 0.0)

    def test_mf_shape_and_finite(self, setup):
        _, _, vocab, indexed, sampler = setup
        model = MatrixFactorizationRecommender(
            vocab.n_users, vocab.n_items, embedding_dim=8, epochs=3, seed=1)
        model.fit(indexed, sampler=sampler)
        scores = model.score_users(np.array([0, 1]))
        assert scores.shape == (2, vocab.n_items) and np.isfinite(scores).all()

    def test_score_before_fit_raises(self, setup):
        _, _, vocab, _, _ = setup
        with pytest.raises(RuntimeError, match="fit"):
            PopularityRecommender(vocab.n_users, vocab.n_items).score_users(np.array([0]))


class TestEvaluator:
    def test_masks_training_positives(self, setup):
        train, test, vocab, indexed, _ = setup
        evaluator = Evaluator(train, test, vocab, k_values=[5], split_name="test")
        model = PopularityRecommender(vocab.n_users, vocab.n_items).fit(indexed)
        result = evaluator.evaluate(model)
        assert 0.0 <= result.metrics["ndcg@5"] <= 1.0

    def test_cold_rows_are_counted_not_silently_dropped(self, setup):
        train, test, vocab, _, _ = setup
        evaluator = Evaluator(train, test, vocab, k_values=[5], split_name="test")
        model = PopularityRecommender(vocab.n_users, vocab.n_items).fit(vocab.index(train))
        diagnostics = evaluator.evaluate(model).diagnostics
        assert "cold_user_rows_excluded" in diagnostics
        assert "cold_item_rows_excluded" in diagnostics

    def test_wrong_score_shape_is_rejected(self, setup):
        train, test, vocab, _, _ = setup
        evaluator = Evaluator(train, test, vocab, k_values=[5])

        class Broken:
            name = "broken"
            def score_users(self, users):
                return np.zeros((len(users), 3))
        with pytest.raises(ValueError, match="expected"):
            evaluator.evaluate(Broken())


class TestExplanations:
    def test_evidence_is_retrieved_from_real_data(self, setup, products):
        train, _, vocab, _, _ = setup
        retriever = EvidenceRetriever(train, products, vocab)
        user = train["user_id"].iloc[0]
        item = train["item_id"].iloc[0]
        evidence = retriever.retrieve(user, item)
        assert all(0.0 <= e.strength <= 1.0 or e.kind == "semantic_similarity"
                   for e in evidence)

    def test_no_evidence_yields_an_honest_statement_not_a_fabrication(self, setup, products):
        train, _, vocab, _, _ = setup
        retriever = EvidenceRetriever(train, products, vocab)
        explanation = Explainer(retriever).explain("NOT_A_USER", "P001")
        assert explanation.has_support is False
        assert "no graph evidence" in explanation.text.lower()

    def test_supported_explanation_mentions_only_retrieved_facts(self, setup, products):
        train, _, vocab, _, _ = setup
        retriever = EvidenceRetriever(train, products, vocab)
        explainer = Explainer(retriever)
        user = train["user_id"].iloc[0]
        for item in train["item_id"].unique()[:5]:
            explanation = explainer.explain(user, item)
            if explanation.has_support:
                assert explanation.evidence
                break


class TestCounterfactual:
    def _analyzer(self, vocab, dim=16):
        rng = np.random.default_rng(3)
        users = rng.normal(size=(vocab.n_users, dim)).astype(np.float32)
        items = rng.normal(size=(vocab.n_items, dim)).astype(np.float32)
        categories = np.arange(vocab.n_items) % 3
        return CounterfactualAnalyzer(users, items, vocab, item_categories=categories)

    def test_already_top_k_is_reported_as_such(self, setup):
        _, _, vocab, _, _ = setup
        analyzer = self._analyzer(vocab)
        user = vocab.user_ids[0]
        scores = analyzer.item_embeddings @ analyzer.user_embeddings[0]
        best = vocab.item_ids[int(np.argmax(scores))]
        result = analyzer.analyse(user, best, k=5)
        assert result.feasible and "already ranked" in result.text

    def test_result_is_either_feasible_with_alpha_or_explicitly_infeasible(self, setup):
        _, _, vocab, _, _ = setup
        analyzer = self._analyzer(vocab)
        user = vocab.user_ids[0]
        scores = analyzer.item_embeddings @ analyzer.user_embeddings[0]
        worst = vocab.item_ids[int(np.argmin(scores))]
        result = analyzer.analyse(user, worst, k=3)
        if result.feasible:
            assert result.required_alpha is not None and result.achieved_rank <= 3
        else:
            assert "no tested factor" in result.text

    def test_unknown_ids_rejected(self, setup):
        _, _, vocab, _, _ = setup
        with pytest.raises(KeyError):
            self._analyzer(vocab).analyse("NOPE", "ALSO_NOPE", k=5)


class TestConfidence:
    def _estimator(self, counts, trust):
        config = Config({
            "confidence": {"method": "heuristic", "mc_samples": 3,
                           "components": {"predictive_stability": 0.5,
                                          "evidence_volume": 0.3, "evidence_trust": 0.2},
                           "cold_start_penalty": True},
            "cold_start": {"min_interactions_for_personalised": 3},
        }, name="c")
        return ConfidenceEstimator(config, user_interaction_counts=counts, trust_map=trust)

    def test_bounded(self):
        estimator = self._estimator({"U1": 50}, {"U1": 0.9})
        estimates = estimator.estimate(None, "U1", 0, np.array([0, 1]), ["a", "b"])
        assert all(0.0 <= e.confidence <= 1.0 for e in estimates)

    def test_cold_start_user_gets_low_confidence(self):
        """The behaviour that makes confidence worth having at all."""
        warm = self._estimator({"U1": 100}, {"U1": 0.9})
        cold = self._estimator({"U2": 1}, {"U2": 0.9})
        warm_conf = warm.estimate(None, "U1", 0, np.array([0]), ["a"])[0].confidence
        cold_conf = cold.estimate(None, "U2", 0, np.array([0]), ["a"])[0].confidence
        assert cold_conf < warm_conf
        assert cold_conf < 0.5

    def test_reason_explains_the_penalty(self):
        estimator = self._estimator({"U2": 1}, {"U2": 0.5})
        estimate = estimator.estimate(None, "U2", 0, np.array([0]), ["a"])[0]
        assert "cold-start" in estimate.reason

    def test_more_evidence_raises_confidence(self):
        low = self._estimator({"U1": 5}, {"U1": 0.5})
        high = self._estimator({"U1": 200}, {"U1": 0.5})
        assert (high.estimate(None, "U1", 0, np.array([0]), ["a"])[0].confidence >
                low.estimate(None, "U1", 0, np.array([0]), ["a"])[0].confidence)

    def test_supplied_stability_is_used_instead_of_mc_dropout(self):
        estimator = self._estimator({"U1": 50}, {"U1": 0.9})
        stable = estimator.estimate(None, "U1", 0, np.array([0]), ["a"],
                                    stability=np.array([1.0]))[0]
        shaky = estimator.estimate(None, "U1", 0, np.array([0]), ["a"],
                                   stability=np.array([0.0]))[0]
        assert stable.confidence > shaky.confidence
        assert "predictive_stability" in stable.components

    def test_ensemble_stability_is_high_when_members_agree(self):
        agree = np.tile(np.array([[1.0, 2.0, 3.0]]), (4, 1))
        disagree = np.array([[1.0, 2.0, 3.0], [3.0, -2.0, 0.5], [-1.0, 4.0, 2.0], [0.0, 0.0, 9.0]])
        assert ConfidenceEstimator.ensemble_stability(agree).min() > 0.99
        assert (ConfidenceEstimator.ensemble_stability(disagree).mean()
                < ConfidenceEstimator.ensemble_stability(agree).mean())
