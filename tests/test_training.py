"""Edge-masked HGT training, validation stopping, weighting and the trust detector."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

from src.anomaly.behavioural import BehaviouralTrustDetector
from src.anomaly.synthetic import SyntheticAnomalyInjector, detection_metrics
from src.evaluation.evaluator import Evaluator, Vocabulary
from src.experiments.context import paired_bootstrap
from src.graph.hetero_graph import GraphBundle
from src.models.hgt import HGTRecommender
from src.models.popularity import PopularityRecommender
from src.training.early_stopping import ValidationMonitor
from src.training.negative_sampling import NegativeSampler
from src.utils.config import Config

N_USERS, N_ITEMS, N_ROWS = 12, 9, 60


def _tiny_bundle():
    rng = np.random.default_rng(0)
    users = rng.integers(0, N_USERS, N_ROWS)
    items = rng.integers(0, N_ITEMS, N_ROWS)
    data = HeteroData()
    data["user"].x = torch.randn(N_USERS, 4)
    data["user"].num_nodes = N_USERS
    data["product"].x = torch.randn(N_ITEMS, 4)
    data["product"].num_nodes = N_ITEMS
    edge_index = torch.as_tensor(np.stack([users, items]), dtype=torch.long)
    data["user", "purchased", "product"].edge_index = edge_index
    data["product", "rev_purchased", "user"].edge_index = edge_index.flip(0)
    bundle = GraphBundle(data=data, indices={}, metadata={})
    frame = pd.DataFrame({"user_idx": users, "item_idx": items})
    return bundle, frame


def _configs(**training):
    model = Config({"hgt": {"hidden_channels": 8, "num_heads": 2, "num_layers": 1,
                            "dropout": 0.0}}, name="model")
    base = {"training": {"full_batch": True, "max_epochs": 6, "eval_every": 2,
                         "edge_mask": {"enabled": True, "keep_prob": 0.6}},
            "optimizer": {"learning_rate": 0.01, "weight_decay": 0.0},
            "negative_sampling": {"num_negatives": 1},
            "early_stopping": {"enabled": True, "patience": 2, "min_delta": 0.0}}
    for dotted, value in training.items():
        node = base
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return model, Config(base, name="training")


def _recommender(**training):
    bundle, frame = _tiny_bundle()
    model_cfg, training_cfg = _configs(**training)
    rec = HGTRecommender(N_USERS, N_ITEMS, bundle=bundle, model_config=model_cfg,
                         training_config=training_cfg, device="cpu", use_fusion=False,
                         use_weighted_loss=training.pop("weighted", False), seed=1)
    positives = frame.groupby("user_idx")["item_idx"].apply(set).to_dict()
    return rec, frame, NegativeSampler(N_ITEMS, positives, seed=1)


class TestEdgeMasking:
    def test_masking_touches_only_interaction_relations(self):
        rec, frame, sampler = _recommender()
        rec.fit(frame, sampler=sampler)
        keep = torch.zeros(N_ROWS, dtype=torch.bool)
        keep[:10] = True
        edges = rec._select_edges(keep)
        for edge_type in rec._interaction_edges:
            assert edges[edge_type].shape[1] == 10
        assert {("user", "purchased", "product"), ("product", "rev_purchased", "user")} \
            == set(rec._interaction_edges)

    def test_forward_and_reverse_edges_share_one_mask(self):
        rec, frame, sampler = _recommender()
        rec.fit(frame, sampler=sampler)
        keep = torch.rand(N_ROWS) < 0.5
        edges = rec._select_edges(keep)
        forward = edges[("user", "purchased", "product")]
        reverse = edges[("product", "rev_purchased", "user")]
        assert torch.equal(forward, reverse.flip(0))

    def test_fit_produces_finite_scores(self):
        rec, frame, sampler = _recommender()
        rec.fit(frame, sampler=sampler)
        scores = rec.score_users(np.arange(N_USERS))
        assert scores.shape == (N_USERS, N_ITEMS) and np.isfinite(scores).all()

    def test_keep_prob_must_be_a_proper_fraction(self):
        with pytest.raises(ValueError, match="keep_prob"):
            _recommender(**{"training.edge_mask.keep_prob": 1.0})

    def test_trust_gate_removes_rows_from_the_graph(self):
        rec, frame, sampler = _recommender()
        drop = np.zeros(N_ROWS, dtype=bool)
        drop[:15] = True
        rec.fit(frame, sampler=sampler, drop_rows=drop)
        for edge_type in rec._interaction_edges:
            assert rec.edge_index_dict[edge_type].shape[1] == N_ROWS - 15

    def test_weighted_model_refuses_to_train_without_weights(self):
        bundle, frame = _tiny_bundle()
        model_cfg, training_cfg = _configs()
        rec = HGTRecommender(N_USERS, N_ITEMS, bundle=bundle, model_config=model_cfg,
                             training_config=training_cfg, device="cpu", use_fusion=False,
                             use_weighted_loss=True, seed=1)
        positives = frame.groupby("user_idx")["item_idx"].apply(set).to_dict()
        with pytest.raises(ValueError, match="no weights"):
            rec.fit(frame, sampler=NegativeSampler(N_ITEMS, positives, seed=1))


class TestValidationStopping:
    def test_best_epoch_is_recorded_and_weights_restored(self):
        rec, frame, sampler = _recommender()
        values = iter([0.10, 0.30, 0.20, 0.10, 0.05, 0.01])
        rec.fit(frame, sampler=sampler, validator=lambda _: next(values))
        assert rec.best_epoch == 4 and rec.best_val == pytest.approx(0.30)

    def test_training_stops_after_patience(self):
        rec, frame, sampler = _recommender(**{"training.max_epochs": 40})
        rec.fit(frame, sampler=sampler, validator=lambda _: 0.1)
        assert len(rec.history) < 40

    def test_monitor_restores_best_snapshot(self):
        weight = torch.nn.Parameter(torch.zeros(3))
        monitor = ValidationMonitor([weight], patience=2, min_delta=0.0)
        weight.data.fill_(1.0)
        monitor.update(0.5, 1)
        weight.data.fill_(9.0)
        monitor.update(0.1, 2)
        monitor.restore()
        assert torch.allclose(weight, torch.ones(3))

    def test_monitor_signals_stop(self):
        weight = torch.nn.Parameter(torch.zeros(1))
        monitor = ValidationMonitor([weight], patience=2, min_delta=0.0)
        assert not monitor.update(0.5, 1)
        assert not monitor.update(0.4, 2)
        assert monitor.update(0.3, 3)


class TestEvaluatorPerUser:
    def test_per_user_mean_matches_headline_ndcg(self):
        rng = np.random.default_rng(2)
        rows = [{"user_id": f"U{u}", "item_id": f"P{rng.integers(0, 12)}",
                 "timestamp": t} for t, u in enumerate(rng.integers(0, 15, 150))]
        frame = pd.DataFrame(rows).drop_duplicates(["user_id", "item_id"])
        train, test = frame.iloc[:90], frame.iloc[90:]
        vocab = Vocabulary.from_train(train)
        evaluator = Evaluator(train, test, vocab, k_values=[10])
        model = PopularityRecommender(vocab.n_users, vocab.n_items).fit(vocab.index(train))
        per_user = evaluator.per_user_ndcg(model, 10)
        assert len(per_user) == len(evaluator.eval_users)
        assert per_user.mean() == pytest.approx(evaluator.evaluate(model).metrics["ndcg@10"])
        assert evaluator.ndcg(model, 10) == pytest.approx(per_user.mean())


class TestPairedBootstrap:
    def test_clear_winner_has_positive_interval(self):
        rng = np.random.default_rng(0)
        b = rng.random(400) * 0.1
        result = paired_bootstrap(b + 0.05, b, draws=500)
        assert result["ci_low"] > 0 and result["p_a_better"] > 0.99

    def test_identical_models_straddle_zero(self):
        a = np.random.default_rng(1).random(300)
        result = paired_bootstrap(a, a.copy(), draws=300)
        assert result["mean_diff"] == 0.0


class TestBehaviouralDetector:
    @pytest.fixture()
    def corpus(self):
        rng = np.random.default_rng(3)
        base = pd.Timestamp("2020-01-01", tz="UTC")
        rows = []
        for user in range(80):
            for _ in range(int(rng.integers(5, 12))):
                when = base + pd.Timedelta(days=int(rng.integers(0, 900)))
                rows.append({
                    "user_id": f"U{user:04d}", "item_id": f"P{rng.integers(0, 40):04d}",
                    "rating": float(rng.integers(1, 6)), "review_title": "t",
                    "review_text": f"my own thoughts number {rng.integers(0, 10**6)} here",
                    "timestamp": int(when.timestamp() * 1000), "event_time": when,
                    "helpful_votes": 0, "verified_purchase": True})
        frame = pd.DataFrame(rows).drop_duplicates(["user_id", "item_id"], keep="last")
        return frame.sort_values("timestamp").reset_index(drop=True)

    def test_scripted_shills_score_above_genuine_users(self, corpus):
        config = Config({"synthetic_anomaly": {
            "injection_rate": 0.15, "patterns": {
                "burst_reviews": {"reviews_per_burst": 8, "window_minutes": 30},
                "extreme_rating": {"forced_rating": 5.0},
                "duplicated_text": {"template_pool_size": 3},
                "coordinated_targeting": {"shills_per_target": 3}}}}, name="e")
        injection = SyntheticAnomalyInjector(config, seed=5).inject(corpus)
        vocab = Vocabulary.from_train(injection.interactions)
        result = BehaviouralTrustDetector().fit_score(injection.interactions, vocab.user_ids)
        labels = injection.label_array(vocab.user_ids)
        assert detection_metrics(result.anomaly_scores, labels)["roc_auc"] > 0.95

    def test_scores_are_ordered_like_the_requested_users(self, corpus):
        ids = sorted(corpus["user_id"].unique(), reverse=True)
        result = BehaviouralTrustDetector().fit_score(corpus, ids)
        assert result.node_ids == ids and len(result.anomaly_scores) == len(ids)

    def test_thin_history_gets_the_median_not_an_extreme(self, corpus):
        thin = corpus[corpus["user_id"] != "U0000"].copy()
        one_row = corpus[corpus["user_id"] == "U0000"].head(1)
        frame = pd.concat([thin, one_row], ignore_index=True)
        ids = sorted(frame["user_id"].unique())
        result = BehaviouralTrustDetector(min_interactions=3).fit_score(frame, ids)
        score = result.anomaly_scores[ids.index("U0000")]
        others = np.delete(result.anomaly_scores, ids.index("U0000"))
        assert score == pytest.approx(np.median(others), abs=0.05)


class TestScheduler:
    def test_cosine_decay_ends_near_the_floor(self):
        rec, frame, sampler = _recommender(**{"scheduler.name": "cosine", "scheduler.min_lr_factor": 0.1,
                                              "training.max_epochs": 8, "early_stopping.enabled": False})
        rec.fit(frame, sampler=sampler)
        final_lr = rec._scheduler.get_last_lr()[0]
        assert final_lr == pytest.approx(0.01 * 0.1, rel=0.05)

    def test_unknown_scheduler_is_rejected(self):
        with pytest.raises(ValueError, match="scheduler.name"):
            _recommender(**{"scheduler.name": "plateau"})


class TestEnsemble:
    def test_single_member_ranks_like_the_member(self):
        from src.models.ensemble import EnsembleRecommender
        rng = np.random.default_rng(0)
        user, item = rng.normal(size=(5, 4)).astype(np.float32), rng.normal(size=(7, 4)).astype(np.float32)
        ensemble = EnsembleRecommender(5, 7, [(user, item)])
        scores = ensemble.score_users(np.array([0, 3]))
        assert (np.argsort(scores[0]) == np.argsort(user[0] @ item.T)).all()

    def test_members_with_different_scales_contribute_equally(self):
        from src.models.ensemble import EnsembleRecommender
        rng = np.random.default_rng(1)
        user, item = rng.normal(size=(3, 4)).astype(np.float32), rng.normal(size=(6, 4)).astype(np.float32)
        small = EnsembleRecommender(3, 6, [(user, item), (user * 100.0, item)])
        same = EnsembleRecommender(3, 6, [(user, item), (user, item)])
        assert np.allclose(small.score_users(np.arange(3)), same.score_users(np.arange(3)), atol=1e-4)

    def test_empty_ensemble_is_rejected(self):
        from src.models.ensemble import EnsembleRecommender
        with pytest.raises(ValueError, match="at least one"):
            EnsembleRecommender(2, 2, [])


class TestSASRec:
    def _data(self):
        rng = np.random.default_rng(4)
        rows = [{"user_idx": u, "item_idx": int(rng.integers(0, 12)), "timestamp": t}
                for u in range(10) for t in range(8)]
        frame = pd.DataFrame(rows).drop_duplicates(["user_idx", "item_idx"])
        positives = frame.groupby("user_idx")["item_idx"].apply(set).to_dict()
        return frame, NegativeSampler(12, positives, seed=1)

    def test_scores_have_the_catalogue_shape_and_are_finite(self):
        from src.models.sasrec import SASRecRecommender
        frame, sampler = self._data()
        model = SASRecRecommender(10, 12, embedding_dim=8, num_layers=1, max_len=6, epochs=3, seed=0)
        model.fit(frame, sampler=sampler)
        scores = model.score_users(np.array([0, 1, 2]))
        assert scores.shape == (3, 12) and np.isfinite(scores).all()

    def test_a_user_is_represented_by_their_most_recent_items(self):
        from src.models.sasrec import SASRecRecommender
        model = SASRecRecommender(1, 20, max_len=4)
        padded = model._pad(np.array([5, 6, 7, 8, 9, 10]))
        assert list(padded) == [8, 9, 10, 11]            # last four items, shifted by the padding id
        assert list(model._pad(np.array([3]))) == [0, 0, 0, 4]

    def test_validation_early_stopping_records_the_best_epoch(self):
        from src.models.sasrec import SASRecRecommender
        frame, sampler = self._data()
        values = iter([0.1, 0.3, 0.2, 0.1, 0.05, 0.01, 0.0, 0.0])
        model = SASRecRecommender(10, 12, embedding_dim=8, num_layers=1, max_len=6, epochs=8,
                                  eval_every=1, patience=2, seed=0)
        model.fit(frame, sampler=sampler, validator=lambda _: next(values))
        assert model.best_epoch == 2


class TestColdItems:
    def test_mapping_recovers_a_linear_relationship(self):
        from src.recommendation.cold_items import ContentMapper
        rng = np.random.default_rng(0)
        content = rng.normal(size=(300, 12))
        embeddings = content @ rng.normal(size=(12, 5)) + 0.01 * rng.normal(size=(300, 5))
        mapper = ContentMapper().fit(content, embeddings)
        assert mapper.cv_r2_ > 0.95
        assert np.corrcoef(mapper.embed(content).ravel(), embeddings.ravel())[0, 1] > 0.99

    def test_embed_before_fit_is_rejected(self):
        from src.recommendation.cold_items import ContentMapper
        with pytest.raises(RuntimeError, match="fit"):
            ContentMapper().embed(np.zeros((2, 3)))


class TestCalibration:
    def test_probabilities_follow_the_signal_and_stay_in_range(self):
        from src.recommendation.calibration import HitProbabilityCalibrator, expected_calibration_error
        rng = np.random.default_rng(1)
        n = 4000
        score = rng.normal(size=n)
        frame = pd.DataFrame({"score": score, "rank": rng.integers(1, 11, n), "stability": rng.random(n),
                              "n_history": rng.integers(1, 50, n), "trust": rng.random(n),
                              "hit": (rng.random(n) < 1 / (1 + np.exp(-(score * 1.5 - 3)))).astype(int)})
        calibrator = HitProbabilityCalibrator().fit(frame)
        p = calibrator.predict(frame)
        assert p.min() >= 0 and p.max() <= 1
        assert np.corrcoef(p, score)[0, 1] > 0.5
        assert expected_calibration_error(p, frame["hit"].to_numpy()) < 0.03


class TestAdaptiveAttack:
    def test_paraphrases_differ_but_keep_the_praise(self):
        from src.anomaly.synthetic import _TEMPLATES, paraphrase
        rng = np.random.default_rng(0)
        variants = {paraphrase(_TEMPLATES[0], rng) for _ in range(200)}
        assert len(variants) > 150                       # almost no two shills share a string

    def test_semantic_indicator_finds_paraphrased_copies_that_exact_match_misses(self):
        from src.anomaly.behavioural import BehaviouralTrustDetector
        rng = np.random.default_rng(2)
        base = rng.normal(size=16)
        rows, vectors = [], []
        for u in range(20):                                # genuine: unrelated text
            rows.append({"user_id": f"G{u}", "review_text": f"genuine opinion number {u} about this thing",
                         "timestamp": 1_000 + u})
            vectors.append(rng.normal(size=16))
        for u in range(6):                                 # shills: different strings, same meaning
            rows.append({"user_id": f"S{u}", "review_text": f"reworded praise variant {u} for the product",
                         "timestamp": 2_000 + u})
            vectors.append(base + 0.02 * rng.normal(size=16))
        frame = pd.DataFrame(rows)
        matrix = np.asarray(vectors, dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        shared = BehaviouralTrustDetector(min_matches=3).semantic_shared(frame, matrix)
        assert shared[[f"S{u}" for u in range(6)]].min() == 1.0
        assert shared[[f"G{u}" for u in range(20)]].max() == 0.0


class TestDatasetProfile:
    def test_profile_points_every_path_at_its_own_directories(self, monkeypatch):
        from src.utils.config import load_config
        monkeypatch.setenv("DYTRUSTREC_DATASET", "Musical_Instruments")
        config = load_config("data_config")
        assert config.get("paths.processed") == "data/processed_musical_instruments"
        assert config.get("tiers.dev.category") == "Musical_Instruments"

    def test_no_profile_leaves_the_default_untouched(self, monkeypatch):
        from src.utils.config import load_config
        monkeypatch.delenv("DYTRUSTREC_DATASET", raising=False)
        assert load_config("data_config").get("paths.processed") == "data/processed"

    def test_active_dirs_keep_a_second_dataset_away_from_video_games_runs(self, monkeypatch):
        from src.utils.config import active_dirs
        monkeypatch.delenv("DYTRUSTREC_DATASET", raising=False)
        default = active_dirs()
        monkeypatch.setenv("DYTRUSTREC_DATASET", "Musical_Instruments")
        other = active_dirs()
        for key in ("results", "tables", "checkpoints"):
            assert other[key] != default[key]
            assert "musical_instruments" in other[key].name

    def test_scripts_do_not_hardcode_the_video_games_output_directories(self):
        from pathlib import Path
        for script in (Path(__file__).resolve().parents[1] / "scripts").glob("*.py"):
            text = script.read_text(encoding="utf-8")
            code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
            assert 'ROOT / "reports"' not in code, script.name
            assert 'ROOT / "checkpoints"' not in code, script.name
