"""Shared setup for the experiment scripts.

``run_experiments.py``, ``tune_models.py`` and ``run_robustness.py`` all need the
same things: the chronological splits, a vocabulary built from training only, a
validation and a test evaluator, a negative sampler that excludes positives from
every split, and the graph. Building them once here keeps the three scripts from
drifting apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..anomaly.behavioural import BehaviouralTrustDetector
from ..anomaly.trust import TrustScores, build_trust_scores
from ..evaluation.evaluator import Evaluator, Vocabulary
from ..features.product_features import extract_category_levels
from ..features.review_features import ReviewFeatureBuilder
from ..graph.builder import HeteroGraphBuilder
from ..training.negative_sampling import NegativeSampler
from ..utils.config import load_config, resolve_path
from ..utils.device import resolve_device
from ..utils.logging import get_logger

__all__ = ["ExperimentContext", "paired_bootstrap"]

logger = get_logger(__name__)


class ExperimentContext:
    """Splits, vocabulary, evaluators and graph for one experiment run."""

    def __init__(self, device: str | None = None, train: pd.DataFrame | None = None,
                 embeddings: np.ndarray | None = None,
                 structured: pd.DataFrame | None = None) -> None:
        """Load splits and build evaluators.

        ``train``, ``embeddings`` and ``structured`` let the robustness study swap
        in a poisoned copy of the training split (with row-aligned review
        features) while validation and test stay exactly as they are.
        """
        self.data_config = load_config("data_config")
        self.model_config = load_config("model_config")
        self.training_config = load_config("training_config")
        self.experiment_config = load_config("experiment_config")

        self.processed = resolve_path(self.data_config.require("paths.processed"))
        self.results_dir = resolve_path(self.data_config.require("paths.results"))
        self.tables_dir = resolve_path(self.data_config.require("paths.tables"))
        for directory in (self.results_dir, self.tables_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.device = resolve_device(device or self.model_config.get("device", "auto"))

        self.train = (train if train is not None
                      else pd.read_parquet(self.processed / "split_train.parquet"))
        self.val = pd.read_parquet(self.processed / "split_val.parquet")
        self.test = pd.read_parquet(self.processed / "split_test.parquet")
        products_path = self.processed / "products_clean.parquet"
        self.products = pd.read_parquet(products_path) if products_path.is_file() else None

        self.vocab = Vocabulary.from_train(self.train)
        self.train_idx = self.vocab.index(self.train)

        # A negative must not be a positive in ANY split, or val/test items leak
        # into the negative pool.
        everything = self.vocab.index(pd.concat([self.train, self.val, self.test],
                                                ignore_index=True))
        everything = everything[(everything["user_idx"] >= 0) & (everything["item_idx"] >= 0)]
        self._positives = everything.groupby("user_idx")["item_idx"].apply(set).to_dict()

        self.item_categories = self._category_codes()
        k_values = list(self.experiment_config.get("k_values", [5, 10, 20]))
        self.val_eval = Evaluator(self.train, self.val, self.vocab, k_values=k_values,
                                  item_categories=self.item_categories, split_name="val")
        self.test_eval = Evaluator(self.train, self.test, self.vocab, k_values=k_values,
                                   item_categories=self.item_categories, split_name="test")

        self._bundles: dict[bool, object] = {}
        self._embeddings = embeddings
        self._structured = structured

    # -- refit protocol ----------------------------------------------------
    @classmethod
    def for_refit(cls, device: str | None = None) -> "ExperimentContext":
        """Context whose training split is train + validation.

        Hyper-parameters and epoch counts come from the validation phase; the
        final models are then refit on everything that precedes the test period,
        which is what a deployed system would do and which keeps the horizon
        between the last training interaction and the test period as short as it
        was between train and validation. Review features for the validation rows
        are computed once and cached.
        """
        from ..features.review_features import ReviewFeatureBuilder
        from ..features.sbert import SBERTEncoder
        from ..features.sentiment import add_sentiment_features

        base = cls(device)
        merged = pd.concat([base.train, base.val], ignore_index=True)

        cache = base.processed / "review_features_val.parquet"
        if cache.is_file():
            val_sentiment = pd.read_parquet(cache)["sentiment_compound"].to_numpy()
        else:
            scored = add_sentiment_features(base.val, model_config=base.model_config)
            val_sentiment = scored["sentiment_compound"].to_numpy(dtype=np.float32)
            pd.DataFrame({"sentiment_compound": val_sentiment}).to_parquet(cache, index=False)
        train_features = pd.read_parquet(base.processed / "review_features_train.parquet")
        sentiment = np.concatenate([train_features["sentiment_compound"].to_numpy(), val_sentiment])

        encoder = SBERTEncoder(base.model_config)
        embeddings = np.vstack([base.review_embeddings(), encoder.encode_frame(base.val)])

        builder = ReviewFeatureBuilder(
            lambda_per_year=float(base.model_config.get("temporal.lambda_per_year", 0.5)),
            min_weight=float(base.model_config.get("temporal.min_weight", 0.01))).fit(merged)
        features = builder.transform(merged.assign(sentiment_compound=sentiment))
        wanted = base.model_config.get("fusion.structured_features", [])
        structured = features[[c for c in wanted if c in features.columns]]
        logger.info("Refit context: %s training rows (train %s + validation %s).",
                    f"{len(merged):,}", f"{len(base.train):,}", f"{len(base.val):,}")
        return cls(device, train=merged, embeddings=embeddings, structured=structured)

    # -- helpers -----------------------------------------------------------
    def _category_codes(self) -> np.ndarray:
        if self.products is None:
            return np.zeros(self.vocab.n_items, dtype=np.int64)
        level = str(self.model_config.get("graph.category_level", "l2"))
        resolved = extract_category_levels(self.products, node_level=level)
        mapping = resolved.set_index("item_id")["category_node"]
        codes = pd.Series(self.vocab.item_ids).map(mapping).astype("category").cat.codes
        return codes.to_numpy()

    def sampler(self, seed: int, strategy: str | None = None) -> NegativeSampler:
        """A fresh sampler per call, so no two runs share random state."""
        strategy = strategy or str(self.training_config.get("negative_sampling.strategy", "uniform"))
        popularity = np.bincount(self.train_idx["item_idx"].to_numpy(),
                                 minlength=self.vocab.n_items).astype(np.float64) ** 0.75
        return NegativeSampler(
            n_items=self.vocab.n_items, positives_by_user=self._positives, strategy=strategy,
            item_popularity=popularity,
            max_attempts=int(self.training_config.get("negative_sampling.max_resample_attempts", 20)),
            seed=seed,
        )

    # -- graph -------------------------------------------------------------
    def review_embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            from ..features.sbert import SBERTEncoder
            self._embeddings = SBERTEncoder(self.model_config).encode_frame(self.train)
        return self._embeddings

    def structured_features(self) -> pd.DataFrame:
        if self._structured is None:
            frame = pd.read_parquet(self.processed / "review_features_train.parquet")
            wanted = self.model_config.get("fusion.structured_features", [])
            self._structured = frame[[c for c in wanted if c in frame.columns]]
        return self._structured

    def bundle(self, sbert: bool):
        """Training graph with or without SBERT review semantics (cached)."""
        if sbert not in self._bundles:
            # The builder still stores an edge weight; the model does not read
            # it, so the weights are applied through the loss instead.
            config = self.model_config.with_overrides({
                "edge_weighting.use_trust": False, "edge_weighting.use_temporal": False})
            self._bundles[sbert] = HeteroGraphBuilder(self.data_config, config).build(
                train=self.train, products=self.products,
                review_embeddings=self.review_embeddings() if sbert else None,
                review_structured=self.structured_features(),
            )
        return self._bundles[sbert]

    # -- trust and recency -------------------------------------------------
    def behavioural_scores(self, frame: pd.DataFrame | None = None) -> TrustScores:
        """Trust table from the behavioural detector, ordered like ``vocab.user_ids``."""
        frame = self.train if frame is None else frame
        result = BehaviouralTrustDetector().fit_score(frame, self.vocab.user_ids)
        return build_trust_scores(result, self.model_config, entity_type="user")

    def behavioural_trust(self, frame: pd.DataFrame | None = None) -> np.ndarray:
        """Per-user trust in [floor, 1], ordered like ``vocab.user_ids``."""
        return np.asarray(self.behavioural_scores(frame).trust, dtype=np.float32)

    def recency(self, lambda_per_year: float | None = None) -> np.ndarray:
        """Exponential recency weight per training row."""
        lam = (lambda_per_year if lambda_per_year is not None
               else float(self.model_config.get("temporal.lambda_per_year", 0.5)))
        builder = ReviewFeatureBuilder(
            lambda_per_year=lam,
            min_weight=float(self.model_config.get("temporal.min_weight", 0.01)),
        ).fit(self.train)
        return builder.recency_weight(self.train["timestamp"]).to_numpy(dtype=np.float32)

    def row_trust(self, user_trust: np.ndarray) -> np.ndarray:
        """Spread per-user trust onto the training rows."""
        return user_trust[self.train_idx["user_idx"].to_numpy()].astype(np.float32)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, draws: int = 4000, seed: int = 0) -> dict:
    """Paired bootstrap over users for the difference of two per-user metrics."""
    difference = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(difference)
    means = np.array([difference[rng.integers(0, n, n)].mean() for _ in range(draws)])
    low, high = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(difference.mean()), "ci_low": float(low),
            "ci_high": float(high), "p_a_better": float((means > 0).mean())}
