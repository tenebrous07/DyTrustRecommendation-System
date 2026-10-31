"""Cold-start items: score a product the model has never seen from its text alone.

A trained recommender has an embedding for every item it saw in training and nothing
for a new product. The catalogue gains items all the time, and in this split about 60%
of test rows involve an item or a user that training never saw, so a model that cannot
score them is not usable as it stands.

The fix here is deliberately simple. The product's title and description are encoded
with Sentence-BERT, and a ridge regression, fitted on the items the model *does* know,
maps that text vector into the model's embedding space. A new product then gets an
embedding without any interaction. A second, tiny linear map puts the new item's
scores on the same scale as the learned items' scores, so one ranked list can mix both.
Nothing about test interactions is used: both maps are fitted on training items only.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

__all__ = ["ContentMapper"]


class ContentMapper:
    """Text vector -> item embedding, fitted on warm items."""

    def __init__(self, alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0), seed: int = 0) -> None:
        self.alphas = alphas
        self.seed = seed
        self.alpha_: float | None = None
        self.cv_r2_: float | None = None
        self.model_: Ridge | None = None
        self.scale_: tuple[float, float] = (1.0, 0.0)

    @staticmethod
    def _r2(truth: np.ndarray, pred: np.ndarray) -> float:
        residual = ((truth - pred) ** 2).sum()
        total = ((truth - truth.mean(axis=0, keepdims=True)) ** 2).sum()
        return float(1.0 - residual / total) if total > 0 else 0.0

    def fit(self, content: np.ndarray, embeddings: np.ndarray) -> "ContentMapper":
        """Choose the ridge strength by 5-fold cross-validation over the warm items."""
        best = (-np.inf, self.alphas[0])
        folds = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        for alpha in self.alphas:
            scores = []
            for train, held in folds.split(content):
                fitted = Ridge(alpha=alpha).fit(content[train], embeddings[train])
                scores.append(self._r2(embeddings[held], fitted.predict(content[held])))
            if np.mean(scores) > best[0]:
                best = (float(np.mean(scores)), alpha)
        self.cv_r2_, self.alpha_ = best
        self.model_ = Ridge(alpha=self.alpha_).fit(content, embeddings)
        return self

    def out_of_fold(self, content: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Predictions for every warm item from a model that never saw that item."""
        predicted = np.zeros_like(embeddings, dtype=np.float32)
        for train, held in KFold(n_splits=5, shuffle=True, random_state=self.seed).split(content):
            predicted[held] = Ridge(alpha=self.alpha_).fit(content[train], embeddings[train]).predict(content[held])
        return predicted

    def embed(self, content: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("fit() must be called before embed().")
        return self.model_.predict(content).astype(np.float32)

    def calibrate(self, user_emb: np.ndarray, learned: np.ndarray, mapped: np.ndarray,
                  sample_users: int = 2000, seed: int = 0) -> "ContentMapper":
        """Fit a * mapped_score + b to the learned scores on warm items (training items only).

        Mapped embeddings are smoother than learned ones, so their scores are compressed;
        without this a new item could never reach the top of a list that mixes both.
        """
        rng = np.random.default_rng(seed)
        users = rng.choice(len(user_emb), size=min(sample_users, len(user_emb)), replace=False)
        learned_scores = (user_emb[users] @ learned.T).ravel()
        mapped_scores = (user_emb[users] @ mapped.T).ravel()
        a, b = np.polyfit(mapped_scores, learned_scores, deg=1)
        self.scale_ = (float(a), float(b))
        return self

    def rescale(self, scores: np.ndarray) -> np.ndarray:
        a, b = self.scale_
        return a * scores + b
