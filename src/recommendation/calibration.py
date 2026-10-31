"""A calibrated probability that a recommended item will actually be taken.

The Module 18 confidence blend answers "is there enough evidence behind this
recommendation?" It does not answer "how likely is the user to take it?": on the test
split its AUROC for predicting hits is 0.456. Those are different questions and need
different numbers, so this is a second one.

A logistic regression is fitted on VALIDATION recommendations (did the user interact
with the recommended item in the validation period?) from five things known at
recommendation time, then scored once on test. Logistic regression is deliberate: with a
hit rate under 1% and a handful of features, a flexible model would mostly fit noise, and
the output is a probability by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

__all__ = ["HitProbabilityCalibrator", "expected_calibration_error", "FEATURES"]

FEATURES = ("score", "rank", "stability", "log_history", "trust")


class HitProbabilityCalibrator:
    """P(hit | score, rank, ensemble stability, history length, user trust)."""

    def __init__(self, regularisation: float = 1.0) -> None:
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=regularisation, max_iter=1000)

    @staticmethod
    def design(frame: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            frame["score"], np.log1p(frame["rank"]), frame["stability"],
            np.log1p(frame["n_history"]), frame["trust"]])

    def fit(self, frame: pd.DataFrame) -> "HitProbabilityCalibrator":
        self.model.fit(self.scaler.fit_transform(self.design(frame)), frame["hit"].to_numpy())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(self.design(frame)))[:, 1]

    def weights(self) -> dict[str, float]:
        names = ("score", "log(rank)", "stability", "log(history)", "trust")
        return dict(zip(names, map(float, self.model.coef_[0])))


def expected_calibration_error(probability: np.ndarray, outcome: np.ndarray, bins: int = 10) -> float:
    """Mean gap between predicted probability and observed rate, weighted by bin size.

    Bins hold equal numbers of items (quantiles), because with a base rate under 1% equal
    width bins would put nearly everything in the first one.
    """
    order = np.argsort(probability)
    chunks = np.array_split(order, bins)
    total = len(probability)
    return float(sum(len(c) / total * abs(probability[c].mean() - outcome[c].mean()) for c in chunks if len(c)))
