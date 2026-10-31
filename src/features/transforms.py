"""Leakage-safe numeric transforms.

Added beyond the brief's file list because the project needs exactly one place
where "fitted on training data only" is enforced, rather than a scaler
instantiated ad hoc in each feature module.

The contract: :meth:`LeakageSafeScaler.fit` may only ever be handed training
rows. It records what it was fitted on, and :meth:`transform` refuses to run
before a fit. Statistics computed here (means, medians, quantiles) are exactly
the values that silently import future information when fitted on all data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

__all__ = ["LeakageSafeScaler"]

logger = get_logger(__name__)


@dataclass
class LeakageSafeScaler:
    """Median-impute then standardise, fitted once on training rows.

    Attributes:
        columns: Columns to transform. Fixed at fit time so an unexpected column
            later cannot silently change the feature matrix width.
        clip_sigma: Values beyond this many standard deviations are clipped.
            Behavioural counts have extreme tails; without clipping a single
            user with 656 interactions dominates the scaled space.
    """

    clip_sigma: float = 8.0
    columns: list[str] = field(default_factory=list)
    medians: dict[str, float] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    log_columns: set[str] = field(default_factory=set)
    fitted_scope: str = "not fitted"
    _fitted: bool = False

    def fit(self, frame: pd.DataFrame, columns: list[str] | None = None,
            scope: str = "TRAIN SPLIT ONLY", log_skewed: bool = True) -> "LeakageSafeScaler":
        """Fit imputation and standardisation statistics.

        Args:
            frame: Rows to fit on. Must be the training partition.
            columns: Numeric columns to include. Defaults to all numeric columns.
            scope: Recorded description, so a leakage audit can read what was used.
            log_skewed: Apply log1p to non-negative columns whose skew exceeds 2,
                before computing the mean and standard deviation.
        """
        numeric = frame.select_dtypes(include=[np.number])
        self.columns = list(columns) if columns else list(numeric.columns)
        missing = [c for c in self.columns if c not in frame.columns]
        if missing:
            raise KeyError(f"Cannot fit scaler: columns absent from the frame: {missing}")

        for column in self.columns:
            values = pd.to_numeric(frame[column], errors="coerce").astype("float64")
            values = values.replace([np.inf, -np.inf], np.nan)
            median = float(values.median()) if values.notna().any() else 0.0
            self.medians[column] = median
            filled = values.fillna(median)

            if log_skewed and filled.min() >= 0 and abs(float(filled.skew())) > 2.0:
                self.log_columns.add(column)
                filled = np.log1p(filled)

            std = float(filled.std())
            self.means[column] = float(filled.mean())
            self.stds[column] = std if std > 1e-9 else 1.0

        self.fitted_scope = scope
        self._fitted = True
        logger.info("Scaler fitted on %s: %d columns (%d log-transformed)",
                    scope, len(self.columns), len(self.log_columns))
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transform. Columns absent from ``frame`` become 0."""
        if not self._fitted:
            raise RuntimeError("LeakageSafeScaler.transform called before fit().")

        out = pd.DataFrame(index=frame.index)
        for column in self.columns:
            if column not in frame.columns:
                out[column] = 0.0
                continue
            values = pd.to_numeric(frame[column], errors="coerce").astype("float64")
            values = values.replace([np.inf, -np.inf], np.nan).fillna(self.medians[column])
            if column in self.log_columns:
                values = np.log1p(values.clip(lower=0))
            scaled = (values - self.means[column]) / self.stds[column]
            out[column] = scaled.clip(-self.clip_sigma, self.clip_sigma)
        return out.astype("float32")

    def to_dict(self) -> dict[str, Any]:
        return {"fitted_scope": self.fitted_scope, "n_columns": len(self.columns),
                "columns": self.columns, "log_columns": sorted(self.log_columns),
                "clip_sigma": self.clip_sigma}
