"""Thresholding strategies for reconstruction-error-based anomaly detection."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from torch import Tensor

ThresholdMethod = Literal["percentile", "zscore", "mad"]


class AnomalyThreshold:
    """Fit and apply anomaly score thresholds in an unsupervised setup."""

    def __init__(
        self,
        method: ThresholdMethod = "percentile",
        percentile: float = 99.0,
        std_factor: float = 3.0,
        mad_factor: float = 3.5,
    ) -> None:
        """
        Initialize thresholding strategy.

        Args:
            method: Strategy name.
            percentile: Percentile for method="percentile".
            std_factor: Multiplier for method="zscore" threshold.
            mad_factor: Multiplier for method="mad" threshold.
        """
        self.method: ThresholdMethod = method
        self.percentile: float = percentile
        self.std_factor: float = std_factor
        self.mad_factor: float = mad_factor
        self.threshold: float | None = None

    def fit(self, scores: Sequence[float] | np.ndarray | Tensor) -> float:
        """Fit a threshold from normal-behavior anomaly scores."""
        values = _to_numpy(scores)
        if values.size == 0:
            raise ValueError("scores must not be empty.")

        if self.method == "percentile":
            if not (0.0 <= self.percentile <= 100.0):
                raise ValueError("percentile must be in [0, 100].")
            threshold = float(np.percentile(values, self.percentile))
        elif self.method == "zscore":
            if self.std_factor < 0.0:
                raise ValueError("std_factor must be >= 0.")
            mean = float(np.mean(values))
            std = float(np.std(values))
            threshold = mean + (self.std_factor * std)
        elif self.method == "mad":
            if self.mad_factor < 0.0:
                raise ValueError("mad_factor must be >= 0.")
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            robust_std = 1.4826 * mad
            threshold = median + (self.mad_factor * robust_std)
        else:
            raise ValueError(f"Unsupported threshold method: {self.method}.")

        self.threshold = threshold
        return threshold

    def predict(self, scores: Sequence[float] | np.ndarray | Tensor) -> np.ndarray:
        """Predict anomaly labels where True indicates anomalous behavior."""
        if self.threshold is None:
            raise RuntimeError("Threshold is not fitted. Call fit first.")
        values = _to_numpy(scores)
        return values > self.threshold

    def is_anomaly(self, score: float) -> bool:
        """Classify a single anomaly score."""
        if self.threshold is None:
            raise RuntimeError("Threshold is not fitted. Call fit first.")
        return score > self.threshold

    def configure(
        self,
        method: ThresholdMethod | None = None,
        percentile: float | None = None,
        std_factor: float | None = None,
        mad_factor: float | None = None,
    ) -> None:
        """Update threshold configuration parameters."""
        if method is not None:
            self.method = method
        if percentile is not None:
            self.percentile = percentile
        if std_factor is not None:
            self.std_factor = std_factor
        if mad_factor is not None:
            self.mad_factor = mad_factor


def _to_numpy(values: Sequence[float] | np.ndarray | Tensor) -> np.ndarray:
    """Convert supported score containers to a flat float64 numpy array."""
    if isinstance(values, np.ndarray):
        array = values
    elif isinstance(values, Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)

    return np.asarray(array, dtype=np.float64).reshape(-1)
