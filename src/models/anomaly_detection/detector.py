"""Integration-ready autoencoder anomaly detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from .losses import per_sample_reconstruction_error
from .thresholding import AnomalyThreshold, ThresholdMethod
from .trainer import AutoencoderTrainer, TrainingHistory, extract_features


@dataclass
class FitResult:
    """Output produced by the detector fit routine."""

    threshold: float
    history: TrainingHistory | None


@dataclass
class DetectionResult:
    """Batch detection output containing scores and anomaly labels."""

    scores: np.ndarray
    labels: np.ndarray
    threshold: float


class AutoencoderAnomalyDetector:
    """High-level interface for training and serving an autoencoder detector."""

    def __init__(
        self,
        model: nn.Module,
        trainer: AutoencoderTrainer | None = None,
        thresholding: AnomalyThreshold | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """
        Initialize detector.

        Args:
            model: Autoencoder model.
            trainer: Optional trainer used for fitting model parameters.
            thresholding: Optional thresholding strategy.
            device: Inference device if trainer is not provided.
        """
        self.model = model
        self.trainer = trainer
        self.thresholding = thresholding or AnomalyThreshold()

        if self.trainer is not None:
            self.device = self.trainer.device
        else:
            self.device = torch.device(device) if device is not None else torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.model.to(self.device)

    def fit(
        self,
        train_loader: Iterable[Any],
        epochs: int = 20,
        val_loader: Iterable[Any] | None = None,
        calibration_loader: Iterable[Any] | None = None,
        threshold_method: ThresholdMethod | None = None,
        percentile: float | None = None,
        std_factor: float | None = None,
        mad_factor: float | None = None,
    ) -> FitResult:
        """
        Fit model and threshold using normal-behavior data.

        Training is performed only when a trainer is provided.
        Threshold is fitted on calibration scores from calibration_loader, then
        validation loader if available, otherwise training loader.
        """
        history: TrainingHistory | None = None

        if self.trainer is not None and epochs > 0:
            history = self.trainer.fit(
                train_loader=train_loader,
                epochs=epochs,
                val_loader=val_loader,
            )
        elif self.trainer is None and epochs > 0:
            raise ValueError(
                "epochs > 0 requires a trainer. Provide AutoencoderTrainer or set epochs=0."
            )

        self.thresholding.configure(
            method=threshold_method,
            percentile=percentile,
            std_factor=std_factor,
            mad_factor=mad_factor,
        )

        threshold_loader = calibration_loader or val_loader or train_loader
        calibration_scores = self.score_loader(threshold_loader)
        threshold = self.thresholding.fit(calibration_scores)

        return FitResult(threshold=threshold, history=history)

    def reconstruct(self, inputs: Tensor) -> Tensor:
        """Return reconstructed output for a batch tensor."""
        prepared = self._prepare_tensor(inputs)
        self.model.eval()
        with torch.no_grad():
            return self.model(prepared)

    def score_loader(self, data_loader: Iterable[Any]) -> np.ndarray:
        """Compute anomaly scores for all batches in a data loader."""
        self.model.eval()
        all_scores: list[np.ndarray] = []

        with torch.no_grad():
            for batch in data_loader:
                inputs = self._prepare_batch(batch)
                reconstructions = self.model(inputs)
                batch_scores = per_sample_reconstruction_error(inputs, reconstructions)
                all_scores.append(batch_scores.detach().cpu().numpy())

        if not all_scores:
            raise ValueError("Data loader is empty.")

        return np.concatenate(all_scores, axis=0)

    def score_samples(self, inputs: Tensor | np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Compute anomaly scores for in-memory feature tensors or arrays."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}.")

        tensor = self._prepare_array_or_tensor(inputs)
        dataset = TensorDataset(tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return self.score_loader(loader)

    def predict_loader(self, data_loader: Iterable[Any]) -> DetectionResult:
        """Run anomaly prediction on a loader."""
        if self.thresholding.threshold is None:
            raise RuntimeError("Threshold is not fitted. Call fit first.")

        scores = self.score_loader(data_loader)
        labels = self.thresholding.predict(scores)
        return DetectionResult(scores=scores, labels=labels, threshold=self.thresholding.threshold)

    def predict_samples(
        self,
        inputs: Tensor | np.ndarray,
        batch_size: int = 256,
    ) -> DetectionResult:
        """Run anomaly prediction on in-memory feature vectors."""
        if self.thresholding.threshold is None:
            raise RuntimeError("Threshold is not fitted. Call fit first.")

        scores = self.score_samples(inputs, batch_size=batch_size)
        labels = self.thresholding.predict(scores)
        return DetectionResult(scores=scores, labels=labels, threshold=self.thresholding.threshold)

    def score_sample(self, input_sample: Tensor | np.ndarray) -> float:
        """Compute anomaly score for one sample for real-time inference."""
        score = self.score_samples(input_sample, batch_size=1)
        return float(score[0])

    def is_anomaly(self, input_sample: Tensor | np.ndarray) -> bool:
        """Classify a single sample as anomalous or normal."""
        score = self.score_sample(input_sample)
        return self.thresholding.is_anomaly(score)

    def _prepare_batch(self, batch: Any) -> Tensor:
        """Extract and prepare model inputs from a loader batch."""
        features = extract_features(batch)
        return self._prepare_tensor(features)

    def _prepare_tensor(self, tensor: Tensor) -> Tensor:
        """Move tensor to target device and ensure float32 dtype."""
        return tensor.to(device=self.device, dtype=torch.float32, non_blocking=True)

    def _prepare_array_or_tensor(self, inputs: Tensor | np.ndarray) -> Tensor:
        """Convert in-memory inputs to a batch tensor on the target device."""
        if isinstance(inputs, np.ndarray):
            tensor = torch.from_numpy(inputs)
        elif isinstance(inputs, Tensor):
            tensor = inputs
        else:
            raise TypeError(f"Unsupported input type: {type(inputs)!r}.")

        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        return self._prepare_tensor(tensor)
