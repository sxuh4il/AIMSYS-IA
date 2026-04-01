"""Training utilities for autoencoder-based anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .losses import reconstruction_loss


def extract_features(batch: Any) -> Tensor:
    """
    Extract the feature tensor from common DataLoader batch formats.

    Supports:
    - Tensor
    - Tuple/List where the first item is the feature tensor
    - Dict containing one of: features, x, inputs, data
    """
    if isinstance(batch, Tensor):
        return batch

    if isinstance(batch, (tuple, list)):
        if len(batch) == 0:
            raise ValueError("Batch tuple/list is empty.")
        if not isinstance(batch[0], Tensor):
            raise TypeError("First item in batch tuple/list must be a torch.Tensor.")
        return batch[0]

    if isinstance(batch, dict):
        for key in ("features", "x", "inputs", "data"):
            value = batch.get(key)
            if isinstance(value, Tensor):
                return value

        for value in batch.values():
            if isinstance(value, Tensor):
                return value

        raise TypeError("No tensor field found in dictionary batch.")

    raise TypeError(f"Unsupported batch type: {type(batch)!r}.")


@dataclass
class TrainingHistory:
    """Container for epoch-wise training metrics."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_loss: float = float("inf")


class AutoencoderTrainer:
    """Trainer for unsupervised autoencoder reconstruction learning."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: torch.device | str | None = None,
        clip_grad_norm: float | None = None,
    ) -> None:
        """
        Initialize trainer.

        Args:
            model: Autoencoder model to train.
            optimizer: Optimizer instance.
            device: Explicit compute device. If omitted, CUDA is used when available.
            clip_grad_norm: Optional gradient clipping max norm.
        """
        self.model = model
        self.optimizer = optimizer
        self.clip_grad_norm = clip_grad_norm
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def fit(
        self,
        train_loader: Iterable[Any],
        epochs: int,
        val_loader: Iterable[Any] | None = None,
    ) -> TrainingHistory:
        """
        Train model for a fixed number of epochs.

        Best weights are restored at the end based on validation loss if available,
        otherwise based on training loss.
        """
        if epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {epochs}.")

        history = TrainingHistory()
        best_state: dict[str, Tensor] | None = None

        for _ in range(epochs):
            train_loss = self._train_epoch(train_loader)
            history.train_losses.append(train_loss)

            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                history.val_losses.append(val_loss)
                metric = val_loss
            else:
                metric = train_loss

            if metric < history.best_loss:
                history.best_loss = metric
                best_state = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in self.model.state_dict().items()
                }

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        return history

    def evaluate(self, data_loader: Iterable[Any]) -> float:
        """Evaluate average reconstruction loss on a loader."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                inputs = self._prepare_inputs(batch)
                reconstructions = self.model(inputs)
                loss = reconstruction_loss(inputs, reconstructions, reduction="mean")
                total_loss += float(loss.item())
                num_batches += 1

        if num_batches == 0:
            raise ValueError("Data loader is empty.")

        return total_loss / num_batches

    def _train_epoch(self, train_loader: Iterable[Any]) -> float:
        """Run one optimization epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            inputs = self._prepare_inputs(batch)

            self.optimizer.zero_grad(set_to_none=True)
            reconstructions = self.model(inputs)
            loss = reconstruction_loss(inputs, reconstructions, reduction="mean")
            loss.backward()

            if self.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            self.optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        if num_batches == 0:
            raise ValueError("Training loader is empty.")

        return total_loss / num_batches

    def _prepare_inputs(self, batch: Any) -> Tensor:
        """Move extracted features to device and cast to float32."""
        features = extract_features(batch)
        return features.to(device=self.device, dtype=torch.float32, non_blocking=True)
