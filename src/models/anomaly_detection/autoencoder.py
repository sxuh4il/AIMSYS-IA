"""Autoencoder model architectures for anomaly detection."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


def _ensure_positive(value: int, name: str) -> None:
    """Validate that an integer hyperparameter is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}.")


class MLPAutoencoder(nn.Module):
    """Fully connected autoencoder for tabular feature vectors."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.0,
    ) -> None:
        """
        Create a lightweight MLP autoencoder.

        Args:
            input_dim: Number of input features.
            latent_dim: Dimension of the latent representation.
            hidden_dims: Hidden layer widths for the encoder.
            dropout: Optional dropout probability for hidden layers.
        """
        super().__init__()

        _ensure_positive(input_dim, "input_dim")
        _ensure_positive(latent_dim, "latent_dim")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}.")

        hidden_dims = tuple(hidden_dims)
        for hidden_dim in hidden_dims:
            _ensure_positive(int(hidden_dim), "hidden_dim")

        encoder_layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(previous_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(p=dropout))
            previous_dim = hidden_dim

        self.encoder_backbone = nn.Sequential(*encoder_layers)
        self.to_latent = nn.Linear(previous_dim, latent_dim)

        decoder_layers: list[nn.Module] = []
        previous_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(previous_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(p=dropout))
            previous_dim = hidden_dim

        self.decoder_backbone = nn.Sequential(*decoder_layers)
        self.reconstruction_head = nn.Linear(previous_dim, input_dim)

    def encode(self, inputs: Tensor) -> Tensor:
        """Encode input vectors into latent embeddings."""
        if inputs.dim() != 2:
            raise ValueError(
                f"MLPAutoencoder expects [batch, features] input, got {tuple(inputs.shape)}."
            )
        hidden = self.encoder_backbone(inputs)
        return self.to_latent(hidden)

    def decode(self, latent: Tensor) -> Tensor:
        """Decode latent embeddings back to reconstructed vectors."""
        if latent.dim() != 2:
            raise ValueError(
                f"MLPAutoencoder expects [batch, latent_dim] latent tensor, got {tuple(latent.shape)}."
            )
        hidden = self.decoder_backbone(latent)
        return self.reconstruction_head(hidden)

    def forward(self, inputs: Tensor) -> Tensor:
        """Run end-to-end reconstruction."""
        latent = self.encode(inputs)
        return self.decode(latent)


class LSTMAutoencoder(nn.Module):
    """LSTM autoencoder for sequence-like numerical feature vectors."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        """
        Create an LSTM-based autoencoder for temporal or sequence-like features.

        Args:
            input_dim: Number of features per time step.
            latent_dim: Latent bottleneck dimension.
            hidden_dim: Hidden size used by encoder/decoder LSTMs.
            num_layers: Number of LSTM layers.
            dropout: LSTM dropout applied between stacked layers.
        """
        super().__init__()

        _ensure_positive(input_dim, "input_dim")
        _ensure_positive(latent_dim, "latent_dim")
        _ensure_positive(hidden_dim, "hidden_dim")
        _ensure_positive(num_layers, "num_layers")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}.")

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.reconstruction_head = nn.Linear(hidden_dim, input_dim)

    def encode(self, inputs: Tensor) -> Tensor:
        """Encode an input sequence into a fixed-size latent vector."""
        if inputs.dim() != 3:
            raise ValueError(
                "LSTMAutoencoder expects [batch, sequence_length, features] input, "
                f"got {tuple(inputs.shape)}."
            )
        _, (hidden_state, _) = self.encoder_lstm(inputs)
        final_hidden = hidden_state[-1]
        return self.to_latent(final_hidden)

    def decode(self, latent: Tensor, sequence_length: int) -> Tensor:
        """Decode latent vectors into reconstructed sequences."""
        if latent.dim() != 2:
            raise ValueError(
                "LSTMAutoencoder expects [batch, latent_dim] latent tensor, "
                f"got {tuple(latent.shape)}."
            )
        _ensure_positive(sequence_length, "sequence_length")

        seed = torch.tanh(self.from_latent(latent))
        repeated_seed = seed.unsqueeze(1).repeat(1, sequence_length, 1)
        decoded_hidden, _ = self.decoder_lstm(repeated_seed)
        return self.reconstruction_head(decoded_hidden)

    def forward(self, inputs: Tensor) -> Tensor:
        """Run sequence reconstruction."""
        latent = self.encode(inputs)
        return self.decode(latent, sequence_length=inputs.size(1))
