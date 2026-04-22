"""
Dense Autoencoder architecture for anomaly detection in automotive event data.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class Encoder(nn.Module):
    """
    Encoder network that compresses input features to a latent representation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Final layer to latent space
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.network(x)


class Decoder(nn.Module):
    """
    Decoder network that reconstructs input from latent representation.
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        layers = []
        prev_dim = latent_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Final reconstruction layer (no activation for normalized data)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstructed output."""
        return self.network(z)


class DenseAutoencoder(nn.Module):
    """
    Dense (fully connected) Autoencoder for anomaly detection.
    
    This autoencoder learns to compress and reconstruct normal event patterns.
    Anomalies are detected by high reconstruction error, as the model
    struggles to reconstruct patterns it hasn't learned during training.
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        encoder_dims: List[int] = None,
        latent_dim: int = 8,
        decoder_dims: List[int] = None,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        # Default hidden dimensions if not provided
        encoder_dims = encoder_dims or [64, 32, 16]
        decoder_dims = decoder_dims or [16, 32, 64]
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder and decoder
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=encoder_dims,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
        
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_dims,
            output_dim=input_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (reconstructed_output, latent_representation)
        """
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output."""
        return self.decoder(z)
    
    def get_model_summary(self) -> str:
        """Return a string summary of the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        summary = [
            "=" * 60,
            "Dense Autoencoder Architecture",
            "=" * 60,
            f"Input Dimension: {self.input_dim}",
            f"Latent Dimension: {self.latent_dim}",
            "-" * 60,
            "Encoder:",
            str(self.encoder.network),
            "-" * 60,
            "Decoder:", 
            str(self.decoder.network),
            "-" * 60,
            f"Total Parameters: {total_params:,}",
            f"Trainable Parameters: {trainable_params:,}",
            "=" * 60
        ]
        return "\n".join(summary)


def create_autoencoder_from_config(config) -> DenseAutoencoder:
    """
    Factory function to create a DenseAutoencoder from configuration.
    
    Args:
        config: AutoencoderConfig object with model parameters
        
    Returns:
        Initialized DenseAutoencoder model
    """
    return DenseAutoencoder(
        input_dim=config.input_dim,
        encoder_dims=config.encoder_dims,
        latent_dim=config.latent_dim,
        decoder_dims=config.decoder_dims,
        dropout_rate=config.dropout_rate,
        use_batch_norm=config.use_batch_norm
    )
