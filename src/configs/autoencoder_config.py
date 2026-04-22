"""
Configuration parameters for the Autoencoder-based anomaly detection system.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AutoencoderConfig:
    """Configuration for the Dense Autoencoder model."""
    
    # Input features (automotive event data)
    input_dim: int = 10
    
    # Encoder layer dimensions (progressively compress)
    encoder_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    
    # Latent space dimension (bottleneck)
    latent_dim: int = 8
    
    # Decoder layer dimensions (progressively expand)
    decoder_dims: List[int] = field(default_factory=lambda: [16, 32, 64])
    
    # Activation function
    activation: str = "relu"
    
    # Dropout rate for regularization
    dropout_rate: float = 0.1
    
    # Whether to use batch normalization
    use_batch_norm: bool = True


@dataclass
class DataConfig:
    """Configuration for data generation and preprocessing."""
    
    # Number of dummy samples to generate
    num_samples: int = 1000
    
    # Number of features per event
    num_features: int = 10
    
    # Feature names for automotive event data
    feature_names: List[str] = field(default_factory=lambda: [
        "app_id",
        "action_id", 
        "resource_id",
        "bytes_sent",
        "bytes_received",
        "duration_ms",
        "cpu_usage",
        "memory_usage",
        "network_latency",
        "error_code"
    ])
    
    # Percentage of anomalies in dummy data
    anomaly_ratio: float = 0.05
    
    # Random seed for reproducibility
    random_seed: int = 42


@dataclass
class InferenceConfig:
    """Configuration for inference and anomaly detection."""
    
    # Threshold percentile for anomaly detection
    threshold_percentile: float = 95.0
    
    # Batch size for inference
    batch_size: int = 32
    
    # Device for inference (cpu/cuda)
    device: str = "cpu"


# Default configurations
DEFAULT_MODEL_CONFIG = AutoencoderConfig()
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_INFERENCE_CONFIG = InferenceConfig()
