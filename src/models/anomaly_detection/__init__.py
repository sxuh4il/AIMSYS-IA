"""
Anomaly Detection Module for Automotive Cybersecurity.

This module provides autoencoder-based anomaly detection for event-based
cybersecurity data in automotive systems.

Components:
- DenseAutoencoder: Main autoencoder architecture
- DataPreprocessor: Data normalization and transformation
- DummyDataGenerator: Synthetic data generation for testing
- AnomalyDetector: Inference and anomaly scoring
"""

from .autoencoder import (
    DenseAutoencoder,
    Encoder,
    Decoder,
    create_autoencoder_from_config
)

from .preprocessor import (
    DataPreprocessor,
    StandardScaler,
    prepare_data_for_inference
)

from .data_generator import (
    DummyDataGenerator,
    FeatureSpec,
    DEFAULT_FEATURE_SPECS,
    create_sample_event,
    generate_training_validation_test_split
)

from .inference import (
    AnomalyDetector,
    AnomalyResult,
    compute_mse_loss,
    compute_mae_loss,
    batch_inference,
    evaluate_detection_performance
)


__all__ = [
    # Model
    "DenseAutoencoder",
    "Encoder",
    "Decoder",
    "create_autoencoder_from_config",
    
    # Preprocessing
    "DataPreprocessor",
    "StandardScaler",
    "prepare_data_for_inference",
    
    # Data Generation
    "DummyDataGenerator",
    "FeatureSpec",
    "DEFAULT_FEATURE_SPECS",
    "create_sample_event",
    "generate_training_validation_test_split",
    
    # Inference
    "AnomalyDetector",
    "AnomalyResult",
    "compute_mse_loss",
    "compute_mae_loss",
    "batch_inference",
    "evaluate_detection_performance",
]

__version__ = "0.1.0"
