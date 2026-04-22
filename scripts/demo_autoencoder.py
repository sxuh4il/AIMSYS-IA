#!/usr/bin/env python3
"""
Demo script for the Autoencoder-based Anomaly Detection System.

This script demonstrates the complete pipeline:
1. Generate dummy automotive event data
2. Preprocess data (normalization)
3. Initialize autoencoder model
4. Perform forward pass (inference)
5. Compute reconstruction errors
6. Detect anomalies

Note: This demo uses a randomly initialized model (no training).
The purpose is to validate the pipeline architecture and data flow.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np

from models.anomaly_detection import (
    DenseAutoencoder,
    DataPreprocessor,
    DummyDataGenerator,
    AnomalyDetector,
    create_autoencoder_from_config
)
from configs import AutoencoderConfig, DataConfig


def main():
    print("=" * 70)
    print("Autoencoder-based Anomaly Detection System - Pipeline Demo")
    print("=" * 70)
    
    # =========================================================================
    # Step 1: Configuration
    # =========================================================================
    print("\n[1] Loading Configuration...")
    
    model_config = AutoencoderConfig(
        input_dim=10,
        encoder_dims=[64, 32, 16],
        latent_dim=8,
        decoder_dims=[16, 32, 64],
        dropout_rate=0.1,
        use_batch_norm=True
    )
    
    data_config = DataConfig(
        num_samples=1000,
        num_features=10,
        anomaly_ratio=0.05,
        random_seed=42
    )
    
    print(f"    Input dimensions: {model_config.input_dim}")
    print(f"    Latent dimensions: {model_config.latent_dim}")
    print(f"    Encoder layers: {model_config.encoder_dims}")
    print(f"    Number of samples: {data_config.num_samples}")
    print(f"    Anomaly ratio: {data_config.anomaly_ratio}")
    
    # =========================================================================
    # Step 2: Generate Dummy Data
    # =========================================================================
    print("\n[2] Generating Dummy Automotive Event Data...")
    
    generator = DummyDataGenerator(random_seed=data_config.random_seed)
    data, labels = generator.generate_dataset(
        num_samples=data_config.num_samples,
        anomaly_ratio=data_config.anomaly_ratio
    )
    
    num_normal = int(np.sum(labels == 0))
    num_anomaly = int(np.sum(labels == 1))
    
    print(f"    Generated {len(data)} samples")
    print(f"    Normal samples: {num_normal}")
    print(f"    Anomalous samples: {num_anomaly}")
    print(f"    Feature names: {generator.feature_names}")
    print(f"    Data shape: {data.shape}")
    
    # Show sample event
    print("\n    Sample event:")
    for name, value in zip(generator.feature_names, data[0]):
        print(f"      {name}: {value:.2f}")
    
    # =========================================================================
    # Step 3: Preprocess Data (Normalization)
    # =========================================================================
    print("\n[3] Preprocessing Data (Min-Max Normalization)...")
    
    preprocessor = DataPreprocessor()
    normalized_data = preprocessor.fit_transform(data, generator.feature_names)
    
    print(f"    Original data range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"    Normalized data range: [{normalized_data.min():.4f}, {normalized_data.max():.4f}]")
    
    # Convert to tensor
    data_tensor = torch.tensor(normalized_data, dtype=torch.float32)
    print(f"    Tensor shape: {data_tensor.shape}")
    print(f"    Tensor dtype: {data_tensor.dtype}")
    
    # =========================================================================
    # Step 4: Initialize Autoencoder Model
    # =========================================================================
    print("\n[4] Initializing Dense Autoencoder Model...")
    
    model = create_autoencoder_from_config(model_config)
    
    # Print model architecture
    print(model.get_model_summary())
    
    # =========================================================================
    # Step 5: Forward Pass (Inference)
    # =========================================================================
    print("\n[5] Performing Forward Pass (Inference)...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        reconstructed, latent = model(data_tensor)
    
    print(f"    Input shape: {data_tensor.shape}")
    print(f"    Latent representation shape: {latent.shape}")
    print(f"    Reconstructed output shape: {reconstructed.shape}")
    
    # Show sample latent representation
    print(f"\n    Sample latent representation (first 3 samples):")
    for i in range(min(3, len(latent))):
        latent_str = ", ".join([f"{v:.3f}" for v in latent[i].numpy()])
        print(f"      Sample {i}: [{latent_str}]")
    
    # =========================================================================
    # Step 6: Compute Reconstruction Error
    # =========================================================================
    print("\n[6] Computing Reconstruction Errors...")
    
    # Initialize anomaly detector with model
    detector = AnomalyDetector(
        model=model,
        threshold_percentile=95.0,
        device="cpu"
    )
    
    # Compute reconstruction errors
    errors = detector.compute_reconstruction_error(data_tensor)
    
    print(f"    Error statistics:")
    print(f"      Min: {errors.min():.6f}")
    print(f"      Max: {errors.max():.6f}")
    print(f"      Mean: {errors.mean():.6f}")
    print(f"      Std: {errors.std():.6f}")
    
    # Compute threshold from "normal" data (using all data for demo)
    threshold = detector.compute_threshold(data_tensor)
    print(f"\n    Computed anomaly threshold (95th percentile): {threshold:.6f}")
    
    # =========================================================================
    # Step 7: Detect Anomalies
    # =========================================================================
    print("\n[7] Detecting Anomalies...")
    
    # Detect anomalies
    result = detector.detect(data_tensor, return_details=True)
    
    detected_anomalies = np.sum(result.is_anomaly)
    print(f"    Detected {detected_anomalies} anomalies out of {len(data)} samples")
    print(f"    Detection rate: {100 * detected_anomalies / len(data):.2f}%")
    
    # Compare with ground truth
    true_positives = np.sum((result.is_anomaly == True) & (labels == 1))
    false_positives = np.sum((result.is_anomaly == True) & (labels == 0))
    print(f"\n    True positives (correctly detected anomalies): {true_positives}")
    print(f"    False positives (normal flagged as anomaly): {false_positives}")
    
    # Show top anomalies by reconstruction error
    print("\n    Top 5 samples with highest reconstruction error:")
    top_indices = np.argsort(errors)[-5:][::-1]
    for idx in top_indices:
        is_true_anomaly = "ANOMALY" if labels[idx] == 1 else "normal"
        print(f"      Sample {idx}: error={errors[idx]:.6f} (ground truth: {is_true_anomaly})")
    
    # =========================================================================
    # Step 8: Get Anomaly Scores
    # =========================================================================
    print("\n[8] Computing Anomaly Scores...")
    
    scores = detector.get_anomaly_scores(data_tensor)
    
    print(f"    Score statistics (1.0 = at threshold):")
    print(f"      Min: {scores.min():.4f}")
    print(f"      Max: {scores.max():.4f}")
    print(f"      Mean: {scores.mean():.4f}")
    
    # Show score distribution
    bins = [0, 0.5, 0.75, 1.0, 1.5, float('inf')]
    labels_bin = ['0-0.5', '0.5-0.75', '0.75-1.0', '1.0-1.5', '>1.5']
    print("\n    Score distribution:")
    for i in range(len(bins) - 1):
        count = np.sum((scores >= bins[i]) & (scores < bins[i + 1]))
        print(f"      {labels_bin[i]}: {count} samples")
    
    # =========================================================================
    # Step 9: Feature Importance for Anomalies
    # =========================================================================
    print("\n[9] Analyzing Feature Importance for Detected Anomalies...")
    
    # Get per-feature errors for detected anomalies
    anomaly_indices = np.where(result.is_anomaly)[0][:5]  # First 5 anomalies
    
    if len(anomaly_indices) > 0:
        feature_errors = detector.get_feature_importance(data_tensor[anomaly_indices])
        avg_feature_errors = feature_errors.mean(axis=0)
        
        print("    Average per-feature reconstruction error for anomalies:")
        sorted_features = np.argsort(avg_feature_errors)[::-1]
        for idx in sorted_features:
            print(f"      {generator.feature_names[idx]}: {avg_feature_errors[idx]:.6f}")
    else:
        print("    No anomalies detected to analyze.")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Pipeline Validation Complete!")
    print("=" * 70)
    print("""
Summary:
- Autoencoder architecture: Validated
- Dummy data generation: Working
- Data preprocessing: Working  
- Forward pass (inference): Working
- Reconstruction error computation: Working
- Anomaly detection: Working

Note: This demo uses a randomly initialized model (no training).
Model performance will improve significantly after proper training
on normal event data. The pipeline is ready for training integration.
""")


if __name__ == "__main__":
    main()
