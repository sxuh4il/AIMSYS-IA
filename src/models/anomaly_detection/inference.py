"""
Inference module for the autoencoder-based anomaly detection system.
Provides forward pass, reconstruction error computation, and anomaly scoring.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Union
from dataclasses import dataclass


@dataclass
class AnomalyResult:
    """Result of anomaly detection inference."""
    reconstruction_errors: np.ndarray  # Per-sample reconstruction errors
    is_anomaly: np.ndarray  # Boolean array indicating anomalies
    threshold: float  # Threshold used for detection
    latent_representations: Optional[np.ndarray] = None  # Latent space embeddings
    reconstructed_data: Optional[np.ndarray] = None  # Reconstructed samples


class AnomalyDetector:
    """
    Anomaly detector using autoencoder reconstruction error.
    
    Samples with high reconstruction error are flagged as anomalies,
    since the autoencoder trained on normal data cannot accurately
    reconstruct anomalous patterns.
    """
    
    def __init__(
        self,
        model: nn.Module,
        threshold: Optional[float] = None,
        threshold_percentile: float = 95.0,
        device: str = "cpu"
    ):
        """
        Initialize the anomaly detector.
        
        Args:
            model: Trained autoencoder model
            threshold: Fixed threshold for anomaly detection (optional)
            threshold_percentile: Percentile for dynamic threshold computation
            device: Device for inference ('cpu' or 'cuda')
        """
        self.model = model
        self.threshold = threshold
        self.threshold_percentile = threshold_percentile
        self.device = device
        
        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def forward_pass(
        self, 
        data: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform forward pass through the autoencoder.
        
        Args:
            data: Input data (tensor or numpy array)
            
        Returns:
            Tuple of (reconstructed_output, latent_representation)
        """
        # Convert to tensor if needed
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)
        
        # Ensure correct shape
        if data.dim() == 1:
            data = data.unsqueeze(0)
        
        # Move to device
        data = data.to(self.device)
        
        # Forward pass
        reconstructed, latent = self.model(data)
        
        return reconstructed, latent
    
    @torch.no_grad()
    def compute_reconstruction_error(
        self,
        data: Union[torch.Tensor, np.ndarray],
        reduction: str = "mean"
    ) -> np.ndarray:
        """
        Compute reconstruction error for input samples.
        
        Args:
            data: Input data (tensor or numpy array)
            reduction: How to reduce feature-wise errors ('mean', 'sum', 'none')
            
        Returns:
            Array of reconstruction errors per sample
        """
        # Convert to tensor if needed
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)
        
        # Ensure correct shape
        if data.dim() == 1:
            data = data.unsqueeze(0)
        
        # Move to device
        data = data.to(self.device)
        
        # Get reconstruction
        reconstructed, _ = self.model(data)
        
        # Compute MSE per sample
        mse = (data - reconstructed) ** 2
        
        if reduction == "mean":
            errors = mse.mean(dim=1)
        elif reduction == "sum":
            errors = mse.sum(dim=1)
        else:  # none
            errors = mse
        
        return errors.cpu().numpy()
    
    def compute_threshold(
        self,
        normal_data: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """
        Compute anomaly threshold from normal data reconstruction errors.
        
        Args:
            normal_data: Data known to be normal for threshold computation
            
        Returns:
            Computed threshold value
        """
        errors = self.compute_reconstruction_error(normal_data)
        self.threshold = float(np.percentile(errors, self.threshold_percentile))
        return self.threshold
    
    @torch.no_grad()
    def detect(
        self,
        data: Union[torch.Tensor, np.ndarray],
        return_details: bool = False
    ) -> Union[np.ndarray, AnomalyResult]:
        """
        Detect anomalies in input data.
        
        Args:
            data: Input data to analyze
            return_details: Whether to return detailed results
            
        Returns:
            Boolean array of anomaly flags, or AnomalyResult if return_details=True
        """
        if self.threshold is None:
            raise RuntimeError(
                "Threshold not set. Call compute_threshold() with normal data first, "
                "or provide threshold in constructor."
            )
        
        # Compute reconstruction errors
        errors = self.compute_reconstruction_error(data)
        
        # Classify as anomaly if error exceeds threshold
        is_anomaly = errors > self.threshold
        
        if not return_details:
            return is_anomaly
        
        # Get detailed results
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)
        if data.dim() == 1:
            data = data.unsqueeze(0)
        
        data = data.to(self.device)
        reconstructed, latent = self.model(data)
        
        return AnomalyResult(
            reconstruction_errors=errors,
            is_anomaly=is_anomaly,
            threshold=self.threshold,
            latent_representations=latent.cpu().numpy(),
            reconstructed_data=reconstructed.cpu().numpy()
        )
    
    def get_anomaly_scores(
        self,
        data: Union[torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        """
        Get normalized anomaly scores (0 = normal, 1 = highly anomalous).
        
        Args:
            data: Input data to score
            
        Returns:
            Array of anomaly scores in [0, 1] range
        """
        if self.threshold is None:
            raise RuntimeError("Threshold not set.")
        
        errors = self.compute_reconstruction_error(data)
        
        # Normalize scores relative to threshold
        # Score of 1.0 means error equals threshold
        scores = errors / self.threshold
        
        return scores
    
    def get_feature_importance(
        self,
        data: Union[torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        """
        Get per-feature reconstruction errors to identify which features
        contribute most to anomaly detection.
        
        Args:
            data: Input data (single sample or batch)
            
        Returns:
            Array of per-feature reconstruction errors
        """
        return self.compute_reconstruction_error(data, reduction="none")


def compute_mse_loss(
    original: torch.Tensor,
    reconstructed: torch.Tensor
) -> torch.Tensor:
    """
    Compute Mean Squared Error between original and reconstructed data.
    
    Args:
        original: Original input tensor
        reconstructed: Reconstructed output tensor
        
    Returns:
        MSE loss value
    """
    return nn.functional.mse_loss(reconstructed, original)


def compute_mae_loss(
    original: torch.Tensor,
    reconstructed: torch.Tensor
) -> torch.Tensor:
    """
    Compute Mean Absolute Error between original and reconstructed data.
    
    Args:
        original: Original input tensor
        reconstructed: Reconstructed output tensor
        
    Returns:
        MAE loss value
    """
    return nn.functional.l1_loss(reconstructed, original)


def batch_inference(
    model: nn.Module,
    data: Union[torch.Tensor, np.ndarray],
    batch_size: int = 32,
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform batched inference for large datasets.
    
    Args:
        model: Autoencoder model
        data: Input data
        batch_size: Size of each batch
        device: Compute device
        
    Returns:
        Tuple of (all_reconstructed, all_errors)
    """
    model.eval()
    model.to(device)
    
    # Convert to tensor if needed
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
    
    num_samples = data.shape[0]
    all_reconstructed = []
    all_errors = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch = data[i:i + batch_size].to(device)
            reconstructed, _ = model(batch)
            
            # Compute per-sample MSE
            errors = ((batch - reconstructed) ** 2).mean(dim=1)
            
            all_reconstructed.append(reconstructed.cpu())
            all_errors.append(errors.cpu())
    
    return (
        torch.cat(all_reconstructed).numpy(),
        torch.cat(all_errors).numpy()
    )


def evaluate_detection_performance(
    detector: AnomalyDetector,
    data: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate anomaly detection performance given ground truth labels.
    
    Args:
        detector: Configured AnomalyDetector
        data: Test data
        labels: Ground truth labels (0 = normal, 1 = anomaly)
        
    Returns:
        Dictionary with precision, recall, f1, and accuracy metrics
    """
    predictions = detector.detect(data)
    
    # Compute metrics
    true_positives = np.sum((predictions == 1) & (labels == 1))
    false_positives = np.sum((predictions == 1) & (labels == 0))
    true_negatives = np.sum((predictions == 0) & (labels == 0))
    false_negatives = np.sum((predictions == 0) & (labels == 1))
    
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    accuracy = (true_positives + true_negatives) / len(labels)
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "accuracy": float(accuracy),
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "true_negatives": int(true_negatives),
        "false_negatives": int(false_negatives)
    }
