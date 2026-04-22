"""
Data preprocessing utilities for the anomaly detection system.
Provides normalization and data transformation functions.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any


class DataPreprocessor:
    """
    Preprocessor for normalizing and transforming automotive event data.
    
    Uses min-max normalization to scale features to [0, 1] range,
    which is suitable for autoencoder-based anomaly detection.
    """
    
    def __init__(self):
        self.min_vals: Optional[np.ndarray] = None
        self.max_vals: Optional[np.ndarray] = None
        self.feature_names: Optional[list] = None
        self._is_fitted: bool = False
    
    def fit(self, data: np.ndarray, feature_names: Optional[list] = None) -> "DataPreprocessor":
        """
        Fit the preprocessor on training data to compute normalization parameters.
        
        Args:
            data: Training data of shape (num_samples, num_features)
            feature_names: Optional list of feature names
            
        Returns:
            self for method chaining
        """
        self.min_vals = np.min(data, axis=0)
        self.max_vals = np.max(data, axis=0)
        self.feature_names = feature_names
        self._is_fitted = True
        
        # Handle constant features (avoid division by zero)
        self._range = self.max_vals - self.min_vals
        self._range[self._range == 0] = 1.0
        
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted normalization parameters.
        
        Args:
            data: Data to transform of shape (num_samples, num_features)
            
        Returns:
            Normalized data in [0, 1] range
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform. Call fit() first.")
        
        normalized = (data - self.min_vals) / self._range
        return normalized
    
    def fit_transform(self, data: np.ndarray, feature_names: Optional[list] = None) -> np.ndarray:
        """
        Fit the preprocessor and transform data in one step.
        
        Args:
            data: Training data of shape (num_samples, num_features)
            feature_names: Optional list of feature names
            
        Returns:
            Normalized data in [0, 1] range
        """
        self.fit(data, feature_names)
        return self.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Reverse the normalization to get original scale.
        
        Args:
            data: Normalized data of shape (num_samples, num_features)
            
        Returns:
            Data in original scale
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse_transform.")
        
        return data * self._range + self.min_vals
    
    def to_tensor(self, data: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Convert numpy array to PyTorch tensor.
        
        Args:
            data: Numpy array to convert
            dtype: Target tensor dtype
            
        Returns:
            PyTorch tensor
        """
        return torch.tensor(data, dtype=dtype)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get normalization parameters for saving/loading.
        
        Returns:
            Dictionary with min_vals, max_vals, and feature_names
        """
        if not self._is_fitted:
            return {}
        
        return {
            "min_vals": self.min_vals.tolist(),
            "max_vals": self.max_vals.tolist(),
            "feature_names": self.feature_names
        }
    
    def load_params(self, params: Dict[str, Any]) -> "DataPreprocessor":
        """
        Load normalization parameters from a dictionary.
        
        Args:
            params: Dictionary with min_vals, max_vals, and feature_names
            
        Returns:
            self for method chaining
        """
        self.min_vals = np.array(params["min_vals"])
        self.max_vals = np.array(params["max_vals"])
        self.feature_names = params.get("feature_names")
        self._range = self.max_vals - self.min_vals
        self._range[self._range == 0] = 1.0
        self._is_fitted = True
        return self
    
    @property
    def is_fitted(self) -> bool:
        """Check if preprocessor has been fitted."""
        return self._is_fitted


class StandardScaler:
    """
    Alternative preprocessor using z-score normalization (standardization).
    Transforms data to have zero mean and unit variance.
    """
    
    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self._is_fitted: bool = False
    
    def fit(self, data: np.ndarray) -> "StandardScaler":
        """Fit the scaler on training data."""
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.std[self.std == 0] = 1.0  # Avoid division by zero
        self._is_fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters."""
        if not self._is_fitted:
            raise RuntimeError("Scaler must be fitted before transform.")
        return (data - self.mean) / self.std
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse the standardization."""
        if not self._is_fitted:
            raise RuntimeError("Scaler must be fitted before inverse_transform.")
        return data * self.std + self.mean


def prepare_data_for_inference(
    data: np.ndarray,
    preprocessor: DataPreprocessor,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Prepare raw data for model inference.
    
    Args:
        data: Raw data array of shape (num_samples, num_features)
        preprocessor: Fitted DataPreprocessor instance
        device: Target device ('cpu' or 'cuda')
        
    Returns:
        Preprocessed tensor ready for model inference
    """
    # Normalize data
    normalized = preprocessor.transform(data)
    
    # Convert to tensor
    tensor = torch.tensor(normalized, dtype=torch.float32)
    
    # Move to device
    tensor = tensor.to(device)
    
    return tensor
