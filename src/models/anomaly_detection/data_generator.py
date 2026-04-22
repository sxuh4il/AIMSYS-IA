"""
Dummy data generator for validating the anomaly detection pipeline.
Generates synthetic automotive event data with normal and anomalous patterns.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class FeatureSpec:
    """Specification for a single feature."""
    name: str
    min_val: float
    max_val: float
    dtype: str = "float"


# Default feature specifications for automotive event data
DEFAULT_FEATURE_SPECS: List[FeatureSpec] = [
    FeatureSpec("app_id", 1, 100, "int"),
    FeatureSpec("action_id", 1, 50, "int"),
    FeatureSpec("resource_id", 1, 200, "int"),
    FeatureSpec("bytes_sent", 0, 10000, "float"),
    FeatureSpec("bytes_received", 0, 10000, "float"),
    FeatureSpec("duration_ms", 1, 5000, "float"),
    FeatureSpec("cpu_usage", 0, 100, "float"),
    FeatureSpec("memory_usage", 0, 100, "float"),
    FeatureSpec("network_latency", 0, 500, "float"),
    FeatureSpec("error_code", 0, 10, "int"),
]


class DummyDataGenerator:
    """
    Generator for synthetic automotive event data.
    
    Creates realistic-looking event data with configurable normal patterns
    and anomalies for testing the anomaly detection pipeline.
    """
    
    def __init__(
        self,
        feature_specs: Optional[List[FeatureSpec]] = None,
        random_seed: int = 42
    ):
        """
        Initialize the data generator.
        
        Args:
            feature_specs: List of feature specifications (uses defaults if None)
            random_seed: Random seed for reproducibility
        """
        self.feature_specs = feature_specs or DEFAULT_FEATURE_SPECS
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
    
    @property
    def num_features(self) -> int:
        """Number of features."""
        return len(self.feature_specs)
    
    @property
    def feature_names(self) -> List[str]:
        """List of feature names."""
        return [spec.name for spec in self.feature_specs]
    
    def generate_normal_data(self, num_samples: int) -> np.ndarray:
        """
        Generate normal event data following expected patterns.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Array of shape (num_samples, num_features)
        """
        data = np.zeros((num_samples, self.num_features))
        
        for i, spec in enumerate(self.feature_specs):
            # Generate data within normal range with Gaussian distribution
            mean = (spec.min_val + spec.max_val) / 2
            std = (spec.max_val - spec.min_val) / 6  # 99.7% within range
            
            values = self.rng.normal(mean, std, num_samples)
            values = np.clip(values, spec.min_val, spec.max_val)
            
            if spec.dtype == "int":
                values = np.round(values)
            
            data[:, i] = values
        
        return data
    
    def generate_anomalies(self, num_samples: int, anomaly_type: str = "mixed") -> np.ndarray:
        """
        Generate anomalous event data.
        
        Args:
            num_samples: Number of anomalies to generate
            anomaly_type: Type of anomaly ('extreme', 'correlation', 'mixed')
            
        Returns:
            Array of shape (num_samples, num_features)
        """
        if anomaly_type == "extreme":
            return self._generate_extreme_anomalies(num_samples)
        elif anomaly_type == "correlation":
            return self._generate_correlation_anomalies(num_samples)
        else:  # mixed
            half = num_samples // 2
            extreme = self._generate_extreme_anomalies(half)
            correlation = self._generate_correlation_anomalies(num_samples - half)
            return np.vstack([extreme, correlation])
    
    def _generate_extreme_anomalies(self, num_samples: int) -> np.ndarray:
        """Generate anomalies with extreme feature values."""
        data = np.zeros((num_samples, self.num_features))
        
        for i, spec in enumerate(self.feature_specs):
            # Some values at extremes
            extreme_mask = self.rng.random(num_samples) < 0.3
            
            normal_mean = (spec.min_val + spec.max_val) / 2
            normal_std = (spec.max_val - spec.min_val) / 6
            
            values = self.rng.normal(normal_mean, normal_std, num_samples)
            
            # Push extreme values beyond normal range
            extreme_vals = self.rng.choice(
                [spec.min_val - normal_std, spec.max_val + normal_std],
                size=num_samples
            )
            values[extreme_mask] = extreme_vals[extreme_mask]
            
            if spec.dtype == "int":
                values = np.round(values)
            
            data[:, i] = values
        
        return data
    
    def _generate_correlation_anomalies(self, num_samples: int) -> np.ndarray:
        """Generate anomalies with broken correlations between features."""
        # Start with normal data
        data = self.generate_normal_data(num_samples)
        
        # Break expected correlations (e.g., high bytes but low duration)
        # Shuffle some features independently
        features_to_shuffle = self.rng.choice(
            self.num_features,
            size=self.num_features // 3,
            replace=False
        )
        
        for feat_idx in features_to_shuffle:
            self.rng.shuffle(data[:, feat_idx])
        
        return data
    
    def generate_dataset(
        self,
        num_samples: int,
        anomaly_ratio: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a complete dataset with normal and anomalous samples.
        
        Args:
            num_samples: Total number of samples
            anomaly_ratio: Fraction of samples that are anomalies
            
        Returns:
            Tuple of (data, labels) where labels are 0 for normal, 1 for anomaly
        """
        num_anomalies = int(num_samples * anomaly_ratio)
        num_normal = num_samples - num_anomalies
        
        # Generate normal and anomalous data
        normal_data = self.generate_normal_data(num_normal)
        anomaly_data = self.generate_anomalies(num_anomalies)
        
        # Create labels
        normal_labels = np.zeros(num_normal)
        anomaly_labels = np.ones(num_anomalies)
        
        # Combine and shuffle
        data = np.vstack([normal_data, anomaly_data])
        labels = np.concatenate([normal_labels, anomaly_labels])
        
        # Shuffle together
        indices = self.rng.permutation(num_samples)
        data = data[indices]
        labels = labels[indices]
        
        return data, labels
    
    def generate_inference_batch(self, batch_size: int) -> np.ndarray:
        """
        Generate a batch of normal data for inference testing.
        
        Args:
            batch_size: Number of samples in the batch
            
        Returns:
            Array of shape (batch_size, num_features)
        """
        return self.generate_normal_data(batch_size)
    
    def get_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get expected statistics for each feature.
        
        Returns:
            Dictionary mapping feature names to their statistics
        """
        stats = {}
        for spec in self.feature_specs:
            stats[spec.name] = {
                "min": spec.min_val,
                "max": spec.max_val,
                "expected_mean": (spec.min_val + spec.max_val) / 2,
                "dtype": spec.dtype
            }
        return stats


def create_sample_event() -> Dict[str, float]:
    """
    Create a single sample event for demonstration.
    
    Returns:
        Dictionary representing one automotive event
    """
    generator = DummyDataGenerator()
    data = generator.generate_normal_data(1)[0]
    
    return {
        name: float(value) 
        for name, value in zip(generator.feature_names, data)
    }


def generate_training_validation_test_split(
    num_samples: int = 1000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    anomaly_ratio: float = 0.05,
    random_seed: int = 42
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate a complete dataset split for training, validation, and testing.
    
    Args:
        num_samples: Total number of samples
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        anomaly_ratio: Fraction of anomalies in each split
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys, each containing (data, labels)
    """
    generator = DummyDataGenerator(random_seed=random_seed)
    
    # Calculate split sizes
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    test_size = num_samples - train_size - val_size
    
    # Generate each split
    train_data, train_labels = generator.generate_dataset(train_size, anomaly_ratio)
    val_data, val_labels = generator.generate_dataset(val_size, anomaly_ratio)
    test_data, test_labels = generator.generate_dataset(test_size, anomaly_ratio)
    
    return {
        "train": (train_data, train_labels),
        "val": (val_data, val_labels),
        "test": (test_data, test_labels)
    }
