"""Preprocessing Integration for Federated Learning.

This module integrates the preprocessing/ pipeline with FL datasets,
ensuring consistent data transformation across centralized ML/DL and FL training.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

# Import preprocessing from the shared module
from ...preprocessing import (
    PreprocessingPipeline,
    BlockRegistry,
    BasePreprocessingBlock,
    OutputShape,
)

logger = logging.getLogger(__name__)


class FLPreprocessor:
    """Preprocessor for FL datasets using the shared preprocessing pipeline.
    
    This class wraps the preprocessing pipeline to work seamlessly with
    Flower's federated datasets and ensures consistent preprocessing
    across all clients.
    """
    
    def __init__(
        self,
        pipeline_config: Optional[List[Dict[str, Any]]] = None,
        data_type: str = "csi",
    ):
        """Initialize the FL preprocessor.
        
        Args:
            pipeline_config: List of preprocessing block configurations.
                            If None, uses default pipeline for data_type.
            data_type: Type of data (csi, imu, image, audio)
        """
        self.data_type = data_type
        self.pipeline_config = pipeline_config or self._get_default_pipeline(data_type)
        self.pipeline = PreprocessingPipeline(self.pipeline_config)
        self._fitted = False
        self._fit_params: Dict[str, Any] = {}
    
    def _get_default_pipeline(self, data_type: str) -> List[Dict[str, Any]]:
        """Get default preprocessing pipeline for a data type.
        
        Args:
            data_type: Type of data
        
        Returns:
            List of block configurations
        """
        pipelines = {
            "csi": [
                {"type": "subcarrier_filter", "params": {"start": 6, "end": 58}},
                {"type": "lowpass_filter", "params": {"cutoff_freq": 10, "sample_rate": 100}},
                {"type": "zscore_normalize", "params": {}},
                {"type": "sliding_window", "params": {"window_size": 128, "step_size": 64}},
            ],
            "imu": [
                {"type": "zscore_normalize", "params": {}},
                {"type": "sliding_window", "params": {"window_size": 128, "step_size": 64}},
            ],
            "image": [
                {"type": "minmax_normalize", "params": {"feature_range": [0, 1]}},
            ],
            "audio": [
                {"type": "zscore_normalize", "params": {}},
                {"type": "fft_transform", "params": {"n_fft": 512}},
            ],
            "general": [
                {"type": "zscore_normalize", "params": {}},
            ],
        }
        return pipelines.get(data_type, pipelines["general"])
    
    def fit(self, data: np.ndarray) -> "FLPreprocessor":
        """Fit the preprocessor on training data.
        
        This computes any statistics needed for preprocessing (e.g., mean/std for normalization).
        
        Args:
            data: Training data to fit on
        
        Returns:
            self for chaining
        """
        self.pipeline.fit(data)
        self._fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using the fitted pipeline.
        
        Args:
            data: Data to transform
        
        Returns:
            Transformed data
        """
        if not self._fitted:
            logger.warning("Preprocessor not fitted, fitting on input data")
            self.fit(data)
        
        return self.pipeline.transform(data)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.
        
        Args:
            data: Data to fit and transform
        
        Returns:
            Transformed data
        """
        self.fit(data)
        return self.transform(data)
    
    def get_output_shape(self) -> OutputShape:
        """Get the output shape of the pipeline."""
        return self.pipeline.get_output_shape()
    
    def get_config(self) -> Dict[str, Any]:
        """Get the pipeline configuration for serialization."""
        return {
            "data_type": self.data_type,
            "pipeline_config": self.pipeline_config,
            "fitted": self._fitted,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FLPreprocessor":
        """Create preprocessor from configuration."""
        return cls(
            pipeline_config=config.get("pipeline_config"),
            data_type=config.get("data_type", "general"),
        )


def create_fl_transform(
    preprocessor: FLPreprocessor,
    to_tensor: bool = True,
) -> Callable:
    """Create a transform function for Flower datasets.
    
    Args:
        preprocessor: FLPreprocessor instance
        to_tensor: Whether to convert to PyTorch tensor
    
    Returns:
        Transform function compatible with Flower datasets
    """
    def transform_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a batch from Flower dataset."""
        # Get the data key (different datasets use different keys)
        data_key = None
        for key in ["img", "image", "data", "features"]:
            if key in batch:
                data_key = key
                break
        
        if data_key is None:
            return batch
        
        # Transform data
        data = batch[data_key]
        if isinstance(data, list):
            data = np.array(data)
        elif hasattr(data, "numpy"):
            data = data.numpy()
        
        transformed = preprocessor.transform(data)
        
        if to_tensor:
            transformed = torch.tensor(transformed, dtype=torch.float32)
        
        batch["img"] = transformed
        
        # Ensure labels are tensors
        if "label" in batch and to_tensor:
            labels = batch["label"]
            if not isinstance(labels, torch.Tensor):
                batch["label"] = torch.tensor(labels, dtype=torch.long)
        
        return batch
    
    return transform_fn


def get_preprocessing_blocks() -> List[Dict[str, Any]]:
    """Get all available preprocessing blocks.
    
    Returns:
        List of block metadata dictionaries
    """
    return BlockRegistry.list_blocks()


def get_default_pipelines() -> Dict[str, List[Dict[str, Any]]]:
    """Get default preprocessing pipelines for all data types.
    
    Returns:
        Dictionary mapping data_type to pipeline configuration
    """
    preprocessor = FLPreprocessor()
    return {
        "csi": preprocessor._get_default_pipeline("csi"),
        "imu": preprocessor._get_default_pipeline("imu"),
        "image": preprocessor._get_default_pipeline("image"),
        "audio": preprocessor._get_default_pipeline("audio"),
        "general": preprocessor._get_default_pipeline("general"),
    }
