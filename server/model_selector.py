"""Model Selector - Automatic model selection based on data type and preprocessing output.

Selects appropriate models (nano/mini/max) based on:
- Data type (CSI, Image, Video, etc.)
- Preprocessing output shape (1D, 2D, 3D, 4D)
- Available compute resources

Shape Conventions:
- 1D: (batch, features) - Flattened feature vectors -> MLP
- 2D: (batch, seq_len, features) - Sequential/time-series -> LSTM, GRU, CNN1D, Transformer
- 3D: (batch, channels, height, width) - Single images -> CNN2D, ResNet
- 4D: (batch, num_frames, channels, height, width) - Video -> CNN3D
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class DataType(str, Enum):
    """Supported data types."""
    CSI = "csi"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    IMU = "imu"
    GENERAL_CSV = "general_csv"
    UNKNOWN = "unknown"


class OutputShape(str, Enum):
    """Preprocessing output shapes."""
    SHAPE_1D = "1d"  # (batch, features) - Flattened
    SHAPE_2D = "2d"  # (batch, seq_len, features) - Sequential
    SHAPE_3D = "3d"  # (batch, channels, height, width) - Image
    SHAPE_4D = "4d"  # (batch, num_frames, channels, height, width) - Video


@dataclass
class ModelRecommendation:
    """Model recommendation with metadata."""
    model_type: str
    model_name: str
    size: str  # nano, mini, max
    input_shape: str
    description: str
    estimated_params: int
    recommended_for: List[str]
    training_time_factor: float = 1.0  # Relative to nano


class ModelSelector:
    """Selects appropriate models based on data and preprocessing configuration.
    
    The selector considers:
    1. Data type (determines base model architecture)
    2. Preprocessing output shape (filters compatible models)
    3. Model size preference (nano/mini/max)
    4. Available compute resources
    """
    
    # Model definitions by output shape
    MODELS_1D = {
        "nano": [
            ModelRecommendation("mlp_nano", "MLP Nano", "nano", "1d", "Minimal MLP for edge deployment", 5000, ["csi", "imu", "general_csv"], 1.0),
        ],
        "mini": [
            ModelRecommendation("mlp_mini", "MLP Mini", "mini", "1d", "Balanced MLP with batch norm", 50000, ["csi", "imu", "general_csv"], 2.0),
        ],
        "max": [
            ModelRecommendation("mlp_max", "MLP Max", "max", "1d", "Maximum capacity MLP", 500000, ["csi", "imu", "general_csv"], 5.0),
        ],
    }
    
    # Sequential models for 2D data (batch, seq_len, features)
    MODELS_2D = {
        "nano": [
            ModelRecommendation("lstm_nano", "LSTM Nano", "nano", "2d", "Minimal LSTM for sequences", 10000, ["csi", "imu", "audio"], 1.0),
            ModelRecommendation("gru_nano", "GRU Nano", "nano", "2d", "Minimal GRU (faster than LSTM)", 8000, ["csi", "imu"], 0.9),
            ModelRecommendation("cnn1d_nano", "CNN1D Nano", "nano", "2d", "Minimal 1D CNN for time-series", 15000, ["csi", "imu", "audio"], 0.8),
            ModelRecommendation("transformer_nano", "Transformer Nano", "nano", "2d", "Minimal Transformer encoder", 20000, ["csi", "imu"], 1.2),
        ],
        "mini": [
            ModelRecommendation("lstm_mini", "LSTM Mini", "mini", "2d", "Balanced BiLSTM", 100000, ["csi", "imu", "audio"], 2.5),
            ModelRecommendation("gru_mini", "GRU Mini", "mini", "2d", "Balanced BiGRU", 80000, ["csi", "imu"], 2.0),
            ModelRecommendation("cnn1d_mini", "CNN1D Mini", "mini", "2d", "Balanced 1D CNN with residuals", 150000, ["csi", "imu", "audio"], 1.8),
            ModelRecommendation("transformer_mini", "Transformer Mini", "mini", "2d", "Balanced Transformer (3 layers)", 200000, ["csi", "imu"], 3.0),
        ],
        "max": [
            ModelRecommendation("lstm_max", "LSTM Max", "max", "2d", "Maximum BiLSTM with attention", 1000000, ["csi", "imu", "audio"], 8.0),
            ModelRecommendation("gru_max", "GRU Max", "max", "2d", "Maximum BiGRU", 800000, ["csi", "imu"], 6.0),
            ModelRecommendation("cnn1d_max", "CNN1D Max", "max", "2d", "Maximum 1D CNN with deep residuals", 1000000, ["csi", "imu", "audio"], 5.0),
            ModelRecommendation("transformer_max", "Transformer Max", "max", "2d", "Maximum Transformer (6 layers)", 2000000, ["csi", "imu"], 10.0),
        ],
    }
    
    # Image models for 3D data (batch, channels, height, width)
    MODELS_3D = {
        "nano": [
            ModelRecommendation("cnn2d_nano", "CNN2D Nano", "nano", "3d", "Minimal 2D CNN for images", 30000, ["image"], 1.0),
            ModelRecommendation("resnet_nano", "ResNet Nano", "nano", "3d", "Minimal ResNet for images", 100000, ["image"], 1.5),
        ],
        "mini": [
            ModelRecommendation("cnn2d_mini", "CNN2D Mini", "mini", "3d", "Balanced 2D CNN", 500000, ["image"], 3.0),
            ModelRecommendation("resnet_mini", "ResNet Mini", "mini", "3d", "ResNet-18 style", 1000000, ["image"], 4.0),
        ],
        "max": [
            ModelRecommendation("cnn2d_max", "CNN2D Max", "max", "3d", "Maximum 2D CNN with residuals", 5000000, ["image"], 8.0),
            ModelRecommendation("resnet_max", "ResNet Max", "max", "3d", "ResNet-50 style", 25000000, ["image"], 12.0),
        ],
    }
    
    # Video models for 4D data (batch, num_frames, channels, height, width)
    MODELS_4D = {
        "nano": [
            ModelRecommendation("cnn3d_nano", "CNN3D Nano", "nano", "4d", "Minimal 3D CNN for video", 50000, ["video"], 2.0),
        ],
        "mini": [
            ModelRecommendation("cnn3d_mini", "CNN3D Mini", "mini", "4d", "Balanced 3D CNN for video", 500000, ["video"], 5.0),
        ],
        "max": [
            ModelRecommendation("cnn3d_max", "CNN3D Max", "max", "4d", "Maximum 3D CNN for video", 5000000, ["video"], 15.0),
        ],
    }
    
    # ML models (for 1D data only)
    ML_MODELS = [
        {"type": "svm", "name": "Support Vector Machine", "description": "SVM with RBF kernel"},
        {"type": "random_forest", "name": "Random Forest", "description": "Ensemble of decision trees"},
        {"type": "knn", "name": "K-Nearest Neighbors", "description": "Instance-based learning"},
        {"type": "logistic_regression", "name": "Logistic Regression", "description": "Linear classifier"},
        {"type": "gradient_boosting", "name": "Gradient Boosting", "description": "Sequential ensemble"},
        {"type": "decision_tree", "name": "Decision Tree", "description": "Interpretable tree classifier"},
        {"type": "kmeans", "name": "K-Means Clustering", "description": "Unsupervised clustering"},
        {"type": "dbscan", "name": "DBSCAN", "description": "Density-based clustering"},
        {"type": "pca_visualizer", "name": "PCA Visualizer", "description": "Dimensionality reduction & visualization"},
    ]
    
    @classmethod
    def get_compatible_models(
        cls,
        output_shape: str,
        data_type: str = None,
        size_filter: str = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get models compatible with the given output shape.
        
        Args:
            output_shape: Preprocessing output shape (1d, 3d, 4d)
            data_type: Optional data type to filter recommendations
            size_filter: Optional size filter (nano, mini, max)
            
        Returns:
            Dictionary with 'dl_models' and 'ml_models' lists
        """
        result = {"dl_models": [], "ml_models": []}
        
        # Select model pool based on output shape
        if output_shape == "1d":
            model_pool = cls.MODELS_1D
            result["ml_models"] = cls.ML_MODELS
        elif output_shape == "2d":
            model_pool = cls.MODELS_2D
        elif output_shape == "3d":
            model_pool = cls.MODELS_3D
        elif output_shape == "4d":
            model_pool = cls.MODELS_4D
        else:
            return result
        
        # Collect DL models
        sizes = [size_filter] if size_filter else ["nano", "mini", "max"]
        
        for size in sizes:
            if size in model_pool:
                for model in model_pool[size]:
                    # Filter by data type if specified
                    if data_type and data_type not in model.recommended_for:
                        continue
                    
                    result["dl_models"].append({
                        "type": model.model_type,
                        "name": model.model_name,
                        "size": model.size,
                        "input_shape": model.input_shape,
                        "description": model.description,
                        "estimated_params": model.estimated_params,
                        "recommended_for": model.recommended_for,
                        "training_time_factor": model.training_time_factor,
                    })
        
        return result
    
    @classmethod
    def get_recommended_models(
        cls,
        data_type: str,
        output_shape: str,
        prefer_speed: bool = False,
        prefer_accuracy: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get recommended models for a specific use case.
        
        Args:
            data_type: Type of data (csi, image, video, etc.)
            output_shape: Preprocessing output shape
            prefer_speed: Prefer faster models (nano)
            prefer_accuracy: Prefer more accurate models (max)
            
        Returns:
            List of recommended models sorted by relevance
        """
        compatible = cls.get_compatible_models(output_shape, data_type)
        
        # Combine DL and ML models
        all_models = compatible["dl_models"] + [
            {**m, "size": "ml", "input_shape": "1d", "estimated_params": 0, "training_time_factor": 0.5}
            for m in compatible["ml_models"]
        ]
        
        # Score and sort models
        def score_model(model):
            score = 0
            
            # Size preference
            if prefer_speed:
                if model["size"] == "nano":
                    score += 10
                elif model["size"] == "mini":
                    score += 5
                elif model["size"] == "ml":
                    score += 8
            elif prefer_accuracy:
                if model["size"] == "max":
                    score += 10
                elif model["size"] == "mini":
                    score += 5
            else:
                # Default: prefer mini
                if model["size"] == "mini":
                    score += 10
                elif model["size"] == "nano":
                    score += 5
                elif model["size"] == "max":
                    score += 5
            
            # Data type match
            if "recommended_for" in model and data_type in model.get("recommended_for", []):
                score += 5
            
            return score
        
        all_models.sort(key=score_model, reverse=True)
        return all_models
    
    @classmethod
    def get_default_preprocessing(cls, data_type: str) -> List[Dict[str, Any]]:
        """Get default preprocessing pipeline for a data type.
        
        Args:
            data_type: Type of data
            
        Returns:
            List of preprocessing block configurations
        """
        pipelines = {
            "csi": [
                {"type": "subcarrier_filter", "enabled": True, "params": {"start_idx": 5, "end_idx": 59}},
                {"type": "zscore_normalize", "enabled": True, "params": {}},
                {"type": "sliding_window", "enabled": True, "params": {"window_size": 128, "stride": 64}},
            ],
            "imu": [
                {"type": "zscore_normalize", "enabled": True, "params": {}},
                {"type": "lowpass_filter", "enabled": True, "params": {"cutoff_ratio": 0.1}},
                {"type": "sliding_window", "enabled": True, "params": {"window_size": 100, "stride": 50}},
            ],
            "image": [
                {"type": "minmax_normalize", "enabled": True, "params": {"min_val": 0, "max_val": 1}},
            ],
            "video": [
                {"type": "minmax_normalize", "enabled": True, "params": {"min_val": 0, "max_val": 1}},
            ],
            "audio": [
                {"type": "zscore_normalize", "enabled": True, "params": {}},
                {"type": "fft_transform", "enabled": True, "params": {"output_type": "magnitude"}},
                {"type": "sliding_window", "enabled": True, "params": {"window_size": 256, "stride": 128}},
            ],
            "general_csv": [
                {"type": "zscore_normalize", "enabled": True, "params": {}},
            ],
        }
        
        return pipelines.get(data_type, [{"type": "zscore_normalize", "enabled": True, "params": {}}])
    
    @classmethod
    def get_output_shape_for_pipeline(cls, data_type: str, pipeline: List[Dict[str, Any]]) -> str:
        """Determine output shape based on data type and pipeline.
        
        Args:
            data_type: Type of data
            pipeline: Preprocessing pipeline configuration
            
        Returns:
            Output shape (1d, 3d, 4d)
        """
        # Check if pipeline produces specific shape
        for block in reversed(pipeline):
            if not block.get("enabled", True):
                continue
            
            block_type = block.get("type", "")
            
            if block_type == "flatten":
                return "1d"
            elif block_type == "sliding_window":
                return "2d"  # Sequential data
            elif block_type == "reshape":
                target = block.get("params", {}).get("target_shape", [])
                if len(target) == 2:
                    return "1d"
                elif len(target) == 3:
                    return "2d"  # Sequential
                elif len(target) == 4:
                    return "3d"  # Image
                elif len(target) == 5:
                    return "4d"  # Video
        
        # Default based on data type
        defaults = {
            "csi": "2d",    # Sequential
            "imu": "2d",    # Sequential
            "image": "3d",  # Image
            "video": "4d",  # Video
            "audio": "2d",  # Sequential
            "general_csv": "1d",
        }
        
        return defaults.get(data_type, "1d")
    
    @classmethod
    def get_training_config_options(
        cls,
        data_type: str,
        output_shape: str,
    ) -> Dict[str, Any]:
        """Get training configuration options for the modal.
        
        Args:
            data_type: Type of data
            output_shape: Preprocessing output shape
            
        Returns:
            Configuration options for the training modal
        """
        compatible = cls.get_compatible_models(output_shape, data_type)
        
        return {
            "data_type": data_type,
            "output_shape": output_shape,
            "dl_models": {
                "nano": [m for m in compatible["dl_models"] if m["size"] == "nano"],
                "mini": [m for m in compatible["dl_models"] if m["size"] == "mini"],
                "max": [m for m in compatible["dl_models"] if m["size"] == "max"],
            },
            "ml_models": compatible["ml_models"],
            "default_preprocessing": cls.get_default_preprocessing(data_type),
            "recommended": cls.get_recommended_models(data_type, output_shape)[:5],
        }
