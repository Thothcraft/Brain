"""Z-Score Normalization Block.

Normalizes data to zero mean and unit variance: (x - mean) / std
"""

import numpy as np
from typing import Dict, List, Any, Optional

from .base import (
    BasePreprocessingBlock,
    BlockResult,
    OutputShape,
    register_block,
)


@register_block
class ZScoreNormalizeBlock(BasePreprocessingBlock):
    """Z-Score normalization preprocessing block.
    
    Normalizes data to zero mean and unit variance along specified axis.
    Formula: (x - mean) / std
    """
    
    block_type = "zscore_normalize"
    block_name = "Z-Score Normalization"
    block_description = "Normalize data to zero mean and unit variance"
    input_shape = OutputShape.ANY
    output_shape = OutputShape.SAME
    category = "normalization"
    version = "1.0.0"
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {
                "name": "axis",
                "type": "int",
                "default": 0,
                "description": "Axis along which to compute mean and std (0=samples, -1=features)",
            },
            {
                "name": "epsilon",
                "type": "float",
                "default": 1e-8,
                "description": "Small value to prevent division by zero",
            },
        ]
    
    def process(self, data: np.ndarray, **kwargs) -> BlockResult:
        """Apply z-score normalization."""
        axis = self.params.get("axis", 0)
        epsilon = self.params.get("epsilon", 1e-8)
        
        try:
            mean = data.mean(axis=axis, keepdims=True)
            std = data.std(axis=axis, keepdims=True) + epsilon
            normalized = (data - mean) / std
            
            return BlockResult(
                data=normalized.astype(np.float32),
                output_shape=OutputShape.SAME,
                metadata={
                    "mean_sample": mean.flatten()[:5].tolist(),
                    "std_sample": std.flatten()[:5].tolist(),
                    "axis": axis,
                },
                success=True,
            )
        except Exception as e:
            return BlockResult(
                data=data,
                output_shape=OutputShape.SAME,
                success=False,
                error=str(e),
            )
