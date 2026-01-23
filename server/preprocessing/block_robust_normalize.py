"""Robust Normalization Block.

Normalizes data using median and IQR, robust to outliers.
"""

import numpy as np
from typing import Dict, List, Any

from .base import (
    BasePreprocessingBlock,
    BlockResult,
    OutputShape,
    register_block,
)


@register_block
class RobustNormalizeBlock(BasePreprocessingBlock):
    """Robust normalization preprocessing block.
    
    Uses median and interquartile range (IQR) for normalization.
    More robust to outliers than z-score normalization.
    Formula: (x - median) / IQR
    """
    
    block_type = "robust_normalize"
    block_name = "Robust Normalization"
    block_description = "Normalize using median and IQR (robust to outliers)"
    input_shape = OutputShape.ANY
    output_shape = OutputShape.SAME
    category = "normalization"
    version = "1.0.0"
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {
                "name": "q_low",
                "type": "float",
                "default": 25.0,
                "description": "Lower percentile for IQR (default 25)",
            },
            {
                "name": "q_high",
                "type": "float",
                "default": 75.0,
                "description": "Upper percentile for IQR (default 75)",
            },
            {
                "name": "axis",
                "type": "int",
                "default": 0,
                "description": "Axis along which to compute statistics",
            },
        ]
    
    def process(self, data: np.ndarray, **kwargs) -> BlockResult:
        """Apply robust normalization."""
        q_low = self.params.get("q_low", 25.0)
        q_high = self.params.get("q_high", 75.0)
        axis = self.params.get("axis", 0)
        
        try:
            median = np.median(data, axis=axis, keepdims=True)
            q1 = np.percentile(data, q_low, axis=axis, keepdims=True)
            q3 = np.percentile(data, q_high, axis=axis, keepdims=True)
            iqr = q3 - q1 + 1e-8
            
            normalized = (data - median) / iqr
            
            return BlockResult(
                data=normalized.astype(np.float32),
                output_shape=OutputShape.SAME,
                metadata={
                    "median_sample": median.flatten()[:5].tolist(),
                    "iqr_sample": iqr.flatten()[:5].tolist(),
                    "percentiles": [q_low, q_high],
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
