"""Min-Max Normalization Block.

Scales data to a specified range (default [0, 1]).
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
class MinMaxNormalizeBlock(BasePreprocessingBlock):
    """Min-Max normalization preprocessing block.
    
    Scales data to [min_val, max_val] range.
    Formula: (x - min) / (max - min) * (max_val - min_val) + min_val
    """
    
    block_type = "minmax_normalize"
    block_name = "Min-Max Normalization"
    block_description = "Scale data to [0, 1] or custom range"
    input_shape = OutputShape.ANY
    output_shape = OutputShape.SAME
    category = "normalization"
    version = "1.0.0"
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {
                "name": "min_val",
                "type": "float",
                "default": 0.0,
                "description": "Minimum value of output range",
            },
            {
                "name": "max_val",
                "type": "float",
                "default": 1.0,
                "description": "Maximum value of output range",
            },
            {
                "name": "axis",
                "type": "int",
                "default": 0,
                "description": "Axis along which to compute min/max",
            },
        ]
    
    def process(self, data: np.ndarray, **kwargs) -> BlockResult:
        """Apply min-max normalization."""
        min_val = self.params.get("min_val", 0.0)
        max_val = self.params.get("max_val", 1.0)
        axis = self.params.get("axis", 0)
        
        try:
            data_min = data.min(axis=axis, keepdims=True)
            data_max = data.max(axis=axis, keepdims=True)
            data_range = data_max - data_min + 1e-8
            
            normalized = (data - data_min) / data_range
            normalized = normalized * (max_val - min_val) + min_val
            
            return BlockResult(
                data=normalized.astype(np.float32),
                output_shape=OutputShape.SAME,
                metadata={
                    "data_min_sample": data_min.flatten()[:5].tolist(),
                    "data_max_sample": data_max.flatten()[:5].tolist(),
                    "output_range": [min_val, max_val],
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
