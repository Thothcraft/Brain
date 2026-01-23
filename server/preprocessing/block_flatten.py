"""Flatten Block.

Flattens multi-dimensional data to 1D/2D format.
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
class FlattenBlock(BasePreprocessingBlock):
    """Flatten preprocessing block.
    
    Flattens multi-dimensional data to (batch, features) format.
    """
    
    block_type = "flatten"
    block_name = "Flatten"
    block_description = "Flatten multi-dimensional data to 1D features"
    input_shape = OutputShape.ANY
    output_shape = OutputShape.SHAPE_1D
    category = "reshaping"
    version = "1.0.0"
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {
                "name": "start_dim",
                "type": "int",
                "default": 1,
                "description": "First dimension to flatten (0=include batch)",
            },
        ]
    
    def process(self, data: np.ndarray, **kwargs) -> BlockResult:
        """Flatten data."""
        start_dim = self.params.get("start_dim", 1)
        
        try:
            original_shape = data.shape
            
            if data.ndim <= 2:
                return BlockResult(
                    data=data.astype(np.float32),
                    output_shape=OutputShape.SHAPE_1D,
                    metadata={
                        "already_flat": True,
                        "shape": list(data.shape),
                    },
                    success=True,
                )
            
            # Flatten from start_dim onwards
            if start_dim == 0:
                flattened = data.reshape(-1)
            else:
                new_shape = data.shape[:start_dim] + (-1,)
                flattened = data.reshape(new_shape)
            
            return BlockResult(
                data=flattened.astype(np.float32),
                output_shape=OutputShape.SHAPE_1D,
                metadata={
                    "original_shape": list(original_shape),
                    "new_shape": list(flattened.shape),
                    "start_dim": start_dim,
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
