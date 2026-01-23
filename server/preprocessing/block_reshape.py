"""Reshape Block.

Reshapes data to a specified target shape.
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
class ReshapeBlock(BasePreprocessingBlock):
    """Reshape preprocessing block.
    
    Reshapes data to a specified target shape.
    Use -1 for automatic dimension calculation.
    """
    
    block_type = "reshape"
    block_name = "Reshape"
    block_description = "Reshape data to specified dimensions"
    input_shape = OutputShape.ANY
    output_shape = OutputShape.ANY
    category = "reshaping"
    version = "1.0.0"
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {
                "name": "target_shape",
                "type": "list",
                "default": [-1],
                "description": "Target shape (use -1 for auto dimension)",
            },
        ]
    
    def process(self, data: np.ndarray, **kwargs) -> BlockResult:
        """Reshape data to target shape."""
        target_shape = self.params.get("target_shape", [-1])
        
        try:
            original_shape = data.shape
            reshaped = data.reshape(target_shape)
            
            # Determine output shape type
            ndim = reshaped.ndim
            if ndim <= 2:
                out_shape = OutputShape.SHAPE_1D
            elif ndim == 3:
                out_shape = OutputShape.SHAPE_3D
            elif ndim == 4:
                out_shape = OutputShape.SHAPE_4D
            else:
                out_shape = OutputShape.ANY
            
            return BlockResult(
                data=reshaped.astype(np.float32),
                output_shape=out_shape,
                metadata={
                    "original_shape": list(original_shape),
                    "new_shape": list(reshaped.shape),
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
