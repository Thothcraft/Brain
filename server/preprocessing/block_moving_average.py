"""Moving Average Block.

Applies moving average smoothing to time-series data.
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
class MovingAverageBlock(BasePreprocessingBlock):
    """Moving average smoothing preprocessing block.
    
    Applies a moving average filter to smooth time-series data.
    """
    
    block_type = "moving_average"
    block_name = "Moving Average"
    block_description = "Smooth data with moving average filter"
    input_shape = OutputShape.ANY
    output_shape = OutputShape.SAME
    category = "filtering"
    version = "1.0.0"
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {
                "name": "window",
                "type": "int",
                "default": 5,
                "description": "Window size for moving average",
            },
            {
                "name": "mode",
                "type": "str",
                "default": "same",
                "options": ["same", "valid", "full"],
                "description": "Convolution mode",
            },
        ]
    
    def process(self, data: np.ndarray, **kwargs) -> BlockResult:
        """Apply moving average smoothing."""
        window = self.params.get("window", 5)
        mode = self.params.get("mode", "same")
        
        if window < 2:
            return BlockResult(
                data=data,
                output_shape=OutputShape.SAME,
                metadata={"skipped": True, "reason": "window < 2"},
                success=True,
            )
        
        try:
            kernel = np.ones(window) / window
            
            # Apply along last axis
            if data.ndim == 1:
                smoothed = np.convolve(data, kernel, mode=mode)
            else:
                smoothed = np.apply_along_axis(
                    lambda m: np.convolve(m, kernel, mode=mode),
                    axis=-1, arr=data
                )
            
            return BlockResult(
                data=smoothed.astype(np.float32),
                output_shape=OutputShape.SAME,
                metadata={
                    "window": window,
                    "mode": mode,
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
