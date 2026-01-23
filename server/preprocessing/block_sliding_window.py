"""Sliding Window Block.

Creates overlapping windows from time-series data.
Converts 2D data to 3D sequential format.
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
class SlidingWindowBlock(BasePreprocessingBlock):
    """Sliding window preprocessing block.
    
    Creates overlapping windows from time-series data.
    Converts (samples, features) to (windows, window_size, features).
    """
    
    block_type = "sliding_window"
    block_name = "Sliding Window"
    block_description = "Create overlapping windows from time-series (2D -> 3D)"
    input_shape = OutputShape.SHAPE_2D
    output_shape = OutputShape.SHAPE_3D
    category = "windowing"
    version = "1.0.0"
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {
                "name": "window_size",
                "type": "int",
                "default": 128,
                "description": "Size of each window",
            },
            {
                "name": "stride",
                "type": "int",
                "default": 64,
                "description": "Step size between windows (default: window_size/2)",
            },
            {
                "name": "drop_last",
                "type": "bool",
                "default": True,
                "description": "Drop last incomplete window",
            },
        ]
    
    def process(self, data: np.ndarray, **kwargs) -> BlockResult:
        """Create sliding windows from data."""
        window_size = self.params.get("window_size", 128)
        stride = self.params.get("stride", window_size // 2)
        drop_last = self.params.get("drop_last", True)
        
        try:
            # Handle different input dimensions
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            n_samples = data.shape[0]
            
            if n_samples < window_size:
                return BlockResult(
                    data=data,
                    output_shape=OutputShape.SHAPE_2D,
                    success=False,
                    error=f"Data length {n_samples} < window_size {window_size}",
                )
            
            # Calculate number of windows
            if drop_last:
                n_windows = (n_samples - window_size) // stride + 1
            else:
                n_windows = (n_samples - 1) // stride + 1
            
            # Create windows
            windows = []
            for i in range(n_windows):
                start = i * stride
                end = start + window_size
                
                if end <= n_samples:
                    windows.append(data[start:end])
                elif not drop_last:
                    # Pad last window
                    window = np.zeros((window_size, data.shape[1]), dtype=data.dtype)
                    window[:n_samples - start] = data[start:]
                    windows.append(window)
            
            if not windows:
                return BlockResult(
                    data=data,
                    output_shape=OutputShape.SHAPE_2D,
                    success=False,
                    error="No windows created",
                )
            
            result = np.stack(windows, axis=0)
            
            return BlockResult(
                data=result.astype(np.float32),
                output_shape=OutputShape.SHAPE_3D,
                metadata={
                    "num_windows": len(windows),
                    "window_size": window_size,
                    "stride": stride,
                    "input_samples": n_samples,
                    "output_shape": list(result.shape),
                },
                success=True,
            )
        except Exception as e:
            return BlockResult(
                data=data,
                output_shape=OutputShape.SHAPE_2D,
                success=False,
                error=str(e),
            )
