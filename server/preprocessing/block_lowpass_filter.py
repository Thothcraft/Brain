"""Lowpass Filter Block.

Applies lowpass filtering to remove high-frequency noise.
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
class LowpassFilterBlock(BasePreprocessingBlock):
    """Lowpass filter preprocessing block.
    
    Applies a simple lowpass filter using convolution.
    For more advanced filtering, use scipy if available.
    """
    
    block_type = "lowpass_filter"
    block_name = "Lowpass Filter"
    block_description = "Remove high-frequency noise with lowpass filter"
    input_shape = OutputShape.ANY
    output_shape = OutputShape.SAME
    category = "filtering"
    version = "1.0.0"
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {
                "name": "cutoff_ratio",
                "type": "float",
                "default": 0.1,
                "description": "Cutoff frequency as ratio of Nyquist (0-1)",
            },
            {
                "name": "order",
                "type": "int",
                "default": 5,
                "description": "Filter order (higher = sharper cutoff)",
            },
        ]
    
    def process(self, data: np.ndarray, **kwargs) -> BlockResult:
        """Apply lowpass filter."""
        cutoff_ratio = self.params.get("cutoff_ratio", 0.1)
        order = self.params.get("order", 5)
        
        try:
            # Try to use scipy for better filtering
            try:
                from scipy.signal import butter, filtfilt
                
                b, a = butter(order, cutoff_ratio, btype='low')
                
                if data.ndim == 1:
                    filtered = filtfilt(b, a, data)
                else:
                    filtered = np.apply_along_axis(
                        lambda x: filtfilt(b, a, x), axis=-1, arr=data
                    )
                
                method = "butterworth"
                
            except ImportError:
                # Fallback to simple moving average
                window = max(3, int(1 / cutoff_ratio))
                kernel = np.ones(window) / window
                
                if data.ndim == 1:
                    filtered = np.convolve(data, kernel, mode='same')
                else:
                    filtered = np.apply_along_axis(
                        lambda x: np.convolve(x, kernel, mode='same'),
                        axis=-1, arr=data
                    )
                
                method = "moving_average_fallback"
            
            return BlockResult(
                data=filtered.astype(np.float32),
                output_shape=OutputShape.SAME,
                metadata={
                    "cutoff_ratio": cutoff_ratio,
                    "order": order,
                    "method": method,
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
