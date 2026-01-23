"""Subcarrier Filter Block.

Filters CSI subcarriers to remove noisy edge subcarriers.
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
class SubcarrierFilterBlock(BasePreprocessingBlock):
    """Subcarrier filter preprocessing block.
    
    Filters CSI data to keep only specified subcarrier range.
    Removes noisy edge subcarriers common in WiFi CSI.
    """
    
    block_type = "subcarrier_filter"
    block_name = "Subcarrier Filter"
    block_description = "Filter CSI subcarriers (remove noisy edges)"
    input_shape = OutputShape.ANY
    output_shape = OutputShape.SAME
    category = "csi"
    version = "1.0.0"
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {
                "name": "start_idx",
                "type": "int",
                "default": 5,
                "description": "Start subcarrier index (skip first N)",
            },
            {
                "name": "end_idx",
                "type": "int",
                "default": 59,
                "description": "End subcarrier index (keep up to N)",
            },
            {
                "name": "subcarrier_axis",
                "type": "int",
                "default": -1,
                "description": "Axis containing subcarriers",
            },
        ]
    
    def process(self, data: np.ndarray, **kwargs) -> BlockResult:
        """Filter subcarriers."""
        start_idx = self.params.get("start_idx", 5)
        end_idx = self.params.get("end_idx", 59)
        axis = self.params.get("subcarrier_axis", -1)
        
        try:
            original_shape = data.shape
            
            # Create slice for the specified axis
            slices = [slice(None)] * data.ndim
            slices[axis] = slice(start_idx, end_idx)
            
            filtered = data[tuple(slices)]
            
            return BlockResult(
                data=filtered.astype(np.float32),
                output_shape=OutputShape.SAME,
                metadata={
                    "original_subcarriers": original_shape[axis],
                    "filtered_subcarriers": filtered.shape[axis],
                    "range": [start_idx, end_idx],
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
