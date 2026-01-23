"""Noise Injection Block.

Adds Gaussian noise for data augmentation.
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
class NoiseInjectionBlock(BasePreprocessingBlock):
    """Noise injection preprocessing block.
    
    Adds Gaussian noise to data for augmentation.
    Noise level is relative to data standard deviation.
    """
    
    block_type = "noise_injection"
    block_name = "Noise Injection"
    block_description = "Add Gaussian noise for data augmentation"
    input_shape = OutputShape.ANY
    output_shape = OutputShape.SAME
    category = "augmentation"
    version = "1.0.0"
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {
                "name": "noise_level",
                "type": "float",
                "default": 0.01,
                "description": "Noise level relative to data std (0.01 = 1%)",
            },
            {
                "name": "seed",
                "type": "int",
                "default": None,
                "description": "Random seed for reproducibility",
            },
        ]
    
    def process(self, data: np.ndarray, **kwargs) -> BlockResult:
        """Add Gaussian noise to data."""
        noise_level = self.params.get("noise_level", 0.01)
        seed = self.params.get("seed")
        
        try:
            if seed is not None:
                np.random.seed(seed)
            
            data_std = data.std()
            noise = np.random.randn(*data.shape) * noise_level * data_std
            augmented = data + noise
            
            return BlockResult(
                data=augmented.astype(np.float32),
                output_shape=OutputShape.SAME,
                metadata={
                    "noise_level": noise_level,
                    "data_std": float(data_std),
                    "noise_std": float(noise_level * data_std),
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
