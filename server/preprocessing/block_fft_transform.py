"""FFT Transform Block.

Applies Fast Fourier Transform for frequency domain analysis.
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
class FFTTransformBlock(BasePreprocessingBlock):
    """FFT transform preprocessing block.
    
    Converts time-domain data to frequency domain using FFT.
    """
    
    block_type = "fft_transform"
    block_name = "FFT Transform"
    block_description = "Convert to frequency domain with Fast Fourier Transform"
    input_shape = OutputShape.ANY
    output_shape = OutputShape.SAME
    category = "transform"
    version = "1.0.0"
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {
                "name": "output_type",
                "type": "str",
                "default": "magnitude",
                "options": ["magnitude", "power", "complex", "phase"],
                "description": "Type of FFT output",
            },
            {
                "name": "normalize",
                "type": "bool",
                "default": True,
                "description": "Normalize FFT output",
            },
            {
                "name": "keep_dc",
                "type": "bool",
                "default": False,
                "description": "Keep DC component (index 0)",
            },
        ]
    
    def process(self, data: np.ndarray, **kwargs) -> BlockResult:
        """Apply FFT transform."""
        output_type = self.params.get("output_type", "magnitude")
        normalize = self.params.get("normalize", True)
        keep_dc = self.params.get("keep_dc", False)
        
        try:
            # Apply FFT along last axis
            fft_result = np.fft.fft(data, axis=-1)
            
            # Keep only positive frequencies
            n = data.shape[-1]
            fft_result = fft_result[..., :n//2]
            
            # Remove DC if requested
            if not keep_dc:
                fft_result = fft_result[..., 1:]
            
            # Convert to requested output type
            if output_type == "magnitude":
                output = np.abs(fft_result)
            elif output_type == "power":
                output = np.abs(fft_result) ** 2
            elif output_type == "phase":
                output = np.angle(fft_result)
            else:  # complex
                output = fft_result
            
            # Normalize if requested
            if normalize and output_type != "phase":
                output = output / (n / 2)
            
            return BlockResult(
                data=output.astype(np.float32) if output_type != "complex" else output,
                output_shape=OutputShape.SAME,
                metadata={
                    "output_type": output_type,
                    "normalized": normalize,
                    "frequency_bins": output.shape[-1],
                    "original_samples": n,
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
