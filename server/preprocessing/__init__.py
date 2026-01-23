"""Preprocessing Blocks Module.

This module provides modular preprocessing blocks for data transformation.
Each block is in a separate file with standardized naming conventions.

Naming Convention:
- File: block_{block_name}.py (e.g., block_zscore_normalize.py)
- Class: {BlockName}Block (e.g., ZScoreNormalizeBlock)

Output Shapes:
- 1D: (batch, features) - Flattened features
- 3D: (batch, seq_len, features) - Sequential/time-series
- 4D: (batch, frames, height, width) or (batch, channels, height, width) - Video/Image

Usage:
    from server.preprocessing import PreprocessingRegistry, execute_pipeline
    
    # List available blocks
    blocks = PreprocessingRegistry.list_blocks()
    
    # Execute pipeline
    result = execute_pipeline(data, [
        {"type": "zscore_normalize", "params": {}},
        {"type": "sliding_window", "params": {"window_size": 128}},
    ])
"""

from .base import (
    BasePreprocessingBlock,
    PreprocessingRegistry,
    OutputShape,
    BlockMetadata,
)
from .pipeline import PreprocessingPipeline, execute_pipeline

# Import all blocks to register them
from . import block_zscore_normalize
from . import block_minmax_normalize
from . import block_robust_normalize
from . import block_moving_average
from . import block_sliding_window
from . import block_flatten
from . import block_reshape
from . import block_noise_injection
from . import block_subcarrier_filter
from . import block_lowpass_filter
from . import block_pca_reduction
from . import block_fft_transform

__all__ = [
    "BasePreprocessingBlock",
    "PreprocessingRegistry",
    "PreprocessingPipeline",
    "OutputShape",
    "BlockMetadata",
    "execute_pipeline",
]
