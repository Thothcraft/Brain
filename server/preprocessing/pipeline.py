"""Preprocessing Pipeline Executor.

This module provides the pipeline execution logic for chaining preprocessing blocks.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from .base import (
    BasePreprocessingBlock,
    PreprocessingRegistry,
    BlockResult,
    OutputShape,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineStageResult:
    """Result from a single pipeline stage."""
    block_type: str
    block_name: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    output_shape_type: OutputShape
    execution_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Result from executing a complete pipeline."""
    data: np.ndarray
    final_shape: Tuple[int, ...]
    final_shape_type: OutputShape
    stages: List[PipelineStageResult] = field(default_factory=list)
    total_time_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "final_shape": list(self.final_shape),
            "final_shape_type": self.final_shape_type.value,
            "stages": [
                {
                    "block_type": s.block_type,
                    "block_name": s.block_name,
                    "input_shape": list(s.input_shape),
                    "output_shape": list(s.output_shape),
                    "output_shape_type": s.output_shape_type.value,
                    "execution_time_ms": s.execution_time_ms,
                    "metadata": s.metadata,
                    "success": s.success,
                    "error": s.error,
                }
                for s in self.stages
            ],
            "total_time_ms": self.total_time_ms,
            "success": self.success,
            "error": self.error,
        }


class PreprocessingPipeline:
    """Execute a sequence of preprocessing blocks.
    
    The pipeline tracks:
    - Input/output shapes at each stage
    - Execution time per block
    - Metadata from each block
    - Final output shape type (1D, 3D, 4D)
    """
    
    def __init__(self, blocks: List[Dict[str, Any]]):
        """Initialize pipeline with block configurations.
        
        Args:
            blocks: List of block configs, each with:
                - type: Block type identifier
                - enabled: Whether block is enabled (default True)
                - params: Block-specific parameters
        """
        self.block_configs = blocks
        self.stages: List[PipelineStageResult] = []
    
    def execute(self, data: np.ndarray) -> PipelineResult:
        """Execute the pipeline on input data.
        
        Args:
            data: Input numpy array
            
        Returns:
            PipelineResult with processed data and stage info
        """
        self.stages = []
        current_data = data
        start_time = time.time()
        
        for config in self.block_configs:
            if not config.get("enabled", True):
                continue
            
            block_type = config.get("type", "")
            params = config.get("params", {})
            
            # Create block instance
            block = PreprocessingRegistry.create(block_type, params)
            if block is None:
                logger.warning(f"Unknown preprocessing block: {block_type}")
                continue
            
            # Execute block
            stage_start = time.time()
            input_shape = current_data.shape
            
            try:
                result = block.process(current_data)
                
                if not result.success:
                    return PipelineResult(
                        data=current_data,
                        final_shape=current_data.shape,
                        final_shape_type=self._detect_shape_type(current_data),
                        stages=self.stages,
                        total_time_ms=(time.time() - start_time) * 1000,
                        success=False,
                        error=f"Block {block_type} failed: {result.error}",
                    )
                
                current_data = result.data
                stage_time = (time.time() - stage_start) * 1000
                
                self.stages.append(PipelineStageResult(
                    block_type=block_type,
                    block_name=block.block_name,
                    input_shape=input_shape,
                    output_shape=current_data.shape,
                    output_shape_type=result.output_shape,
                    execution_time_ms=stage_time,
                    metadata=result.metadata,
                    success=True,
                ))
                
            except Exception as e:
                logger.error(f"Error in block {block_type}: {e}")
                self.stages.append(PipelineStageResult(
                    block_type=block_type,
                    block_name=block.block_name,
                    input_shape=input_shape,
                    output_shape=input_shape,
                    output_shape_type=OutputShape.SAME,
                    execution_time_ms=(time.time() - stage_start) * 1000,
                    success=False,
                    error=str(e),
                ))
                
                return PipelineResult(
                    data=current_data,
                    final_shape=current_data.shape,
                    final_shape_type=self._detect_shape_type(current_data),
                    stages=self.stages,
                    total_time_ms=(time.time() - start_time) * 1000,
                    success=False,
                    error=f"Pipeline failed at block {block_type}: {e}",
                )
        
        total_time = (time.time() - start_time) * 1000
        final_shape_type = self._detect_shape_type(current_data)
        
        return PipelineResult(
            data=current_data,
            final_shape=current_data.shape,
            final_shape_type=final_shape_type,
            stages=self.stages,
            total_time_ms=total_time,
            success=True,
        )
    
    def _detect_shape_type(self, data: np.ndarray) -> OutputShape:
        """Detect the output shape type from data dimensions."""
        ndim = data.ndim
        
        if ndim <= 2:
            return OutputShape.SHAPE_1D
        elif ndim == 3:
            return OutputShape.SHAPE_3D
        elif ndim == 4:
            return OutputShape.SHAPE_4D
        else:
            return OutputShape.ANY
    
    def get_expected_output_shape(self) -> OutputShape:
        """Get the expected output shape type based on block sequence."""
        if not self.block_configs:
            return OutputShape.ANY
        
        # Find the last block that changes shape
        for config in reversed(self.block_configs):
            if not config.get("enabled", True):
                continue
            
            block_type = config.get("type", "")
            block_class = PreprocessingRegistry.get(block_type)
            
            if block_class and block_class.output_shape != OutputShape.SAME:
                return block_class.output_shape
        
        return OutputShape.SAME


def execute_pipeline(
    data: np.ndarray,
    blocks: List[Dict[str, Any]],
) -> PipelineResult:
    """Convenience function to execute a preprocessing pipeline.
    
    Args:
        data: Input numpy array
        blocks: List of block configurations
        
    Returns:
        PipelineResult with processed data
    """
    pipeline = PreprocessingPipeline(blocks)
    return pipeline.execute(data)


def get_compatible_models(output_shape: OutputShape) -> Dict[str, List[str]]:
    """Get compatible model architectures for an output shape.
    
    Args:
        output_shape: The preprocessing output shape
        
    Returns:
        Dictionary with model categories and compatible models
    """
    models = {
        OutputShape.SHAPE_1D: {
            "dl_models": ["mlp_nano", "mlp_mini", "mlp_max"],
            "ml_models": ["svm", "random_forest", "knn", "logistic_regression", "gradient_boosting"],
        },
        OutputShape.SHAPE_3D: {
            "dl_models": ["lstm_nano", "lstm_mini", "lstm_max", "gru_nano", "gru_mini", "gru_max", 
                        "cnn1d_nano", "cnn1d_mini", "cnn1d_max", "transformer_nano", "transformer_mini", "transformer_max"],
            "ml_models": [],  # Sequential data typically needs DL
        },
        OutputShape.SHAPE_4D: {
            "dl_models": ["cnn2d_nano", "cnn2d_mini", "cnn2d_max", "cnn3d_nano", "cnn3d_mini", "cnn3d_max",
                        "resnet_nano", "resnet_mini", "resnet_max"],
            "ml_models": [],  # Image/video data needs DL
        },
    }
    
    return models.get(output_shape, {"dl_models": [], "ml_models": []})
