"""Base classes and registry for preprocessing blocks.

This module defines the base class and registry pattern for all preprocessing blocks.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Type
import numpy as np

logger = logging.getLogger(__name__)


class OutputShape(str, Enum):
    """Output shape types for preprocessing blocks."""
    SHAPE_1D = "1d"  # (batch, features) - Flattened
    SHAPE_2D = "2d"  # (batch, features) - Same as 1D but explicit
    SHAPE_3D = "3d"  # (batch, seq_len, features) - Sequential
    SHAPE_4D = "4d"  # (batch, frames/channels, height, width) - Video/Image
    SAME = "same"    # Same as input
    ANY = "any"      # Any shape


@dataclass
class BlockMetadata:
    """Metadata for a preprocessing block."""
    name: str
    description: str
    input_shape: OutputShape = OutputShape.ANY
    output_shape: OutputShape = OutputShape.SAME
    params: List[Dict[str, Any]] = field(default_factory=list)
    data_types: List[str] = field(default_factory=lambda: ["all"])
    category: str = "general"
    version: str = "1.0.0"


@dataclass
class BlockResult:
    """Result from executing a preprocessing block."""
    data: np.ndarray
    output_shape: OutputShape
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class BasePreprocessingBlock(ABC):
    """Base class for all preprocessing blocks.
    
    All preprocessing blocks must inherit from this class and implement:
    - process(): The main processing logic
    - get_metadata(): Return block metadata
    
    Naming Convention:
    - File: block_{block_name}.py
    - Class: {BlockName}Block
    """
    
    # Class-level metadata (override in subclasses)
    block_type: str = "base"
    block_name: str = "Base Block"
    block_description: str = "Base preprocessing block"
    input_shape: OutputShape = OutputShape.ANY
    output_shape: OutputShape = OutputShape.SAME
    category: str = "general"
    version: str = "1.0.0"
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize block with parameters.
        
        Args:
            params: Dictionary of block-specific parameters
        """
        self.params = params or {}
        self._validate_params()
    
    def _validate_params(self):
        """Validate parameters against expected schema."""
        pass  # Override in subclasses if needed
    
    @abstractmethod
    def process(self, data: np.ndarray, **kwargs) -> BlockResult:
        """Process input data and return result.
        
        Args:
            data: Input numpy array
            **kwargs: Additional processing options
            
        Returns:
            BlockResult with processed data and metadata
        """
        pass
    
    @classmethod
    def get_metadata(cls) -> BlockMetadata:
        """Get block metadata for registration and UI display."""
        return BlockMetadata(
            name=cls.block_name,
            description=cls.block_description,
            input_shape=cls.input_shape,
            output_shape=cls.output_shape,
            params=cls.get_param_schema(),
            category=cls.category,
            version=cls.version,
        )
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        """Get parameter schema for the block.
        
        Override in subclasses to define parameters.
        
        Returns:
            List of parameter definitions with name, type, default, description
        """
        return []
    
    def get_info(self) -> Dict[str, Any]:
        """Get block instance information."""
        return {
            "type": self.block_type,
            "name": self.block_name,
            "params": self.params,
            "input_shape": self.input_shape.value,
            "output_shape": self.output_shape.value,
        }


class PreprocessingRegistry:
    """Registry for preprocessing blocks.
    
    Provides centralized registration and lookup of preprocessing blocks.
    """
    
    _blocks: Dict[str, Type[BasePreprocessingBlock]] = {}
    _metadata: Dict[str, BlockMetadata] = {}
    
    @classmethod
    def register(cls, block_class: Type[BasePreprocessingBlock]):
        """Register a preprocessing block.
        
        Args:
            block_class: The block class to register
        """
        block_type = block_class.block_type
        cls._blocks[block_type] = block_class
        cls._metadata[block_type] = block_class.get_metadata()
        logger.debug(f"Registered preprocessing block: {block_type}")
    
    @classmethod
    def get(cls, block_type: str) -> Optional[Type[BasePreprocessingBlock]]:
        """Get a block class by type.
        
        Args:
            block_type: The block type identifier
            
        Returns:
            The block class or None if not found
        """
        return cls._blocks.get(block_type)
    
    @classmethod
    def create(cls, block_type: str, params: Optional[Dict] = None) -> Optional[BasePreprocessingBlock]:
        """Create a block instance.
        
        Args:
            block_type: The block type identifier
            params: Block parameters
            
        Returns:
            Block instance or None if type not found
        """
        block_class = cls.get(block_type)
        if block_class:
            return block_class(params)
        return None
    
    @classmethod
    def list_blocks(cls) -> List[Dict[str, Any]]:
        """List all registered blocks with metadata.
        
        Returns:
            List of block metadata dictionaries
        """
        return [
            {
                "type": block_type,
                "name": meta.name,
                "description": meta.description,
                "input_shape": meta.input_shape.value,
                "output_shape": meta.output_shape.value,
                "params": meta.params,
                "category": meta.category,
            }
            for block_type, meta in cls._metadata.items()
        ]
    
    @classmethod
    def list_by_category(cls, category: str) -> List[Dict[str, Any]]:
        """List blocks filtered by category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of block metadata in the category
        """
        return [
            block for block in cls.list_blocks()
            if block["category"] == category
        ]
    
    @classmethod
    def list_by_input_shape(cls, shape: OutputShape) -> List[Dict[str, Any]]:
        """List blocks that accept a specific input shape.
        
        Args:
            shape: Input shape to filter by
            
        Returns:
            List of compatible blocks
        """
        return [
            block for block in cls.list_blocks()
            if block["input_shape"] in [shape.value, OutputShape.ANY.value]
        ]
    
    @classmethod
    def get_categories(cls) -> List[str]:
        """Get list of all categories."""
        return list(set(meta.category for meta in cls._metadata.values()))


def register_block(cls: Type[BasePreprocessingBlock]) -> Type[BasePreprocessingBlock]:
    """Decorator to register a preprocessing block.
    
    Usage:
        @register_block
        class MyBlock(BasePreprocessingBlock):
            block_type = "my_block"
            ...
    """
    PreprocessingRegistry.register(cls)
    return cls
