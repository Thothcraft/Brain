"""PCA Dimensionality Reduction Block.

Applies Principal Component Analysis for dimensionality reduction.
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
class PCAReductionBlock(BasePreprocessingBlock):
    """PCA dimensionality reduction preprocessing block.
    
    Reduces data dimensionality using Principal Component Analysis.
    """
    
    block_type = "pca_reduction"
    block_name = "PCA Reduction"
    block_description = "Reduce dimensionality with Principal Component Analysis"
    input_shape = OutputShape.SHAPE_2D
    output_shape = OutputShape.SHAPE_2D
    category = "dimensionality"
    version = "1.0.0"
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {
                "name": "n_components",
                "type": "int",
                "default": 50,
                "description": "Number of principal components to keep",
            },
            {
                "name": "variance_ratio",
                "type": "float",
                "default": None,
                "description": "Keep components explaining this variance ratio (0-1)",
            },
        ]
    
    def process(self, data: np.ndarray, **kwargs) -> BlockResult:
        """Apply PCA reduction."""
        n_components = self.params.get("n_components", 50)
        variance_ratio = self.params.get("variance_ratio")
        
        try:
            # Flatten if needed
            original_shape = data.shape
            if data.ndim > 2:
                data = data.reshape(data.shape[0], -1)
            
            # Center the data
            mean = data.mean(axis=0)
            centered = data - mean
            
            # Compute covariance and eigenvectors
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Sort by eigenvalue (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Determine number of components
            if variance_ratio is not None:
                total_var = eigenvalues.sum()
                cumsum = np.cumsum(eigenvalues) / total_var
                n_components = np.searchsorted(cumsum, variance_ratio) + 1
            
            n_components = min(n_components, data.shape[1], data.shape[0])
            
            # Project data
            components = eigenvectors[:, :n_components]
            reduced = centered @ components
            
            # Calculate explained variance
            explained_var = eigenvalues[:n_components].sum() / eigenvalues.sum()
            
            return BlockResult(
                data=reduced.astype(np.float32),
                output_shape=OutputShape.SHAPE_2D,
                metadata={
                    "original_features": original_shape[-1] if len(original_shape) > 1 else original_shape[0],
                    "reduced_features": n_components,
                    "explained_variance_ratio": float(explained_var),
                    "top_eigenvalues": eigenvalues[:5].tolist(),
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
