"""PCA Visualizer Model.

Principal Component Analysis with professional 2D and 3D visualization.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional

from .base import (
    BaseMLModel,
    ModelType,
    TrainingResult,
    PredictionResult,
    register_model,
)


@register_model
class PCAVisualizerModel(BaseMLModel):
    """PCA Visualizer for dimensionality reduction and visualization.
    
    Provides professional 2D and 3D scatter plots with explained variance analysis.
    """
    
    model_type_id = "pca_visualizer"
    model_name = "PCA Visualizer"
    model_description = "Dimensionality reduction with professional 2D/3D visualization"
    model_type = ModelType.DIMENSIONALITY_REDUCTION
    input_shape = "1d"
    supports_proba = False
    supports_feature_importance = True
    category = "visualization"
    version = "1.0.0"
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {"name": "n_components", "type": "int", "default": 3, "description": "Number of components (2 or 3 for visualization)"},
            {"name": "whiten", "type": "bool", "default": False, "description": "Whiten components"},
        ]
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> TrainingResult:
        """Fit PCA model."""
        from sklearn.decomposition import PCA
        
        start_time = time.time()
        
        try:
            n_components = min(self.params.get("n_components", 3), X.shape[1], X.shape[0])
            
            self.model = PCA(
                n_components=n_components,
                whiten=self.params.get("whiten", False),
                random_state=42,
            )
            
            self.transformed_ = self.model.fit_transform(X)
            self.is_fitted = True
            self.n_features_ = X.shape[1]
            self.labels_ = y  # Store labels for visualization
            
            return TrainingResult(
                success=True,
                train_time_ms=(time.time() - start_time) * 1000,
                metrics={
                    "explained_variance_ratio": self.model.explained_variance_ratio_.tolist(),
                    "total_explained_variance": float(sum(self.model.explained_variance_ratio_)),
                    "n_components": n_components,
                },
                feature_importance=np.abs(self.model.components_).sum(axis=0).tolist(),
                metadata={
                    "singular_values": self.model.singular_values_.tolist(),
                },
            )
        except Exception as e:
            return TrainingResult(success=False, train_time_ms=(time.time() - start_time) * 1000, error=str(e))
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """Transform data to principal components."""
        start_time = time.time()
        transformed = self.model.transform(X)
        
        return PredictionResult(
            predictions=transformed,
            inference_time_ms=(time.time() - start_time) * 1000,
        )
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance based on component loadings."""
        return np.abs(self.model.components_).sum(axis=0)
    
    def plot_2d(self, X: np.ndarray = None, labels: np.ndarray = None, save_path: Optional[str] = None) -> str:
        """Generate professional 2D PCA scatter plot."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io
        import base64
        
        if X is not None:
            X_2d = self.model.transform(X)[:, :2]
        else:
            X_2d = self.transformed_[:, :2]
        
        if labels is None:
            labels = self.labels_
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[i]], label=f'{label}', alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
            ax.legend(title='Labels', loc='best', fontsize=10)
        else:
            ax.scatter(X_2d[:, 0], X_2d[:, 1], c='steelblue', alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
        
        var_ratio = self.model.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}% variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}% variance)', fontsize=12)
        ax.set_title(f'PCA 2D Projection (Total: {sum(var_ratio[:2])*100:.1f}% variance)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add annotation
        ax.annotate(f'n={len(X_2d)} samples', xy=(0.02, 0.98), xycoords='axes fraction', fontsize=9, va='top', ha='left', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_str
    
    def plot_3d(self, X: np.ndarray = None, labels: np.ndarray = None, save_path: Optional[str] = None) -> str:
        """Generate professional 3D PCA scatter plot."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import io
        import base64
        
        if X is not None:
            X_3d = self.model.transform(X)[:, :3]
        else:
            X_3d = self.transformed_[:, :3] if self.transformed_.shape[1] >= 3 else np.hstack([self.transformed_, np.zeros((self.transformed_.shape[0], 3 - self.transformed_.shape[1]))])
        
        if labels is None:
            labels = self.labels_
        
        fig = plt.figure(figsize=(12, 10), dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2], c=[colors[i]], label=f'{label}', alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
            ax.legend(title='Labels', loc='best', fontsize=10)
        else:
            ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c='steelblue', alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
        
        var_ratio = self.model.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)', fontsize=11)
        ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)', fontsize=11)
        ax.set_zlabel(f'PC3 ({var_ratio[2]*100:.1f}%)' if len(var_ratio) > 2 else 'PC3', fontsize=11)
        ax.set_title(f'PCA 3D Projection (Total: {sum(var_ratio[:3])*100:.1f}% variance)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_str
    
    def plot_explained_variance(self, save_path: Optional[str] = None) -> str:
        """Plot explained variance ratio (scree plot)."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io
        import base64
        
        var_ratio = self.model.explained_variance_ratio_
        cumsum = np.cumsum(var_ratio)
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        
        x = range(1, len(var_ratio) + 1)
        ax.bar(x, var_ratio * 100, alpha=0.7, color='steelblue', label='Individual')
        ax.plot(x, cumsum * 100, 'ro-', linewidth=2, markersize=8, label='Cumulative')
        
        ax.set_xlabel('Principal Component', fontsize=12)
        ax.set_ylabel('Explained Variance (%)', fontsize=12)
        ax.set_title('PCA Explained Variance (Scree Plot)', fontsize=14, fontweight='bold')
        ax.legend(loc='center right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(x)
        
        # Add 80% and 95% threshold lines
        ax.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='80% threshold')
        ax.axhline(y=95, color='green', linestyle='--', alpha=0.7, label='95% threshold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_str
    
    def plot_loadings(self, feature_names: Optional[List[str]] = None, save_path: Optional[str] = None) -> str:
        """Plot PCA component loadings heatmap."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io
        import base64
        
        loadings = self.model.components_
        n_components, n_features = loadings.shape
        
        if feature_names is None:
            feature_names = [f'F{i}' for i in range(n_features)]
        
        # Limit features for readability
        max_features = 30
        if n_features > max_features:
            # Show top features by importance
            importance = np.abs(loadings).sum(axis=0)
            top_idx = np.argsort(importance)[-max_features:]
            loadings = loadings[:, top_idx]
            feature_names = [feature_names[i] for i in top_idx]
        
        fig, ax = plt.subplots(figsize=(max(10, len(feature_names) * 0.3), 6), dpi=150)
        
        im = ax.imshow(loadings, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(n_components))
        ax.set_yticklabels([f'PC{i+1}' for i in range(n_components)])
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Principal Components', fontsize=12)
        ax.set_title('PCA Component Loadings', fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Loading')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_str
