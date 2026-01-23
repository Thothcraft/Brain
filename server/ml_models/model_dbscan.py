"""DBSCAN Clustering Model.

Density-based clustering that finds arbitrarily shaped clusters.
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
class DBSCANModel(BaseMLModel):
    """DBSCAN clustering model.
    
    Density-based clustering that can find arbitrarily shaped clusters
    and automatically detect outliers.
    """
    
    model_type_id = "dbscan"
    model_name = "DBSCAN Clustering"
    model_description = "Density-based clustering with outlier detection"
    model_type = ModelType.CLUSTERING
    input_shape = "1d"
    supports_proba = False
    supports_feature_importance = False
    category = "clustering"
    version = "1.0.0"
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {"name": "eps", "type": "float", "default": 0.5, "description": "Maximum distance between samples"},
            {"name": "min_samples", "type": "int", "default": 5, "description": "Minimum samples in neighborhood"},
            {"name": "metric", "type": "str", "default": "euclidean", "options": ["euclidean", "manhattan", "cosine"], "description": "Distance metric"},
        ]
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> TrainingResult:
        """Fit DBSCAN clustering."""
        from sklearn.cluster import DBSCAN
        from sklearn.metrics import silhouette_score
        
        start_time = time.time()
        
        try:
            self.model = DBSCAN(
                eps=self.params.get("eps", 0.5),
                min_samples=self.params.get("min_samples", 5),
                metric=self.params.get("metric", "euclidean"),
                n_jobs=-1,
            )
            
            labels = self.model.fit_predict(X)
            self.is_fitted = True
            self.n_features_ = X.shape[1]
            self.labels_ = labels
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            
            metrics = {
                "n_clusters": n_clusters,
                "n_noise_points": int(n_noise),
                "noise_ratio": float(n_noise / len(labels)),
            }
            
            if n_clusters > 1:
                non_noise_mask = labels != -1
                if non_noise_mask.sum() > n_clusters:
                    metrics["silhouette_score"] = float(silhouette_score(X[non_noise_mask], labels[non_noise_mask]))
            
            return TrainingResult(
                success=True,
                train_time_ms=(time.time() - start_time) * 1000,
                metrics=metrics,
                metadata={
                    "cluster_sizes": [int((labels == i).sum()) for i in range(n_clusters)],
                },
            )
        except Exception as e:
            return TrainingResult(success=False, train_time_ms=(time.time() - start_time) * 1000, error=str(e))
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """DBSCAN doesn't support predict on new data directly."""
        return PredictionResult(
            predictions=self.labels_,
            inference_time_ms=0,
            metadata={"note": "DBSCAN returns labels from fit, not new predictions"},
        )
    
    def plot_clusters_2d(self, X: np.ndarray, save_path: Optional[str] = None) -> str:
        """Generate 2D cluster visualization with outliers highlighted."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        import io
        import base64
        
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
        else:
            X_2d = X[:, :2]
        
        labels = self.labels_
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 1)))
        
        for i, label in enumerate(sorted(unique_labels)):
            mask = labels == label
            if label == -1:
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c='gray', marker='x', label='Noise', alpha=0.5, s=30)
            else:
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[label % len(colors)]], label=f'Cluster {label}', alpha=0.6, s=50)
        
        ax.set_xlabel('Component 1', fontsize=12)
        ax.set_ylabel('Component 2', fontsize=12)
        ax.set_title(f'DBSCAN Clustering ({n_clusters} clusters found)', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_str
