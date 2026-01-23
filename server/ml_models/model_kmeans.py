"""K-Means Clustering Model.

Unsupervised clustering with professional visualization.
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
class KMeansModel(BaseMLModel):
    """K-Means clustering model.
    
    Partitions data into K clusters with centroid-based assignment.
    Includes professional 2D/3D visualization support.
    """
    
    model_type_id = "kmeans"
    model_name = "K-Means Clustering"
    model_description = "Partition data into K clusters with visualization"
    model_type = ModelType.CLUSTERING
    input_shape = "1d"
    supports_proba = False
    supports_feature_importance = False
    category = "clustering"
    version = "1.0.0"
    
    @classmethod
    def get_param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {"name": "n_clusters", "type": "int", "default": 3, "description": "Number of clusters"},
            {"name": "init", "type": "str", "default": "k-means++", "options": ["k-means++", "random"], "description": "Initialization method"},
            {"name": "n_init", "type": "int", "default": 10, "description": "Number of initializations"},
            {"name": "max_iter", "type": "int", "default": 300, "description": "Maximum iterations"},
        ]
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> TrainingResult:
        """Fit K-Means clustering."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        
        start_time = time.time()
        
        try:
            self.model = KMeans(
                n_clusters=self.params.get("n_clusters", 3),
                init=self.params.get("init", "k-means++"),
                n_init=self.params.get("n_init", 10),
                max_iter=self.params.get("max_iter", 300),
                random_state=42,
            )
            
            self.model.fit(X)
            self.is_fitted = True
            self.n_features_ = X.shape[1]
            labels = self.model.labels_
            
            # Calculate clustering metrics
            metrics = {
                "inertia": float(self.model.inertia_),
                "n_iterations": self.model.n_iter_,
            }
            
            if len(np.unique(labels)) > 1:
                metrics["silhouette_score"] = float(silhouette_score(X, labels))
                metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X, labels))
                metrics["davies_bouldin_score"] = float(davies_bouldin_score(X, labels))
            
            return TrainingResult(
                success=True,
                train_time_ms=(time.time() - start_time) * 1000,
                metrics=metrics,
                metadata={
                    "n_clusters": self.params.get("n_clusters", 3),
                    "cluster_sizes": [int((labels == i).sum()) for i in range(self.params.get("n_clusters", 3))],
                    "centroids": self.model.cluster_centers_.tolist(),
                },
            )
        except Exception as e:
            return TrainingResult(success=False, train_time_ms=(time.time() - start_time) * 1000, error=str(e))
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """Predict cluster assignments."""
        start_time = time.time()
        predictions = self.model.predict(X)
        
        return PredictionResult(
            predictions=predictions,
            inference_time_ms=(time.time() - start_time) * 1000,
            metadata={"distances_to_centroids": self.model.transform(X).tolist()[:10]},
        )
    
    def get_centroids(self) -> np.ndarray:
        """Get cluster centroids."""
        return self.model.cluster_centers_
    
    def plot_clusters_2d(self, X: np.ndarray, save_path: Optional[str] = None) -> str:
        """Generate 2D cluster visualization."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        import io
        import base64
        
        # Reduce to 2D if needed
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            centroids_2d = pca.transform(self.model.cluster_centers_)
        else:
            X_2d = X[:, :2]
            centroids_2d = self.model.cluster_centers_[:, :2]
        
        labels = self.model.labels_
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        
        colors = plt.cm.tab10(np.linspace(0, 1, self.params.get("n_clusters", 3)))
        
        for i in range(self.params.get("n_clusters", 3)):
            mask = labels == i
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=50)
        
        ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='black', marker='X', s=200, label='Centroids', edgecolors='white', linewidths=2)
        
        ax.set_xlabel('Component 1', fontsize=12)
        ax.set_ylabel('Component 2', fontsize=12)
        ax.set_title('K-Means Clustering (2D Projection)', fontsize=14, fontweight='bold')
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
    
    def plot_clusters_3d(self, X: np.ndarray, save_path: Optional[str] = None) -> str:
        """Generate 3D cluster visualization."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from sklearn.decomposition import PCA
        import io
        import base64
        
        # Reduce to 3D if needed
        if X.shape[1] > 3:
            pca = PCA(n_components=3)
            X_3d = pca.fit_transform(X)
            centroids_3d = pca.transform(self.model.cluster_centers_)
        else:
            X_3d = X[:, :3] if X.shape[1] >= 3 else np.hstack([X, np.zeros((X.shape[0], 3 - X.shape[1]))])
            centroids_3d = self.model.cluster_centers_[:, :3] if self.model.cluster_centers_.shape[1] >= 3 else np.hstack([self.model.cluster_centers_, np.zeros((self.model.cluster_centers_.shape[0], 3 - self.model.cluster_centers_.shape[1]))])
        
        labels = self.model.labels_
        
        fig = plt.figure(figsize=(12, 10), dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        
        colors = plt.cm.tab10(np.linspace(0, 1, self.params.get("n_clusters", 3)))
        
        for i in range(self.params.get("n_clusters", 3)):
            mask = labels == i
            ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2], c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=50)
        
        ax.scatter(centroids_3d[:, 0], centroids_3d[:, 1], centroids_3d[:, 2], c='black', marker='X', s=300, label='Centroids', edgecolors='white', linewidths=2)
        
        ax.set_xlabel('Component 1', fontsize=11)
        ax.set_ylabel('Component 2', fontsize=11)
        ax.set_zlabel('Component 3', fontsize=11)
        ax.set_title('K-Means Clustering (3D Projection)', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_str
