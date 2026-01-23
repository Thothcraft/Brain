# Machine Learning Module

Unified interface for classical ML and deep learning models.

## Structure

```
ml/
├── __init__.py          # Unified exports
├── training.py          # Training utilities
└── README.md            # This file

# Related modules (re-exported):
ml_models/               # Classical ML models
├── base.py              # BaseMLModel, MLModelRegistry
├── model_svm.py
├── model_random_forest.py
├── model_knn.py
├── model_logistic_regression.py
├── model_gradient_boosting.py
├── model_decision_tree.py
├── model_kmeans.py
├── model_dbscan.py
└── model_pca_visualizer.py

dl_models/               # Deep Learning models
├── base.py              # BaseDLModel, DLModelRegistry
├── model_mlp_1d.py
├── model_lstm_3d.py
├── model_gru_3d.py
├── model_cnn1d_3d.py
├── model_transformer_3d.py
├── model_cnn2d_4d.py
├── model_resnet_4d.py
└── model_cnn3d_4d.py
```

## Usage

```python
from server.ml import (
    create_model,
    list_all_models,
    ModelSelector,
)

# List all available models
models = list_all_models()
print(models['ml'])  # Classical ML models
print(models['dl'])  # Deep Learning models

# Create a model (auto-detects ML vs DL)
rf_model = create_model("random_forest", {"n_estimators": 100})
lstm_model = create_model("lstm_mini", {"input_size": 64, "num_classes": 4})

# Use ModelSelector for automatic recommendations
selector = ModelSelector()
recommendations = selector.get_recommendations(
    data_type="csi",
    output_shape="3d",
    size="mini"
)
```

## Model Types

### Classical ML (ml_models/)
- **SVM**: Support Vector Machine
- **Random Forest**: Ensemble of decision trees
- **KNN**: K-Nearest Neighbors
- **Logistic Regression**: Linear classifier
- **Gradient Boosting**: Boosted trees
- **Decision Tree**: Single tree classifier
- **K-Means**: Clustering
- **DBSCAN**: Density-based clustering
- **PCA Visualizer**: Dimensionality reduction

### Deep Learning (dl_models/)

| Model | Input Shape | Sizes | Use Case |
|-------|-------------|-------|----------|
| MLP | 1D (batch, features) | nano/mini/max | Flattened data |
| LSTM | 3D (batch, seq, feat) | nano/mini/max | Time series |
| GRU | 3D (batch, seq, feat) | nano/mini/max | Time series |
| CNN1D | 3D (batch, seq, feat) | nano/mini/max | Time series |
| Transformer | 3D (batch, seq, feat) | nano/mini/max | Time series |
| CNN2D | 4D (batch, c, h, w) | nano/mini/max | Images |
| ResNet | 4D (batch, c, h, w) | nano/mini/max | Images |
| CNN3D | 4D (batch, c, d, h, w) | nano/mini/max | Video |

## Training

```python
from server.ml.training import (
    train_pytorch_model,
    compute_metrics,
    create_data_loaders,
)

# Create data loaders
train_loader, val_loader = create_data_loaders(
    X_train, y_train, X_val, y_val, batch_size=32
)

# Train model
results = train_pytorch_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    learning_rate=0.001,
)

# Compute detailed metrics
metrics = compute_metrics(model, val_loader, class_names)
```
