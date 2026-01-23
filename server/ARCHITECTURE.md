# Brain Server Architecture

## Modular Structure

```
Brain/server/
├── fl/                      # Federated Learning (Flower-based)
│   ├── core/                # Config, models, client
│   ├── algorithms/          # FL strategies + knowledge distillation
│   ├── datasets/            # Data loading with flwr-datasets
│   ├── experiments/         # Multi-model experiments, pipelines
│   ├── visualization/       # FL-specific plots
│   └── session.py           # Session manager
│
├── ml/                      # Unified ML interface
│   ├── __init__.py          # Re-exports from ml_models/ and dl_models/
│   └── training.py          # Training utilities
│
├── ml_models/               # Classical ML models
│   ├── base.py              # BaseMLModel, MLModelRegistry
│   ├── model_svm.py
│   ├── model_random_forest.py
│   ├── model_knn.py
│   └── ...
│
├── dl_models/               # Deep Learning models (PyTorch)
│   ├── base.py              # BaseDLModel, DLModelRegistry
│   ├── model_lstm_3d.py
│   ├── model_cnn1d_3d.py
│   ├── model_transformer_3d.py
│   └── ...
│
├── preprocessing/           # Data preprocessing pipeline
│   ├── base.py              # BaseBlock, BlockRegistry
│   ├── pipeline.py          # PreprocessingPipeline
│   ├── block_*.py           # Individual blocks
│   └── ...
│
├── reporting/               # Training reports
│   └── __init__.py          # Re-exports from training_report.py
│
├── visualization/           # Visualization utilities
│   └── __init__.py          # Re-exports from figure_export.py
│
├── metrics/                 # Metrics tracking
│   ├── tracker.py
│   ├── visualizer.py
│   └── exporter.py
│
├── endpoints/               # FastAPI endpoints
│   ├── fl_endpoints.py      # Federated Learning API
│   ├── training_endpoints.py
│   └── ...
│
├── utils/                   # Utilities
│   ├── logging_utils.py
│   ├── error_handler.py
│   └── ...
│
└── aiagent/                 # AI Agent (in parent directory)
    ├── handler/             # Query handlers
    ├── memory/              # Memory management
    ├── context/             # Context extraction
    └── functions/           # Function registry
```

## Key Modules

### Federated Learning (`fl/`)
- Uses **Flower (flwr)** framework exclusively
- Supports 14+ FL algorithms (FedAvg, FedProx, FedAdam, etc.)
- Knowledge distillation (FedDF, FedMD) for heterogeneous models
- Multi-run experiments with statistical analysis

### Machine Learning (`ml/`, `ml_models/`, `dl_models/`)
- Unified interface via `ml/` module
- Classical ML: SVM, Random Forest, KNN, etc.
- Deep Learning: LSTM, GRU, CNN, Transformer, ResNet
- Model sizes: nano, mini, max

### Preprocessing (`preprocessing/`)
- Block-based pipeline architecture
- Blocks: normalization, windowing, filtering, FFT, PCA
- Composable and extensible

### Reporting & Visualization
- `reporting/`: Training metrics and reports
- `visualization/`: Publication-ready figures (IEEE style)
- `metrics/`: Real-time tracking and export

## Removed Files (Cleanup)
- `fl_algorithms/` - Duplicate FL implementations (now in `fl/`)
- `flower_fl.py` - Monolithic FL file (now modular in `fl/`)
- `modular_ml.py` - Duplicate code (consolidated)

## Usage Examples

### Federated Learning
```python
from server.fl import (
    create_experiment,
    FLExperimentRunner,
    list_pipelines,
)

# Run an experiment
experiment = create_experiment(
    name="CIFAR10-FedAvg",
    algorithm="fedavg",
    model="resnet18",
    num_runs=3,
)
runner = FLExperimentRunner()
result = await runner.run(experiment)
```

### Machine Learning
```python
from server.ml import create_model, list_all_models

# Create models
rf = create_model("random_forest", {"n_estimators": 100})
lstm = create_model("lstm_mini", {"input_size": 64, "num_classes": 4})
```

### Preprocessing
```python
from server.preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline([
    {"type": "zscore_normalize"},
    {"type": "sliding_window", "window_size": 128},
])
X_processed = pipeline.transform(X_raw)
```
