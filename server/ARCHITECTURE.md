# Brain Server Architecture

## Modular Structure

```
Brain/server/
├── main.py                # FastAPI app entry point
├── routes.py              # Central router (imports endpoint modules)
├── db.py                  # SQLAlchemy models
├── auth.py                # JWT authentication
├── config.py              # Environment configuration
├── startup.py             # Lifespan / startup tasks
├── services.py            # Background scheduler
├── entitlements.py        # Plan/feature gating
├── storage.py             # Upload quota checks
├── model_contract.py      # thoth-model/v1 artifact validation (TorchScript)
├── data_validation.py     # File-type validation (structural checks)
├── file_type_detector.py  # Content-based file type detection
├── db_health_monitor.py   # Database health monitoring
├── init_db.py             # Schema bootstrap
├── run_migrations.py      # Lightweight migrations
├── optimize_db.py         # Index/statistics maintenance
│
├── endpoints/             # FastAPI endpoint modules
│   ├── system_endpoints.py    # Health and info
│   ├── auth_endpoints.py      # Login, register, profile
│   ├── ai_endpoints.py        # AI assistant queries
│   ├── device_endpoints.py    # Device registry, pairing, commands
│   ├── data_endpoints.py      # Data operations
│   ├── file_endpoints.py      # File upload & management
│   ├── dataset_endpoints.py   # Datasets + model registry/deployments
│   ├── sensor_endpoints.py    # Sensor data
│   ├── network_endpoints.py   # WiFi configuration
│   ├── activity_endpoints.py  # Activity feed and stats
│   ├── validation_endpoints.py# File validation
│   ├── folders.py             # Folder management
│   ├── admin_endpoints.py     # Admin dashboard
│   ├── labs_endpoints.py      # Labs & submissions
│   ├── stripe_endpoints.py    # Stripe payments
│   ├── spatial_endpoints.py   # Spaces, zones, placement
│   ├── webhook_endpoints.py   # Twilio webhooks
│   ├── resumable_upload.py    # Resumable uploads
│   └── models.py              # Shared request/response models
│
├── sensors/               # Sensor helpers
├── utils/                 # Utilities (logging, storage, capture containers)
└── aiagent/               # AI agent (in parent directory)
    ├── handler/           # Query handlers
    ├── memory/            # Memory management
    ├── context/           # Context extraction
    └── functions/         # Function registry
```

## Key Modules

### Model registry (`endpoints/dataset_endpoints.py` + `model_contract.py`)
- Models are **uploaded artifacts** (`model.pt` + `thoth-model/v1` manifest),
  not produced by the server. `validate_torchscript` verifies the artifact
  loads on CPU and matches the declared input shapes.
- Deployments queue `DeviceDeployment` payloads that edge devices pull on
  heartbeat and acknowledge.

### Datasets
- `TrainingDataset`/`DatasetFile` tables group uploaded files with labels.
- Server-side window parsing/training was removed; parsing and windowing
  live in the Whispy edge SDK.

## Removed (training & federated learning cleanup)
- `fl/` — Flower-based federated learning package
- `ml/`, `ml_models/`, `dl_models/` — server-side model training
- `preprocessing/` — training preprocessing pipelines
- `metrics/`, `plotting/`, `reporting/`, `visualization/` — training metrics/figures
- `training_report.py`, `model_selector.py`, `figure_export.py`,
  `publication_plots.py`, `dataset_manager.py`
- `endpoints/figure_endpoints.py`, `plotting_api.py`, `report_endpoints.py`,
  `processing_endpoints.py`
- `TrainingJob` and `PreprocessingPipeline` ORM models and `/datasets/train/*`
  endpoints
