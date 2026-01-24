"""Enhanced Training Endpoints with DL/ML Split and CSI Data Support.

This module provides:
- Deep Learning (DL) training section with neural networks
- Machine Learning (ML) training section with classical algorithms
- CSI data preprocessing and handling
- Extensive configuration options for both sections
- Pipeline integration for preprocessing blocks
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, UploadFile, File as FastAPIFile
from typing import Dict, List, Optional, Any, Union, Callable
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import uuid
import json
import numpy as np
import pandas as pd
import math
import logging
from collections import defaultdict

# Import shared models
from .models import StandardResponse

# Import real training functions from ml_training
from server.ml_training import (
    train_model,
    train_ml_model,
    IMUClassifier,
    load_dataset_from_db,
    save_model_to_bytes,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/enhanced-training", tags=["enhanced-training"])

# ============================================================================
# ENUMS
# ============================================================================

class TrainingSection(str, Enum):
    """Training section types."""
    DEEP_LEARNING = "dl"
    MACHINE_LEARNING = "ml"

class DLModelType(str, Enum):
    """Deep Learning model architectures."""
    CNN = "cnn"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    CNN_LSTM = "cnn_lstm"
    AUTOENCODER = "autoencoder"
    VAE = "vae"
    GAN = "gan"
    RESNET = "resnet"
    CUSTOM = "custom"

class MLModelType(str, Enum):
    """Machine Learning algorithms."""
    KNN = "knn"
    SVC = "svc"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    ADA_BOOST = "ada_boost"
    DECISION_TREE = "decision_tree"
    NAIVE_BAYES = "naive_bayes"
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_SVC = "linear_svc"

class DataSource(str, Enum):
    """Training data sources."""
    CSI_DATA = "csi_data"
    IMU_DATA = "imu_data"
    SENSOR_DATA = "sensor_data"
    IMAGE_DATA = "image_data"
    AUDIO_DATA = "audio_data"
    TEXT_DATA = "text_data"
    CUSTOM = "custom"

class PreprocessingMethod(str, Enum):
    """CSI data preprocessing methods."""
    AMPLITUDE_PHASE = "amplitude_phase"
    AMPLITUDE_ONLY = "amplitude_only"
    PHASE_ONLY = "phase_only"
    STATISTICAL_FILTER = "statistical_filter"
    FREQUENCY_DOMAIN = "frequency_domain"
    WAVELET_DENOISE = "wavelet_denoise"
    BASEBAND_FILTER = "baseband_filter"
    FILTER_CSI = "filter_csi"
    MOVING_AVERAGE = "moving_average"
    PCA_REDUCTION = "pca_reduction"

# ============================================================================
# CONFIGURATION MODELS
# ============================================================================

class CSIConfig(BaseModel):
    """CSI data specific configuration."""
    include_phase: bool = True
    filter_subcarriers: bool = True
    subcarrier_range: tuple = (5, 32)  # Default range for filtering
    sample_size: int = 1000
    preprocessing_methods: List[PreprocessingMethod] = [PreprocessingMethod.AMPLITUDE_PHASE]
    moving_average_window: int = 5
    statistical_threshold: float = 2.0
    frequency_cutoff: float = 0.1
    wavelet_type: str = "db4"
    wavelet_level: int = 1
    baseband_cutoff: float = 0.1
    baseband_order: int = 5
    pca_components: Optional[int] = None

class DLConfig(BaseModel):
    """Deep Learning configuration."""
    model_type: DLModelType
    input_size: int
    hidden_layers: List[int] = [512, 256, 128]
    dropout_rate: float = 0.1
    activation: str = "relu"
    batch_norm: bool = True
    optimizer: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    loss_function: str = "crossentropy"
    epochs: int = 100
    batch_size: int = 32
    early_stopping_patience: int = 20
    scheduler: Optional[str] = None
    scheduler_params: Dict[str, Any] = {}
    
    # CNN specific
    conv_layers: List[Dict[str, Any]] = []
    pool_size: int = 2
    
    # LSTM specific
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_bidirectional: bool = True
    
    # Transformer specific
    num_heads: int = 8
    num_encoder_layers: int = 6
    d_model: int = 512
    dim_feedforward: int = 2048
    dropout: float = 0.1

class MLConfig(BaseModel):
    """Machine Learning configuration."""
    model_type: MLModelType
    
    # KNN specific
    n_neighbors: int = 5
    weights: str = "uniform"
    algorithm: str = "auto"
    
    # SVC specific
    kernel: str = "rbf"
    C: float = 1.0
    gamma: str = "scale"
    probability: bool = True
    
    # Random Forest specific
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    bootstrap: bool = True
    
    # General
    random_state: int = 42
    cross_validation_folds: int = 5
    grid_search: bool = False
    grid_search_params: Dict[str, Any] = {}

class EnhancedTrainingConfig(BaseModel):
    """Enhanced training configuration."""
    section: TrainingSection
    data_source: DataSource
    csi_config: Optional[CSIConfig] = None
    dl_config: Optional[DLConfig] = None
    ml_config: Optional[MLConfig] = None
    validation_split: float = 0.2
    test_split: float = 0.1
    metrics: List[str] = ["accuracy", "precision", "recall", "f1"]
    save_model: bool = True
    model_name: Optional[str] = None
    device_id: str = "thoth-001"
    pipeline_id: Optional[int] = None

class TrainingJob(BaseModel):
    """Training job information."""
    job_id: str
    config: EnhancedTrainingConfig
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_epoch: int = 0
    total_epochs: int
    metrics: Dict[str, List[float]] = {}
    best_metrics: Dict[str, float] = {}
    error_message: Optional[str] = None
    model_path: Optional[str] = None
    data_shape: Optional[tuple] = None
    preprocessing_info: Dict[str, Any] = {}

# ============================================================================
# IN-MEMORY STORAGE (Replace with database in production)
# ============================================================================

enhanced_training_jobs: Dict[str, TrainingJob] = {}
csi_data_cache: Dict[str, Dict[str, Any]] = {}

# ============================================================================
# CSI DATA PROCESSING FUNCTIONS
# ============================================================================

def parse_csi_data(file_content: str, verbose: bool = True) -> Dict[str, Any]:
    """Parse CSI data from CSV file content."""
    try:
        import pandas as pd
        import math
        import os
        from pathlib import Path
        
        # Parse CSV content
        lines = file_content.strip().split('\n')
        if len(lines) < 2:
            raise ValueError("File must have at least header and one data row")
        
        # Skip header and parse data rows
        csi_rows = []
        for line in lines[1:]:  # Skip header
            try:
                # Extract array content between brackets
                if '[' in line and ']' in line:
                    csi_row_str = line[line.index("[")+1 : line.index("]")]
                    csi_values = [float(x) for x in csi_row_str.split(",")]
                    csi_rows.append(csi_values)
            except Exception as e:
                if verbose:
                    print(f"Error parsing row: {e}")
                continue
        
        if not csi_rows:
            raise ValueError("No valid CSI data rows found")
        
        # Convert to DataFrame
        df = pd.DataFrame(csi_rows)
        
        # Extract amplitude and phase
        total_amps, total_phases = [], []
        for i, value in enumerate(df.values):
            imaginary, real, amplitudes, phases = [], [], [], []
            csi_one_row_lst = value.tolist()
            
            # Separate real and imaginary parts
            for item in range(len(csi_one_row_lst)):
                if item % 2 == 0:
                    imaginary.append(csi_one_row_lst[item])
                else:
                    real.append(csi_one_row_lst[item])
            
            val = len(csi_one_row_lst) // 2
            
            for k in range(val):
                amplitudes.append(round(math.sqrt(float(imaginary[k])**2 + float(real[k])**2), 4))
                phases.append(round(math.atan2(float(imaginary[k]), float(real[k])), 4))
            
            total_amps.append(np.array(amplitudes))
            total_phases.append(np.array(phases))
        
        amps_df = pd.DataFrame(total_amps)
        phases_df = pd.DataFrame(total_phases)
        
        return {
            "raw_data": df,
            "amplitude": amps_df,
            "phase": phases_df,
            "shape": df.shape,
            "samples": len(csi_rows)
        }
        
    except Exception as e:
        raise ValueError(f"Failed to parse CSI data: {str(e)}")

def filter_csi_subcarriers(df: pd.DataFrame, start_idx: int = 5, end_idx: int = 32) -> pd.DataFrame:
    """Filter CSI subcarriers based on specified range.
    
    For 802.11a/g/n, useful subcarriers are typically:
    - First range: start_idx to end_idx (e.g., 5:32 for 802.11n)
    - Second range: end_idx+1 to end_idx+28 (skip null guard band at end_idx)
    
    This removes null guard bands at the edges of the spectrum.
    
    Args:
        df: DataFrame with subcarrier columns
        start_idx: Start index for first range (default 5)
        end_idx: End index for first range (default 32)
        
    Returns:
        Filtered DataFrame with concatenated useful subcarriers
    """
    try:
        n_cols = df.shape[1]
        
        # Ensure indices are within bounds
        if end_idx + 28 > n_cols:
            # Fallback: just filter from start_idx to a reasonable end
            return df.iloc[:, start_idx:min(n_cols, start_idx + 54)]
        
        # First range: start_idx to end_idx (e.g., columns 5-31 for 802.11n)
        df1 = df.iloc[:, start_idx:end_idx]
        # Second range: skip guard band at end_idx, take next 27 columns
        df2 = df.iloc[:, end_idx+1:end_idx+28]
        
        filtered_df = pd.concat([df1, df2], axis=1)
        filtered_df.columns = range(filtered_df.shape[1])  # Reset column indices
        return filtered_df
    except Exception as e:
        # Fallback if range is invalid
        import logging
        logging.warning(f"Subcarrier filtering failed: {e}, using fallback")
        return df.iloc[:, 5:min(df.shape[1], 60)]

def select_data_portions(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """Select data portions for training."""
    selected_df_list = []
    for item in range(0, len(df) - sample_size, sample_size):
        selected_df = df.iloc[item:item+sample_size].to_numpy().flatten()
        selected_df_list.append(selected_df)
    return pd.DataFrame(selected_df_list)

def apply_preprocessing(data: Dict[str, Any], config: CSIConfig) -> Dict[str, Any]:
    """Apply preprocessing methods to CSI data."""
    try:
        processed_data = data.copy()
        preprocessing_info = {"methods_applied": []}
        
        # Start with amplitude and phase data
        amp_df = data["amplitude"]
        phase_df = data["phase"]
        
        # Apply subcarrier filtering
        if config.filter_subcarriers:
            amp_df = filter_csi_subcarriers(amp_df, config.subcarrier_range[0], config.subcarrier_range[1])
            phase_df = filter_csi_subcarriers(phase_df, config.subcarrier_range[0], config.subcarrier_range[1])
            preprocessing_info["methods_applied"].append("subcarrier_filtering")
        
        # Apply preprocessing methods
        for method in config.preprocessing_methods:
            if method == PreprocessingMethod.MOVING_AVERAGE:
                # Apply moving average
                amp_df = amp_df.rolling(window=config.moving_average_window).mean().dropna()
                phase_df = phase_df.rolling(window=config.moving_average_window).mean().dropna()
                preprocessing_info["methods_applied"].append("moving_average")
                
            elif method == PreprocessingMethod.STATISTICAL_FILTER:
                # Remove outliers using z-score
                from scipy import stats
                z_scores = np.abs(stats.zscore(amp_df))
                amp_df = amp_df[(z_scores < config.statistical_threshold).all(axis=1)]
                z_scores = np.abs(stats.zscore(phase_df))
                phase_df = phase_df[(z_scores < config.statistical_threshold).all(axis=1)]
                preprocessing_info["methods_applied"].append("statistical_filter")
                
            elif method == PreprocessingMethod.PCA_REDUCTION:
                # Apply PCA dimensionality reduction
                from sklearn.decomposition import PCA
                n_components = config.pca_components or min(50, min(amp_df.shape))
                pca = PCA(n_components=n_components)
                amp_df = pd.DataFrame(pca.fit_transform(amp_df))
                phase_df = pd.DataFrame(pca.fit_transform(phase_df))
                preprocessing_info["methods_applied"].append("pca_reduction")
                preprocessing_info["pca_components"] = n_components
        
        # Combine amplitude and phase if requested
        if config.include_phase:
            combined_df = pd.concat([amp_df, phase_df], axis=1)
        else:
            combined_df = amp_df
        
        # Select data portions
        final_df = select_data_portions(combined_df, config.sample_size)
        
        processed_data["processed"] = final_df
        processed_data["preprocessing_info"] = preprocessing_info
        
        return processed_data
        
    except Exception as e:
        raise ValueError(f"Preprocessing failed: {str(e)}")

# ============================================================================
# REAL TRAINING FUNCTIONS (using ml_training.py)
# ============================================================================

async def run_dl_training(job: TrainingJob, db_session=None):
    """Run real Deep Learning training using PyTorch.
    
    Uses IMUClassifier from ml_training.py for actual neural network training.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    
    try:
        job.status = "running"
        job.started_at = datetime.now()
        
        config = job.config.dl_config
        epochs = config.epochs
        
        logger.info(f"Starting DL training job {job.job_id}")
        logger.info(f"  Model: {config.model_type}, Epochs: {epochs}")
        
        # Check if we have a dataset_id to load real data
        dataset_id = getattr(job.config, 'dataset_id', None)
        
        if dataset_id and db_session:
            # Load real data from database
            logger.info(f"Loading dataset {dataset_id} from database...")
            X, y, class_names = load_dataset_from_db(
                db_session=db_session,
                dataset_id=dataset_id,
                window_size=config.window_size if hasattr(config, 'window_size') else 128,
                output_shape='sequence'
            )
            logger.info(f"Loaded {len(X)} samples, {len(class_names)} classes")
        else:
            # No dataset provided - return error
            raise ValueError(
                "No dataset_id provided. Please create a dataset with labeled files "
                "and provide the dataset_id in the training configuration."
            )
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=config.validation_split, 
            random_state=42, stratify=y
        )
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        
        # Determine input shape
        seq_length = X_train.shape[1]
        input_channels = X_train.shape[2] if len(X_train.shape) > 2 else 1
        num_classes = len(class_names)
        
        # Create model
        model = IMUClassifier(
            input_channels=input_channels,
            seq_length=seq_length,
            num_classes=num_classes,
            architecture_size=config.architecture_size if hasattr(config, 'architecture_size') else 'medium'
        )
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Training on device: {device}")
        
        # Training callback to update job progress
        def training_callback(epoch, total_epochs, train_loss, train_acc, val_loss, val_acc):
            job.current_epoch = epoch
            
            if "loss" not in job.metrics:
                job.metrics["loss"] = []
            if "accuracy" not in job.metrics:
                job.metrics["accuracy"] = []
            if "val_loss" not in job.metrics:
                job.metrics["val_loss"] = []
            if "val_accuracy" not in job.metrics:
                job.metrics["val_accuracy"] = []
            
            job.metrics["loss"].append(train_loss)
            job.metrics["accuracy"].append(train_acc)
            job.metrics["val_loss"].append(val_loss)
            job.metrics["val_accuracy"].append(val_acc)
            
            if "val_accuracy" not in job.best_metrics or val_acc > job.best_metrics["val_accuracy"]:
                job.best_metrics["val_accuracy"] = val_acc
                job.best_metrics["best_epoch"] = epoch
            
            logger.info(f"Epoch {epoch}/{total_epochs}: loss={train_loss:.4f}, acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        
        # Run real training
        results = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=epochs,
            learning_rate=config.learning_rate,
            device=device,
            callback=training_callback
        )
        
        # Save model
        if job.status == "running":
            job.status = "completed"
            job.completed_at = datetime.now()
            
            # Save model bytes
            model_bytes = save_model_to_bytes(model, {
                'model_type': config.model_type.value,
                'num_classes': num_classes,
                'class_names': class_names,
                'input_channels': input_channels,
                'seq_length': seq_length
            })
            job.model_bytes = model_bytes
            job.model_path = f"/models/dl/{job.job_id}/model.pth"
            
            logger.info(f"DL training completed. Best val accuracy: {results['best_val_accuracy']:.4f}")
            
    except Exception as e:
        logger.error(f"DL training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        job.status = "failed"
        job.error_message = str(e)

async def run_ml_training(job: TrainingJob, db_session=None):
    """Run real Machine Learning training using sklearn.
    
    Uses train_ml_model from ml_training.py for actual ML model training.
    """
    from sklearn.model_selection import train_test_split
    
    try:
        job.status = "running"
        job.started_at = datetime.now()
        
        config = job.config.ml_config
        
        logger.info(f"Starting ML training job {job.job_id}")
        logger.info(f"  Model: {config.model_type}")
        
        # Check if we have a dataset_id to load real data
        dataset_id = getattr(job.config, 'dataset_id', None)
        
        if dataset_id and db_session:
            # Load real data from database
            logger.info(f"Loading dataset {dataset_id} from database...")
            X, y, class_names = load_dataset_from_db(
                db_session=db_session,
                dataset_id=dataset_id,
                window_size=1000,  # ML models use flattened data
                output_shape='flattened'
            )
            logger.info(f"Loaded {len(X)} samples, {len(class_names)} classes")
        else:
            raise ValueError(
                "No dataset_id provided. Please create a dataset with labeled files "
                "and provide the dataset_id in the training configuration."
            )
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=config.validation_split if hasattr(config, 'validation_split') else 0.2,
            random_state=42, stratify=y
        )
        
        # Map model type to ml_training format
        model_type_map = {
            'knn': 'knn',
            'svc': 'svc',
            'ada_boost': 'adaboost',
            'random_forest': 'random_forest',
            'gradient_boosting': 'gradient_boosting',
            'decision_tree': 'decision_tree',
            'logistic_regression': 'logistic_regression',
        }
        ml_model_type = model_type_map.get(config.model_type.value, 'knn')
        
        # Run real ML training
        results = await train_ml_model(
            job_id=job.job_id,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            class_names=class_names,
            model_type=ml_model_type,
            config={'ml_params': config.model_dump() if hasattr(config, 'model_dump') else {}},
            db_session=db_session
        )
        
        # Update job with results
        job.metrics["accuracy"] = results['train_accuracies']
        job.metrics["val_accuracy"] = results['val_accuracies']
        job.best_metrics["accuracy"] = results['best_val_accuracy']
        job.best_metrics["val_accuracy"] = results['best_val_accuracy']
        
        job.current_epoch = 1  # ML models train in one step
        job.status = "completed"
        job.completed_at = datetime.now()
        job.model_bytes = results.get('model_bytes')
        job.model_path = f"/models/ml/{job.job_id}/model.pkl"
        
        logger.info(f"ML training completed. Val accuracy: {results['best_val_accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"ML training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        job.status = "failed"
        job.error_message = str(e)

# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/upload-csi-data", response_model=StandardResponse)
async def upload_csi_data(
    file: UploadFile = FastAPIFile(...),
    data_id: Optional[str] = None
):
    """Upload and parse CSI data from CSV file."""
    try:
        # Read file content
        content = await file.read()
        file_content = content.decode('utf-8')
        
        # Parse CSI data
        csi_data = parse_csi_data(file_content)
        
        # Generate data ID if not provided
        if not data_id:
            data_id = str(uuid.uuid4())
        
        # Cache the data
        csi_data_cache[data_id] = csi_data
        
        return StandardResponse(
            success=True,
            message="CSI data uploaded and parsed successfully",
            data={
                "data_id": data_id,
                "shape": csi_data["shape"],
                "samples": csi_data["samples"],
                "filename": file.filename
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to upload CSI data: {str(e)}")

@router.post("/preprocess-csi-data", response_model=StandardResponse)
async def preprocess_csi_data(
    data_id: str,
    config: CSIConfig
):
    """Apply preprocessing to uploaded CSI data."""
    try:
        if data_id not in csi_data_cache:
            raise HTTPException(status_code=404, detail="CSI data not found")
        
        # Apply preprocessing
        processed_data = apply_preprocessing(csi_data_cache[data_id], config)
        
        return StandardResponse(
            success=True,
            message="CSI data preprocessed successfully",
            data={
                "data_id": data_id,
                "original_shape": processed_data["shape"],
                "processed_shape": processed_data["processed"].shape,
                "preprocessing_info": processed_data["preprocessing_info"]
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to preprocess CSI data: {str(e)}")

@router.post("/start-training", response_model=StandardResponse)
async def start_enhanced_training(
    config: EnhancedTrainingConfig,
    background_tasks: BackgroundTasks,
    data_id: Optional[str] = None
):
    """Start enhanced training with DL/ML sections."""
    try:
        # Validate configuration
        if config.section == TrainingSection.DEEP_LEARNING and not config.dl_config:
            raise ValueError("DL configuration required for deep learning section")
        
        if config.section == TrainingSection.MACHINE_LEARNING and not config.ml_config:
            raise ValueError("ML configuration required for machine learning section")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Get data shape if CSI data is provided
        data_shape = None
        if data_id and data_id in csi_data_cache:
            data_shape = csi_data_cache[data_id]["shape"]
        
        # Create training job
        job = TrainingJob(
            job_id=job_id,
            config=config,
            status="pending",
            created_at=datetime.now(),
            total_epochs=config.dl_config.epochs if config.dl_config else config.ml_config.cross_validation_folds,
            data_shape=data_shape
        )
        
        # Store job
        enhanced_training_jobs[job_id] = job
        
        # Start real training in background (using ml_training.py functions)
        # Note: db_session needs to be passed for database access
        from server.db import SessionLocal
        db_session = SessionLocal()
        
        if config.section == TrainingSection.DEEP_LEARNING:
            background_tasks.add_task(run_dl_training, job, db_session)
        else:
            background_tasks.add_task(run_ml_training, job, db_session)
        
        return StandardResponse(
            success=True,
            message=f"Enhanced training job {job_id} created successfully",
            data={
                "job_id": job_id,
                "section": config.section,
                "model_type": config.dl_config.model_type.value if config.dl_config else config.ml_config.model_type.value,
                "data_source": config.data_source,
                "status": job.status
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start enhanced training: {str(e)}")

@router.get("/training-status", response_model=Union[TrainingJob, Dict[str, Any]])
async def get_enhanced_training_status(
    job_id: Optional[str] = Query(None),
    section: Optional[TrainingSection] = Query(None)
):
    """Get enhanced training status."""
    try:
        if job_id:
            if job_id not in enhanced_training_jobs:
                raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
            
            return enhanced_training_jobs[job_id]
        
        else:
            # Return all jobs, optionally filtered by section
            jobs = []
            for jid, job in enhanced_training_jobs.items():
                if section and job.config.section != section:
                    continue
                jobs.append(job)
            
            return {
                "success": True,
                "total_jobs": len(jobs),
                "jobs": jobs
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get training status: {str(e)}")

@router.get("/model-configs", response_model=Dict[str, Any])
async def get_model_configurations():
    """Get available model configurations and parameters."""
    try:
        return {
            "success": True,
            "sections": {
                "deep_learning": {
                    "model_types": [model.value for model in DLModelType],
                    "default_configs": {
                        "cnn": {
                            "hidden_layers": [512, 256, 128],
                            "dropout_rate": 0.1,
                            "conv_layers": [
                                {"filters": 32, "kernel_size": 3, "activation": "relu"},
                                {"filters": 64, "kernel_size": 3, "activation": "relu"}
                            ]
                        },
                        "lstm": {
                            "hidden_layers": [256, 128],
                            "lstm_hidden_size": 128,
                            "lstm_num_layers": 2,
                            "lstm_bidirectional": True
                        },
                        "transformer": {
                            "hidden_layers": [512, 256],
                            "num_heads": 8,
                            "num_encoder_layers": 6,
                            "d_model": 512
                        }
                    }
                },
                "machine_learning": {
                    "model_types": [model.value for model in MLModelType],
                    "default_configs": {
                        "knn": {"n_neighbors": 5, "weights": "uniform"},
                        "svc": {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
                        "random_forest": {"n_estimators": 100, "max_depth": None}
                    }
                }
            },
            "preprocessing_methods": [method.value for method in PreprocessingMethod],
            "data_sources": [source.value for source in DataSource]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model configurations: {str(e)}")

@router.get("/csi-data-info", response_model=Dict[str, Any])
async def get_csi_data_info(data_id: str):
    """Get information about uploaded CSI data."""
    try:
        if data_id not in csi_data_cache:
            raise HTTPException(status_code=404, detail="CSI data not found")
        
        data = csi_data_cache[data_id]
        
        return {
            "success": True,
            "data_id": data_id,
            "shape": data["shape"],
            "samples": data["samples"],
            "amplitude_shape": data["amplitude"].shape,
            "phase_shape": data["phase"].shape,
            "data_types": {
                "amplitude": str(data["amplitude"].dtypes),
                "phase": str(data["phase"].dtypes)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get CSI data info: {str(e)}")

@router.delete("/training-job/{job_id}", response_model=StandardResponse)
async def cancel_training_job(job_id: str):
    """Cancel a training job."""
    try:
        if job_id not in enhanced_training_jobs:
            raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
        
        job = enhanced_training_jobs[job_id]
        
        if job.status in ["completed", "failed"]:
            raise ValueError("Cannot cancel completed or failed jobs")
        
        job.status = "cancelled"
        job.completed_at = datetime.now()
        
        return StandardResponse(
            success=True,
            message=f"Training job {job_id} cancelled",
            data={"job_id": job_id, "status": job.status}
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
