"""Enhanced Processing Pipeline Endpoints with CSI Preprocessing Blocks.

This module provides:
- CSI data preprocessing blocks
- Pipeline management with DL/ML sections
- Block configuration and connections
- Pipeline execution with CSI data flow
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from sqlalchemy.orm import Session
from enum import Enum
import json

from ..db import get_db, PreprocessingPipeline
from ..auth import get_current_user
from .models import StandardResponse

router = APIRouter(prefix="/enhanced-processing", tags=["enhanced-processing"])

# ============================================================================
# BLOCK TYPE DEFINITIONS
# ============================================================================

class BlockType(str, Enum):
    """Available processing block types."""
    
    # CSI Data Processing
    CSI_LOADER = "csi_loader"
    AMPLITUDE_EXTRACTOR = "amplitude_extractor"
    PHASE_EXTRACTOR = "phase_extractor"
    SUBCARRIER_FILTER = "subcarrier_filter"
    MOVING_AVERAGE = "moving_average"
    STATISTICAL_FILTER = "statistical_filter"
    FREQUENCY_DOMAIN_FILTER = "frequency_domain_filter"
    WAVELET_DENOISE = "wavelet_denoise"
    BASEBAND_FILTER = "baseband_filter"
    FILTER_CSI = "filter_csi"
    PCA_REDUCTION = "pca_reduction"
    DATA_PORTION_SELECTOR = "data_portion_selector"
    
    # Feature Engineering
    FEATURE_SCALER = "feature_scaler"
    FEATURE_SELECTOR = "feature_selector"
    FEATURE_ENGINEER = "feature_engineer"
    
    # Data Splitting
    TRAIN_TEST_SPLIT = "train_test_split"
    CROSS_VALIDATION = "cross_validation"
    
    # Model Training
    DL_TRAINER = "dl_trainer"
    ML_TRAINER = "ml_trainer"
    
    # Evaluation
    MODEL_EVALUATOR = "model_evaluator"
    METRICS_CALCULATOR = "metrics_calculator"
    
    # Utilities
    DATA_SAVER = "data_saver"
    MODEL_SAVER = "model_saver"
    VISUALIZER = "visualizer"

class ScalingMethod(str, Enum):
    """Feature scaling methods."""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    NORMALIZER = "normalizer"

class SelectionMethod(str, Enum):
    """Feature selection methods."""
    VARIANCE_THRESHOLD = "variance_threshold"
    SELECT_K_BEST = "select_k_best"
    RFE = "rfe"
    LASSO = "lasso"

# ============================================================================
# BLOCK CONFIGURATION MODELS
# ============================================================================

class BlockConfig(BaseModel):
    """Configuration for a processing block."""
    block_id: str
    block_type: BlockType
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = {}
    position: Dict[str, float] = {"x": 0, "y": 0}
    enabled: bool = True

class CSIConfig(BaseModel):
    """CSI loader configuration."""
    file_path: str
    include_phase: bool = True
    parse_options: Dict[str, Any] = {}

class SubcarrierFilterConfig(BaseModel):
    """Subcarrier filter configuration."""
    start_index: int = 5
    end_index: int = 32
    filter_method: str = "range"

class MovingAverageConfig(BaseModel):
    """Moving average configuration."""
    window_size: int = 5
    center: bool = True

class StatisticalFilterConfig(BaseModel):
    """Statistical filter configuration."""
    method: str = "zscore"
    threshold: float = 2.0
    axis: int = 0

class FrequencyFilterConfig(BaseModel):
    """Frequency domain filter configuration."""
    cutoff_frequency: float = 0.1
    filter_type: str = "lowpass"
    order: int = 5

class WaveletConfig(BaseModel):
    """Wavelet denoising configuration."""
    wavelet: str = "db4"
    level: int = 1
    threshold_method: str = "soft"

class BasebandFilterConfig(BaseModel):
    """Baseband filter configuration."""
    cutoff: float = 0.1
    order: int = 5
    sampling_rate: float = 1.0

class PCAConfig(BaseModel):
    """PCA reduction configuration."""
    n_components: Optional[int] = None
    variance_ratio: float = 0.95
    whiten: bool = False

class DataPortionConfig(BaseModel):
    """Data portion selector configuration."""
    sample_size: int = 1000
    step_size: int = 1000
    flatten: bool = True

class FeatureScalerConfig(BaseModel):
    """Feature scaler configuration."""
    method: ScalingMethod = ScalingMethod.STANDARD
    feature_range: tuple = (0, 1)

class FeatureSelectorConfig(BaseModel):
    """Feature selector configuration."""
    method: SelectionMethod = SelectionMethod.SELECT_K_BEST
    k: int = 10
    threshold: float = 0.01

class TrainTestSplitConfig(BaseModel):
    """Train-test split configuration."""
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    stratify: bool = True

class DLTrainerConfig(BaseModel):
    """Deep Learning trainer configuration."""
    model_type: str = "cnn"
    hidden_layers: List[int] = [512, 256, 128]
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    early_stopping: bool = True
    patience: int = 20

class MLTrainerConfig(BaseModel):
    """Machine Learning trainer configuration."""
    algorithm: str = "random_forest"
    hyperparameters: Dict[str, Any] = {}
    cross_validation: bool = True
    cv_folds: int = 5
    grid_search: bool = False

class ModelEvaluatorConfig(BaseModel):
    """Model evaluator configuration."""
    metrics: List[str] = ["accuracy", "precision", "recall", "f1"]
    cross_validate: bool = True
    cv_folds: int = 5

# ============================================================================
# PIPELINE MODELS
# ============================================================================

class PipelineConnection(BaseModel):
    """Connection between pipeline blocks."""
    source_block_id: str
    source_output: str
    target_block_id: str
    target_input: str

class EnhancedPipeline(BaseModel):
    """Enhanced processing pipeline."""
    pipeline_id: str
    name: str
    description: Optional[str] = None
    section: str  # "dl" or "ml"
    blocks: List[BlockConfig]
    connections: List[PipelineConnection]
    created_at: datetime
    updated_at: datetime
    created_by: str
    status: str = "draft"

# ============================================================================
# BLOCK TEMPLATES
# ============================================================================

BLOCK_TEMPLATES = {
    BlockType.CSI_LOADER: {
        "name": "CSI Data Loader",
        "description": "Load and parse CSI data from CSV files",
        "parameters": {
            "file_path": "",
            "include_phase": True,
            "parse_options": {}
        }
    },
    BlockType.AMPLITUDE_EXTRACTOR: {
        "name": "Amplitude Extractor",
        "description": "Extract amplitude from complex CSI values",
        "parameters": {}
    },
    BlockType.PHASE_EXTRACTOR: {
        "name": "Phase Extractor", 
        "description": "Extract phase from complex CSI values",
        "parameters": {}
    },
    BlockType.SUBCARRIER_FILTER: {
        "name": "Subcarrier Filter",
        "description": "Filter specific subcarrier ranges",
        "parameters": {
            "start_index": 5,
            "end_index": 32,
            "filter_method": "range"
        }
    },
    BlockType.MOVING_AVERAGE: {
        "name": "Moving Average Filter",
        "description": "Apply moving average smoothing",
        "parameters": {
            "window_size": 5,
            "center": True
        }
    },
    BlockType.STATISTICAL_FILTER: {
        "name": "Statistical Filter",
        "description": "Remove outliers using statistical methods",
        "parameters": {
            "method": "zscore",
            "threshold": 2.0,
            "axis": 0
        }
    },
    BlockType.FREQUENCY_DOMAIN_FILTER: {
        "name": "Frequency Domain Filter",
        "description": "Apply frequency domain filtering",
        "parameters": {
            "cutoff_frequency": 0.1,
            "filter_type": "lowpass",
            "order": 5
        }
    },
    BlockType.WAVELET_DENOISE: {
        "name": "Wavelet Denoising",
        "description": "Denoise using wavelet transform",
        "parameters": {
            "wavelet": "db4",
            "level": 1,
            "threshold_method": "soft"
        }
    },
    BlockType.BASEBAND_FILTER: {
        "name": "Baseband Filter",
        "description": "Apply baseband filtering",
        "parameters": {
            "cutoff": 0.1,
            "order": 5,
            "sampling_rate": 1.0
        }
    },
    BlockType.FILTER_CSI: {
        "name": "CSI Pattern Filter",
        "description": "Apply triplet pattern filtering",
        "parameters": {}
    },
    BlockType.PCA_REDUCTION: {
        "name": "PCA Dimensionality Reduction",
        "description": "Reduce dimensions using PCA",
        "parameters": {
            "n_components": None,
            "variance_ratio": 0.95,
            "whiten": False
        }
    },
    BlockType.DATA_PORTION_SELECTOR: {
        "name": "Data Portion Selector",
        "description": "Select data portions for training",
        "parameters": {
            "sample_size": 1000,
            "step_size": 1000,
            "flatten": True
        }
    },
    BlockType.FEATURE_SCALER: {
        "name": "Feature Scaler",
        "description": "Scale features using various methods",
        "parameters": {
            "method": "standard",
            "feature_range": [0, 1]
        }
    },
    BlockType.FEATURE_SELECTOR: {
        "name": "Feature Selector",
        "description": "Select best features",
        "parameters": {
            "method": "select_k_best",
            "k": 10,
            "threshold": 0.01
        }
    },
    BlockType.TRAIN_TEST_SPLIT: {
        "name": "Train-Test Split",
        "description": "Split data into train, validation, and test sets",
        "parameters": {
            "test_size": 0.2,
            "validation_size": 0.1,
            "random_state": 42,
            "stratify": True
        }
    },
    BlockType.DL_TRAINER: {
        "name": "Deep Learning Trainer",
        "description": "Train deep learning models",
        "parameters": {
            "model_type": "cnn",
            "hidden_layers": [512, 256, 128],
            "dropout_rate": 0.1,
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 32,
            "early_stopping": True,
            "patience": 20
        }
    },
    BlockType.ML_TRAINER: {
        "name": "Machine Learning Trainer",
        "description": "Train machine learning models",
        "parameters": {
            "algorithm": "random_forest",
            "hyperparameters": {},
            "cross_validation": True,
            "cv_folds": 5,
            "grid_search": False
        }
    },
    BlockType.MODEL_EVALUATOR: {
        "name": "Model Evaluator",
        "description": "Evaluate model performance",
        "parameters": {
            "metrics": ["accuracy", "precision", "recall", "f1"],
            "cross_validate": True,
            "cv_folds": 5
        }
    }
}

# ============================================================================
# IN-MEMORY STORAGE (for backward compatibility, prefer DB storage)
# ============================================================================

enhanced_pipelines: Dict[str, EnhancedPipeline] = {}
_pipeline_counter = 1

# ============================================================================
# MODEL AVAILABILITY BASED ON DATA SHAPE
# ============================================================================

MODELS_BY_OUTPUT_SHAPE = {
    "flattened": {
        "ml": ["knn", "svc", "random_forest", "gradient_boosting", "xgboost", 
               "lightgbm", "ada_boost", "decision_tree", "naive_bayes", 
               "logistic_regression", "linear_svc"],
        "dl": ["mlp", "autoencoder"]  # Simple feedforward networks
    },
    "sequence": {
        "ml": ["knn", "svc", "random_forest"],  # Can work with flattened sequences
        "dl": ["cnn", "lstm", "gru", "transformer", "cnn_lstm", "resnet"]
    }
}

def get_available_models(output_shape: str) -> Dict[str, List[str]]:
    """Get available models based on output shape."""
    return MODELS_BY_OUTPUT_SHAPE.get(output_shape, MODELS_BY_OUTPUT_SHAPE["flattened"])

# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/block-templates", response_model=Dict[str, Any])
async def get_block_templates(
    block_type: Optional[BlockType] = Query(None),
    section: Optional[str] = Query(None)
):
    """Get available block templates."""
    try:
        templates = {}
        
        for block_type_key, template in BLOCK_TEMPLATES.items():
            if block_type and block_type_key != block_type:
                continue
                
            # Filter by section if specified
            if section:
                if section == "dl" and block_type_key not in [
                    BlockType.CSI_LOADER, BlockType.AMPLITUDE_EXTRACTOR, 
                    BlockType.PHASE_EXTRACTOR, BlockType.SUBCARRIER_FILTER,
                    BlockType.MOVING_AVERAGE, BlockType.STATISTICAL_FILTER,
                    BlockType.FREQUENCY_DOMAIN_FILTER, BlockType.WAVELET_DENOISE,
                    BlockType.BASEBAND_FILTER, BlockType.FILTER_CSI,
                    BlockType.PCA_REDUCTION, BlockType.DATA_PORTION_SELECTOR,
                    BlockType.FEATURE_SCALER, BlockType.FEATURE_SELECTOR,
                    BlockType.TRAIN_TEST_SPLIT, BlockType.DL_TRAINER,
                    BlockType.MODEL_EVALUATOR
                ]:
                    continue
                elif section == "ml" and block_type_key not in [
                    BlockType.CSI_LOADER, BlockType.AMPLITUDE_EXTRACTOR,
                    BlockType.PHASE_EXTRACTOR, BlockType.SUBCARRIER_FILTER,
                    BlockType.MOVING_AVERAGE, BlockType.STATISTICAL_FILTER,
                    BlockType.FREQUENCY_DOMAIN_FILTER, BlockType.WAVELET_DENOISE,
                    BlockType.BASEBAND_FILTER, BlockType.FILTER_CSI,
                    BlockType.PCA_REDUCTION, BlockType.DATA_PORTION_SELECTOR,
                    BlockType.FEATURE_SCALER, BlockType.FEATURE_SELECTOR,
                    BlockType.TRAIN_TEST_SPLIT, BlockType.ML_TRAINER,
                    BlockType.MODEL_EVALUATOR
                ]:
                    continue
            
            templates[block_type_key.value] = template
        
        return {
            "success": True,
            "templates": templates,
            "total_blocks": len(templates)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get block templates: {str(e)}")

@router.post("/pipelines", response_model=StandardResponse)
async def create_enhanced_pipeline(
    name: str,
    section: str,
    description: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Create a new enhanced processing pipeline."""
    try:
        global _pipeline_counter
        
        pipeline_id = f"pipeline_{_pipeline_counter}"
        _pipeline_counter += 1
        
        pipeline = EnhancedPipeline(
            pipeline_id=pipeline_id,
            name=name,
            description=description,
            section=section,
            blocks=[],
            connections=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            created_by=current_user.id if hasattr(current_user, 'id') else "user"
        )
        
        enhanced_pipelines[pipeline_id] = pipeline
        
        return StandardResponse(
            success=True,
            message=f"Enhanced pipeline '{name}' created successfully",
            data={
                "pipeline_id": pipeline_id,
                "name": name,
                "section": section,
                "status": pipeline.status
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create enhanced pipeline: {str(e)}")

@router.get("/pipelines", response_model=Dict[str, Any])
async def list_enhanced_pipelines(
    section: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """List enhanced processing pipelines."""
    try:
        pipelines = []
        
        for pipeline in enhanced_pipelines.values():
            if section and pipeline.section != section:
                continue
            
            pipelines.append({
                "pipeline_id": pipeline.pipeline_id,
                "name": pipeline.name,
                "description": pipeline.description,
                "section": pipeline.section,
                "status": pipeline.status,
                "blocks_count": len(pipeline.blocks),
                "connections_count": len(pipeline.connections),
                "created_at": pipeline.created_at.isoformat(),
                "updated_at": pipeline.updated_at.isoformat()
            })
        
        return {
            "success": True,
            "pipelines": pipelines,
            "total": len(pipelines)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list enhanced pipelines: {str(e)}")

@router.get("/pipelines/{pipeline_id}", response_model=Dict[str, Any])
async def get_enhanced_pipeline(
    pipeline_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get enhanced pipeline details."""
    try:
        if pipeline_id not in enhanced_pipelines:
            raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
        
        pipeline = enhanced_pipelines[pipeline_id]
        
        return {
            "success": True,
            "pipeline": pipeline.model_dump()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get enhanced pipeline: {str(e)}")

@router.post("/pipelines/{pipeline_id}/blocks", response_model=StandardResponse)
async def add_block_to_pipeline(
    pipeline_id: str,
    block: BlockConfig,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Add a block to an enhanced pipeline."""
    try:
        if pipeline_id not in enhanced_pipelines:
            raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
        
        pipeline = enhanced_pipelines[pipeline_id]
        
        # Check if block ID already exists
        existing_ids = [b.block_id for b in pipeline.blocks]
        if block.block_id in existing_ids:
            raise ValueError(f"Block ID {block.block_id} already exists in pipeline")
        
        # Add block
        pipeline.blocks.append(block)
        pipeline.updated_at = datetime.utcnow()
        
        return StandardResponse(
            success=True,
            message=f"Block {block.block_id} added to pipeline successfully",
            data={
                "block_id": block.block_id,
                "block_type": block.block_type,
                "pipeline_id": pipeline_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add block to pipeline: {str(e)}")

@router.post("/pipelines/{pipeline_id}/connections", response_model=StandardResponse)
async def add_connection_to_pipeline(
    pipeline_id: str,
    connection: PipelineConnection,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Add a connection between blocks in an enhanced pipeline."""
    try:
        if pipeline_id not in enhanced_pipelines:
            raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
        
        pipeline = enhanced_pipelines[pipeline_id]
        
        # Validate that source and target blocks exist
        source_block_ids = [b.block_id for b in pipeline.blocks]
        if connection.source_block_id not in source_block_ids:
            raise ValueError(f"Source block {connection.source_block_id} not found in pipeline")
        
        if connection.target_block_id not in source_block_ids:
            raise ValueError(f"Target block {connection.target_block_id} not found in pipeline")
        
        # Add connection
        pipeline.connections.append(connection)
        pipeline.updated_at = datetime.utcnow()
        
        return StandardResponse(
            success=True,
            message="Connection added to pipeline successfully",
            data={
                "source_block_id": connection.source_block_id,
                "target_block_id": connection.target_block_id,
                "pipeline_id": pipeline_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add connection to pipeline: {str(e)}")

@router.put("/pipelines/{pipeline_id}", response_model=StandardResponse)
async def update_enhanced_pipeline(
    pipeline_id: str,
    blocks: List[BlockConfig],
    connections: List[PipelineConnection],
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Update an enhanced pipeline."""
    try:
        if pipeline_id not in enhanced_pipelines:
            raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
        
        pipeline = enhanced_pipelines[pipeline_id]
        
        # Update blocks and connections
        pipeline.blocks = blocks
        pipeline.connections = connections
        pipeline.updated_at = datetime.utcnow()
        
        return StandardResponse(
            success=True,
            message="Enhanced pipeline updated successfully",
            data={
                "pipeline_id": pipeline_id,
                "blocks_count": len(blocks),
                "connections_count": len(connections)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update enhanced pipeline: {str(e)}")

@router.delete("/pipelines/{pipeline_id}", response_model=StandardResponse)
async def delete_enhanced_pipeline(
    pipeline_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Delete an enhanced pipeline."""
    try:
        if pipeline_id not in enhanced_pipelines:
            raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
        
        del enhanced_pipelines[pipeline_id]
        
        return StandardResponse(
            success=True,
            message=f"Enhanced pipeline {pipeline_id} deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete enhanced pipeline: {str(e)}")

@router.post("/pipelines/{pipeline_id}/validate", response_model=StandardResponse)
async def validate_pipeline(
    pipeline_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Validate pipeline structure and connections."""
    try:
        if pipeline_id not in enhanced_pipelines:
            raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
        
        pipeline = enhanced_pipelines[pipeline_id]
        
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "block_validation": {},
            "connection_validation": {}
        }
        
        # Validate blocks
        block_ids = []
        for block in pipeline.blocks:
            block_ids.append(block.block_id)
            
            # Check if block type is valid
            if block.block_type not in BlockType:
                validation_results["errors"].append(f"Invalid block type: {block.block_type}")
                validation_results["valid"] = False
            
            # Check block parameters
            block_template = BLOCK_TEMPLATES.get(block.block_type, {})
            required_params = block_template.get("parameters", {})
            
            for param_name, param_value in required_params.items():
                if param_name not in block.parameters:
                    validation_results["warnings"].append(
                        f"Missing parameter '{param_name}' in block {block.block_id}"
                    )
        
        # Validate connections
        for connection in pipeline.connections:
            if connection.source_block_id not in block_ids:
                validation_results["errors"].append(
                    f"Source block {connection.source_block_id} not found"
                )
                validation_results["valid"] = False
            
            if connection.target_block_id not in block_ids:
                validation_results["errors"].append(
                    f"Target block {connection.target_block_id} not found"
                )
                validation_results["valid"] = False
        
        # Check for cycles (basic validation)
        if len(pipeline.connections) > 0:
            # This is a simplified cycle detection
            # In production, you'd implement proper graph cycle detection
            if len(pipeline.connections) > len(block_ids) * 2:
                validation_results["warnings"].append(
                    "High number of connections - potential cycles detected"
                )
        
        return StandardResponse(
            success=True,
            message="Pipeline validation completed",
            data=validation_results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to validate pipeline: {str(e)}")

@router.get("/available-models", response_model=Dict[str, Any])
async def get_available_models_endpoint(
    output_shape: str = Query("flattened", description="Output shape: 'flattened' or 'sequence'")
):
    """Get available models based on output shape configuration."""
    try:
        models = get_available_models(output_shape)
        return {
            "success": True,
            "output_shape": output_shape,
            "available_models": models,
            "description": {
                "flattened": "Data is flattened into feature vectors. Best for traditional ML models.",
                "sequence": "Data maintains temporal structure. Best for sequence models (LSTM, CNN, etc.)."
            }.get(output_shape, "Unknown output shape")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get available models: {str(e)}")


# ============================================================================
# DATABASE-BACKED PREPROCESSING PIPELINE ENDPOINTS
# ============================================================================

class CreatePipelineRequest(BaseModel):
    """Request model for creating a preprocessing pipeline."""
    name: str
    description: Optional[str] = None
    data_type: str = "csi"  # csi, imu, sensor
    output_shape: str = "flattened"  # flattened, sequence
    include_phase: bool = True
    window_size: int = 1000
    filter_subcarriers: bool = True
    subcarrier_start: int = 5
    subcarrier_end: int = 32
    config: Dict[str, Any] = {}  # Additional block configuration
    blocks: Optional[List[Dict[str, Any]]] = None  # Canvas blocks with positions and shapes
    connections: Optional[List[Dict[str, str]]] = None  # Connections between blocks


class UpdatePipelineRequest(BaseModel):
    """Request model for updating a preprocessing pipeline."""
    name: Optional[str] = None
    description: Optional[str] = None
    data_type: Optional[str] = None
    output_shape: Optional[str] = None
    include_phase: Optional[bool] = None
    window_size: Optional[int] = None
    filter_subcarriers: Optional[bool] = None
    subcarrier_start: Optional[int] = None
    subcarrier_end: Optional[int] = None
    config: Optional[Dict[str, Any]] = None
    blocks: Optional[List[Dict[str, Any]]] = None  # Canvas blocks with positions and shapes
    connections: Optional[List[Dict[str, str]]] = None  # Connections between blocks
    is_default: Optional[bool] = None


@router.post("/db-pipelines", response_model=StandardResponse)
async def create_db_pipeline(
    request: CreatePipelineRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Create a new preprocessing pipeline (database-backed)."""
    try:
        # Store blocks and connections in the config JSON
        config_data = request.config.copy() if request.config else {}
        if request.blocks is not None:
            config_data['blocks'] = request.blocks
        if request.connections is not None:
            config_data['connections'] = request.connections
        
        pipeline = PreprocessingPipeline(
            user_id=current_user.userId,
            name=request.name,
            description=request.description,
            data_type=request.data_type,
            output_shape=request.output_shape,
            include_phase=request.include_phase,
            window_size=request.window_size,
            filter_subcarriers=request.filter_subcarriers,
            subcarrier_start=request.subcarrier_start,
            subcarrier_end=request.subcarrier_end,
            config=json.dumps(config_data)
        )
        
        db.add(pipeline)
        db.commit()
        db.refresh(pipeline)
        
        return StandardResponse(
            success=True,
            message=f"Preprocessing pipeline '{request.name}' created successfully",
            data=pipeline.to_dict()
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create pipeline: {str(e)}")


@router.get("/db-pipelines", response_model=Dict[str, Any])
async def list_db_pipelines(
    data_type: Optional[str] = Query(None, description="Filter by data type (csi, imu)"),
    output_shape: Optional[str] = Query(None, description="Filter by output shape"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """List all preprocessing pipelines for the current user."""
    try:
        query = db.query(PreprocessingPipeline).filter(
            PreprocessingPipeline.user_id == current_user.userId
        )
        
        if data_type:
            query = query.filter(PreprocessingPipeline.data_type == data_type)
        if output_shape:
            query = query.filter(PreprocessingPipeline.output_shape == output_shape)
        
        pipelines = query.order_by(PreprocessingPipeline.created_at.desc()).all()
        
        return {
            "success": True,
            "pipelines": [p.to_dict() for p in pipelines],
            "total": len(pipelines)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list pipelines: {str(e)}")


@router.get("/db-pipelines/{pipeline_id}", response_model=Dict[str, Any])
async def get_db_pipeline(
    pipeline_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get a specific preprocessing pipeline."""
    try:
        pipeline = db.query(PreprocessingPipeline).filter(
            PreprocessingPipeline.id == pipeline_id,
            PreprocessingPipeline.user_id == current_user.userId
        ).first()
        
        if not pipeline:
            raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
        
        # Get available models for this pipeline's output shape
        available_models = get_available_models(pipeline.output_shape)
        
        return {
            "success": True,
            "pipeline": pipeline.to_dict(),
            "available_models": available_models
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline: {str(e)}")


@router.put("/db-pipelines/{pipeline_id}", response_model=StandardResponse)
async def update_db_pipeline(
    pipeline_id: int,
    request: UpdatePipelineRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Update a preprocessing pipeline."""
    try:
        pipeline = db.query(PreprocessingPipeline).filter(
            PreprocessingPipeline.id == pipeline_id,
            PreprocessingPipeline.user_id == current_user.userId
        ).first()
        
        if not pipeline:
            raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
        
        # Update fields if provided
        if request.name is not None:
            pipeline.name = request.name
        if request.description is not None:
            pipeline.description = request.description
        if request.data_type is not None:
            pipeline.data_type = request.data_type
        if request.output_shape is not None:
            pipeline.output_shape = request.output_shape
        if request.include_phase is not None:
            pipeline.include_phase = request.include_phase
        if request.window_size is not None:
            pipeline.window_size = request.window_size
        if request.filter_subcarriers is not None:
            pipeline.filter_subcarriers = request.filter_subcarriers
        if request.subcarrier_start is not None:
            pipeline.subcarrier_start = request.subcarrier_start
        if request.subcarrier_end is not None:
            pipeline.subcarrier_end = request.subcarrier_end
        # Handle config, blocks, and connections updates
        if request.config is not None or request.blocks is not None or request.connections is not None:
            # Load existing config
            existing_config = json.loads(pipeline.config) if pipeline.config else {}
            
            # Update with new config if provided
            if request.config is not None:
                existing_config.update(request.config)
            
            # Update blocks if provided
            if request.blocks is not None:
                existing_config['blocks'] = request.blocks
            
            # Update connections if provided
            if request.connections is not None:
                existing_config['connections'] = request.connections
            
            pipeline.config = json.dumps(existing_config)
        
        if request.is_default is not None:
            # If setting as default, unset other defaults for this user/data_type
            if request.is_default:
                db.query(PreprocessingPipeline).filter(
                    PreprocessingPipeline.user_id == current_user.userId,
                    PreprocessingPipeline.data_type == pipeline.data_type,
                    PreprocessingPipeline.is_default == True
                ).update({"is_default": False})
            pipeline.is_default = request.is_default
        
        db.commit()
        db.refresh(pipeline)
        
        return StandardResponse(
            success=True,
            message=f"Pipeline '{pipeline.name}' updated successfully",
            data=pipeline.to_dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update pipeline: {str(e)}")


@router.delete("/db-pipelines/{pipeline_id}", response_model=StandardResponse)
async def delete_db_pipeline(
    pipeline_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Delete a preprocessing pipeline."""
    try:
        pipeline = db.query(PreprocessingPipeline).filter(
            PreprocessingPipeline.id == pipeline_id,
            PreprocessingPipeline.user_id == current_user.userId
        ).first()
        
        if not pipeline:
            raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
        
        pipeline_name = pipeline.name
        db.delete(pipeline)
        db.commit()
        
        return StandardResponse(
            success=True,
            message=f"Pipeline '{pipeline_name}' deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete pipeline: {str(e)}")


@router.get("/db-pipelines/default/{data_type}", response_model=Dict[str, Any])
async def get_default_pipeline(
    data_type: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get the default preprocessing pipeline for a data type."""
    try:
        pipeline = db.query(PreprocessingPipeline).filter(
            PreprocessingPipeline.user_id == current_user.userId,
            PreprocessingPipeline.data_type == data_type,
            PreprocessingPipeline.is_default == True
        ).first()
        
        if not pipeline:
            # Return a suggested default config if no default is set
            return {
                "success": True,
                "pipeline": None,
                "suggested_default": {
                    "data_type": data_type,
                    "output_shape": "flattened",
                    "include_phase": True,
                    "window_size": 1000 if data_type == "csi" else 128,
                    "filter_subcarriers": True if data_type == "csi" else False,
                    "subcarrier_start": 5,
                    "subcarrier_end": 32
                }
            }
        
        return {
            "success": True,
            "pipeline": pipeline.to_dict(),
            "available_models": get_available_models(pipeline.output_shape)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get default pipeline: {str(e)}")


@router.get("/preprocessing-methods", response_model=Dict[str, Any])
async def get_preprocessing_methods():
    """Get available CSI preprocessing methods with details."""
    try:
        methods = {
            "amplitude_phase": {
                "name": "Amplitude and Phase Extraction",
                "description": "Extract amplitude and phase from complex CSI values",
                "parameters": ["include_phase"],
                "output_shape": "variable"
            },
            "subcarrier_filter": {
                "name": "Subcarrier Filtering",
                "description": "Filter specific subcarrier ranges to remove guard bands",
                "parameters": ["start_index", "end_index", "filter_method"],
                "output_shape": "reduced"
            },
            "moving_average": {
                "name": "Moving Average Filter",
                "description": "Apply moving average smoothing to reduce noise",
                "parameters": ["window_size", "center"],
                "output_shape": "same"
            },
            "statistical_filter": {
                "name": "Statistical Outlier Removal",
                "description": "Remove outliers using z-score or IQR methods",
                "parameters": ["method", "threshold", "axis"],
                "output_shape": "reduced"
            },
            "frequency_domain": {
                "name": "Frequency Domain Filtering",
                "description": "Apply low-pass, high-pass, or band-pass filters",
                "parameters": ["cutoff_frequency", "filter_type", "order"],
                "output_shape": "same"
            },
            "wavelet_denoise": {
                "name": "Wavelet Denoising",
                "description": "Denoise using wavelet transform with thresholding",
                "parameters": ["wavelet", "level", "threshold_method"],
                "output_shape": "same"
            },
            "baseband_filter": {
                "name": "Baseband Filtering",
                "description": "Apply Butterworth baseband filter",
                "parameters": ["cutoff", "order", "sampling_rate"],
                "output_shape": "same"
            },
            "filter_csi": {
                "name": "CSI Pattern Filtering",
                "description": "Apply triplet pattern-based filtering",
                "parameters": [],
                "output_shape": "reduced"
            },
            "pca_reduction": {
                "name": "PCA Dimensionality Reduction",
                "description": "Reduce dimensions using Principal Component Analysis",
                "parameters": ["n_components", "variance_ratio", "whiten"],
                "output_shape": "reduced"
            }
        }
        
        return {
            "success": True,
            "methods": methods,
            "total_methods": len(methods)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get preprocessing methods: {str(e)}")
