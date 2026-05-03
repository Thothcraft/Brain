"""Dataset and Cloud Training Endpoints.

This module handles:
- Dataset creation and management
- File labeling for training
- Cloud training job management
- Model evaluation and deployment
"""

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks, Body
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from sqlalchemy.orm import Session
import uuid
import json
import asyncio
import random
import logging

from sqlalchemy.orm import selectinload, load_only

from ..db import get_db, TrainingDataset, DatasetFile, TrainingJob, TrainedModel, File, PreprocessingPipeline
from ..auth import get_current_user
from .models import StandardResponse

router = APIRouter(prefix="/datasets", tags=["datasets"])

logger = logging.getLogger(__name__)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class CreateDatasetRequest(BaseModel):
    """Request to create a new training dataset."""
    name: str
    description: Optional[str] = None


class AddFilesToDatasetRequest(BaseModel):
    """Request to add files with labels to a dataset."""
    files: List[Dict[str, Any]]  # [{file_id: int, label: str}, ...]


class UpdateFileLabelRequest(BaseModel):
    """Request to update a file's label."""
    label: str


class CloudTrainingRequest(BaseModel):
    """Request to start cloud training."""
    dataset_id: int
    test_dataset_id: Optional[int] = None
    model_type: str = "cnn"  # cnn, lstm, transformer, linear, knn, svc, adaboost
    model_architecture: str = "small"  # small, medium, large
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    test_split: float = 0.0
    model_name: Optional[str] = None
    window_size: int = 128  # Window size for time series data
    
    # Preprocessing pipeline (optional - if not set, uses inline config)
    preprocessing_pipeline_id: Optional[int] = None

    # Explicit preprocessing blocks (optional override; used by preview/training)
    preprocessing_blocks: Optional[List[Dict[str, Any]]] = None
    
    # CSI-specific preprocessing options (used if no pipeline_id)
    data_type: str = "auto"  # auto, csi, imu
    include_phase: bool = True  # CSI: include phase data
    filter_subcarriers: bool = True  # CSI: filter guard bands
    subcarrier_start: int = 5  # CSI: start index for filtering
    subcarrier_end: int = 32  # CSI: end index for filtering
    output_shape: str = "flattened"  # flattened (ML) or sequence (DL)

    # ML-specific hyperparameters (used for knn/svc/adaboost)
    ml_params: Optional[Dict[str, Any]] = None

    # Optimization (optional)
    grid_search: Optional[Dict[str, Any]] = None
    
    # Bayesian optimization settings
    use_bayesian_optimization: bool = False
    bayesian_trials: int = 20
    bayesian_epochs_per_trial: int = 3
    bayesian_lr_min: float = 0.00001
    bayesian_lr_max: float = 0.01
    bayesian_lr_scale: str = "log"  # 'log' or 'linear'
    bayesian_batch_sizes: List[int] = [16, 32, 64, 128]
    bayesian_weight_decay_min: float = 0.0
    bayesian_weight_decay_max: float = 0.01
    bayesian_optimizers: List[str] = ["adam", "adamw", "sgd"]
    bayesian_exploration_rate: float = 0.3
    bayesian_search_architecture: bool = False  # Whether to search architecture sizes


# ============================================================================
# DATASET ENDPOINTS
# ============================================================================

@router.post("/create", response_model=Dict[str, Any])
async def create_dataset(
    request: CreateDatasetRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Create a new training dataset."""
    try:
        dataset = TrainingDataset(
            user_id=current_user.userId,
            name=request.name,
            description=request.description
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        
        return {
            "success": True,
            "message": f"Dataset '{request.name}' created successfully",
            "dataset": dataset.to_dict()
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create dataset: {str(e)}")


@router.get("/list", response_model=Dict[str, Any])
async def list_datasets(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """List all datasets for the current user."""
    try:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[DATASETS] Starting datasets query for user {current_user.userId}")
        
        # Optimized query - select only necessary columns
        datasets = db.query(
            TrainingDataset.id,
            TrainingDataset.name,
            TrainingDataset.description,
            TrainingDataset.created_at,
            TrainingDataset.updated_at
        ).filter(
            TrainingDataset.user_id == current_user.userId
        ).order_by(TrainingDataset.created_at.desc()).limit(100).all()  # Add limit to prevent large result sets
        
        logger.info(f"[DATASETS] Query completed, found {len(datasets)} datasets")
        
        # Convert to dict efficiently
        dataset_list = []
        for d in datasets:
            dataset_list.append({
                "id": d.id,
                "name": d.name,
                "description": d.description,
                "created_at": d.created_at.isoformat() if d.created_at else None,
                "updated_at": d.updated_at.isoformat() if d.updated_at else None
            })
        
        return {
            "success": True,
            "datasets": dataset_list,
            "total": len(dataset_list),
            "operation": "list_datasets",
            "status": "completed"
        }
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"[DATASETS] Error listing datasets: {str(e)}")
        # Return a consistent error response structure
        return {
            "success": False,
            "datasets": [],
            "total": 0,
            "operation": "list_datasets",
            "status": "error",
            "error": str(e)
        }


class PreprocessingPreviewRequest(BaseModel):
    preprocessing_pipeline_id: Optional[int] = None
    preprocessing_blocks: Optional[List[Dict[str, Any]]] = None
    data_type: str = "auto"  # auto, csi, imu
    include_phase: bool = True
    filter_subcarriers: bool = True
    subcarrier_start: int = 5
    subcarrier_end: int = 32
    output_shape: str = "flattened"  # flattened or sequence
    window_size: int = 1000
    max_preview_values: int = 32


@router.post("/{dataset_id}/preprocessing/preview", response_model=Dict[str, Any])
async def preview_preprocessing(
    dataset_id: int,
    request: PreprocessingPreviewRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    try:
        dataset = db.query(TrainingDataset).filter(
            TrainingDataset.id == dataset_id,
            TrainingDataset.user_id == current_user.userId
        ).first()

        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        df_row = db.query(DatasetFile).filter(
            DatasetFile.dataset_id == dataset_id
        ).order_by(DatasetFile.id.asc()).first()

        if not df_row:
            raise HTTPException(status_code=400, detail="Dataset has no files")

        file_row = db.query(File).filter(
            File.fileId == df_row.file_id,
            File.userId == current_user.userId
        ).first()

        if not file_row or not file_row.content:
            raise HTTPException(status_code=400, detail="Dataset file content not available")

        effective = {
            "preprocessing_pipeline_id": request.preprocessing_pipeline_id,
            "preprocessing_blocks": request.preprocessing_blocks,
            "data_type": request.data_type,
            "include_phase": request.include_phase,
            "filter_subcarriers": request.filter_subcarriers,
            "subcarrier_start": request.subcarrier_start,
            "subcarrier_end": request.subcarrier_end,
            "output_shape": request.output_shape,
            "window_size": request.window_size,
        }

        pipeline = None
        if request.preprocessing_pipeline_id is not None:
            pipeline = db.query(PreprocessingPipeline).filter(
                PreprocessingPipeline.id == request.preprocessing_pipeline_id,
                PreprocessingPipeline.user_id == current_user.userId
            ).first()
            if pipeline:
                effective.update({
                    "data_type": pipeline.data_type,
                    "include_phase": bool(pipeline.include_phase),
                    "filter_subcarriers": bool(pipeline.filter_subcarriers),
                    "subcarrier_start": int(pipeline.subcarrier_start or 5),
                    "subcarrier_end": int(pipeline.subcarrier_end or 32),
                    "output_shape": pipeline.output_shape or "flattened",
                    "window_size": int(pipeline.window_size or request.window_size),
                })

                # Prefer pipeline blocks if request did not provide an override
                if not effective.get("preprocessing_blocks"):
                    try:
                        pipeline_cfg = json.loads(pipeline.config) if pipeline.config else {}
                        pipeline_blocks = pipeline_cfg.get("blocks") if isinstance(pipeline_cfg, dict) else None
                        if isinstance(pipeline_blocks, list) and pipeline_blocks:
                            effective["preprocessing_blocks"] = pipeline_blocks
                    except Exception:
                        pass

        filename = (file_row.filename or "").lower()
        content_type = (file_row.content_type or "").lower()
        data_type = effective["data_type"]
        if data_type == "auto":
            if "imu" in filename or "json" in content_type:
                data_type = "imu"
            elif "csi" in filename or "csv" in content_type:
                data_type = "csi"
            else:
                data_type = "csi"

        stages: List[Dict[str, Any]] = []
        max_vals = max(8, min(256, int(request.max_preview_values or 32)))

        # If we have a block graph, use the unified pipeline executor
        if effective.get("preprocessing_blocks"):
            try:
                from server.ml_training import execute_preprocessing_pipeline_preview
                preview = execute_preprocessing_pipeline_preview(
                    content=file_row.content,
                    filename=file_row.filename or "",
                    base_config={
                        "data_type": data_type,
                        "include_phase": effective.get("include_phase"),
                        "filter_subcarriers": effective.get("filter_subcarriers"),
                        "subcarrier_start": effective.get("subcarrier_start"),
                        "subcarrier_end": effective.get("subcarrier_end"),
                        "output_shape": effective.get("output_shape"),
                        "window_size": effective.get("window_size"),
                    },
                    pipeline_blocks=effective.get("preprocessing_blocks") or [],
                    max_preview_values=max_vals,
                )
                return {
                    "success": True,
                    "preview": {
                        "dataset_id": dataset_id,
                        "file": {
                            "file_id": file_row.fileId,
                            "filename": file_row.filename,
                            "size": file_row.size,
                            "content_type": file_row.content_type,
                        },
                        "data_type": data_type,
                        "effective_config": effective,
                        "stages": preview.get("stages", []),
                    }
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Pipeline preview failed: {str(e)}")

        if data_type == "csi":
            import numpy as np

            text_content = file_row.content.decode('utf-8', errors='ignore').lstrip('\ufeff').strip()
            lines = text_content.split('\n')

            # For preview, limit rows to 2x window_size to avoid timeout on large files
            preview_row_limit = max(2000, int(effective["window_size"]) * 2)
            total_data_lines = len(lines) - 1  # Exclude header

            csi_rows: List[List[float]] = []
            for line in lines[1:preview_row_limit + 1]:
                line = line.strip()
                if not line:
                    continue
                if '[' not in line or ']' not in line:
                    continue
                try:
                    csi_str = line[line.index('[') + 1: line.index(']')]
                    csi_values = [float(x.strip()) for x in csi_str.split(',') if x.strip()]
                    if csi_values:
                        csi_rows.append(csi_values)
                except Exception:
                    continue

            if not csi_rows:
                raise HTTPException(status_code=400, detail="No valid CSI rows found")

            lengths = [len(r) for r in csi_rows]
            expected_len = int(max(set(lengths), key=lengths.count))
            expected_len = expected_len - (expected_len % 2)
            cleaned = []
            dropped_rows = 0
            for r in csi_rows:
                if len(r) < expected_len:
                    dropped_rows += 1
                    continue
                rr = r[:expected_len]
                arr = np.asarray(rr, dtype=np.float32)
                if not np.isfinite(arr).all():
                    dropped_rows += 1
                    continue
                cleaned.append(arr)

            if not cleaned:
                raise HTTPException(status_code=400, detail="No valid CSI rows after cleaning")

            csi_arr = np.stack(cleaned, axis=0)
            imag = csi_arr[:, 0::2]
            real = csi_arr[:, 1::2]
            amp_arr = np.sqrt(imag ** 2 + real ** 2)
            phase_arr = np.arctan2(imag, real)

            stages.append({
                "block": "csi_loader",
                "name": "CSI Loader",
                "shape": list(csi_arr.shape),
                "sample": csi_arr[0, :max_vals].tolist(),
                "metadata": {
                    "total_rows": int(csi_arr.shape[0]),
                    "expected_row_len": int(expected_len),
                    "dropped_rows": int(dropped_rows),
                    "total_file_rows": int(total_data_lines),
                    "preview_limited": total_data_lines > preview_row_limit
                }
            })

            stages.append({
                "block": "amplitude_extractor",
                "name": "Amplitude Extractor",
                "shape": list(amp_arr.shape),
                "sample": amp_arr[0, :max_vals].tolist(),
            })

            if effective["include_phase"]:
                stages.append({
                    "block": "phase_extractor",
                    "name": "Phase Extractor",
                    "shape": list(phase_arr.shape),
                    "sample": phase_arr[0, :max_vals].tolist(),
                })

            if effective["filter_subcarriers"] and amp_arr.shape[1] > int(effective["subcarrier_end"]) + 27:
                s = int(effective["subcarrier_start"])
                e = int(effective["subcarrier_end"])
                amp_arr = np.concatenate([amp_arr[:, s:e], amp_arr[:, e + 1:e + 28]], axis=1)
                phase_arr = np.concatenate([phase_arr[:, s:e], phase_arr[:, e + 1:e + 28]], axis=1)

                stages.append({
                    "block": "subcarrier_filter",
                    "name": "Subcarrier Filter",
                    "shape": list(amp_arr.shape),
                    "sample": amp_arr[0, :max_vals].tolist(),
                    "metadata": {"subcarrier_start": s, "subcarrier_end": e}
                })

            combined = np.concatenate([amp_arr, phase_arr], axis=1) if effective["include_phase"] else amp_arr
            if not np.isfinite(combined).all():
                combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)

            stages.append({
                "block": "feature_concat",
                "name": "Feature Combine",
                "shape": list(combined.shape),
                "sample": combined[0, :max_vals].tolist(),
            })

            window_size = int(effective["window_size"])
            if combined.shape[0] < window_size:
                stages.append({
                    "block": "data_portion_selector",
                    "name": "Windowing",
                    "error": f"Not enough rows ({int(combined.shape[0])}) for window_size={window_size}",
                })
            else:
                windows = []
                for start in range(0, combined.shape[0] - window_size + 1, window_size):
                    w = combined[start:start + window_size]
                    windows.append(w.reshape(-1).astype(np.float32) if effective["output_shape"] == "flattened" else w.astype(np.float32))
                if windows:
                    sample_window = windows[0]
                    stages.append({
                        "block": "data_portion_selector",
                        "name": "Window + Flatten" if effective["output_shape"] == "flattened" else "Window (Sequence)",
                        "shape": list(sample_window.shape),
                        "sample": sample_window.reshape(-1)[:max_vals].tolist(),
                        "metadata": {"n_windows": len(windows), "window_size": window_size, "output_shape": effective["output_shape"]}
                    })

        else:
            from server.ml_training import parse_imu_file
            windows = parse_imu_file(file_row.content, window_size=int(effective["window_size"]))
            sample_window = windows[0] if windows else None
            stages.append({
                "block": "imu_loader",
                "name": "IMU Loader",
                "shape": list(sample_window.shape) if sample_window is not None else None,
                "sample": sample_window.reshape(-1)[:max_vals].tolist() if sample_window is not None else None,
                "metadata": {"n_windows": len(windows), "window_size": int(effective["window_size"])}
            })

        return {
            "success": True,
            "preview": {
                "dataset_id": dataset_id,
                "file": {
                    "file_id": file_row.fileId,
                    "filename": file_row.filename,
                    "size": file_row.size,
                    "content_type": file_row.content_type,
                },
                "data_type": data_type,
                "effective_config": effective,
                "stages": stages,
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"[PREVIEW] Failed to preview preprocessing for dataset {dataset_id}: {str(e)}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to preview preprocessing: {str(e)}")


# ============================================================================
# FILE LINE COUNT ENDPOINT
# ============================================================================

@router.get("/files/{file_id}/line-count", response_model=Dict[str, Any])
async def get_file_line_count(
    file_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get the number of data lines in a file (excluding header)."""
    from sqlalchemy import text
    try:
        # First, check file exists and get metadata without loading content
        file_meta = db.query(File.fileId, File.filename, File.content_type, File.size).filter(
            File.fileId == file_id,
            File.userId == current_user.userId
        ).first()
        
        if not file_meta:
            raise HTTPException(status_code=404, detail="File not found")
        
        filename = file_meta.filename
        content_type = file_meta.content_type
        file_size = file_meta.size
        
        # For large files, estimate line count based on file size
        # Average CSI line is ~500 bytes, IMU line is ~100 bytes
        filename_lower = (filename or "").lower()
        is_csi = 'csi' in filename_lower or (content_type and 'csv' in content_type.lower())
        
        if file_size and file_size > 10_000_000:  # > 10MB, estimate instead
            avg_line_size = 500 if is_csi else 100
            estimated_lines = file_size // avg_line_size
            return {
                "success": True,
                "file_id": file_id,
                "filename": filename,
                "total_lines": estimated_lines,
                "data_lines": estimated_lines - 1 if is_csi else estimated_lines,
                "is_csi": is_csi,
                "estimated": True,
                "note": f"Estimated from file size ({file_size:,} bytes)"
            }
        
        # For smaller files, count lines efficiently using raw SQL to avoid ORM overhead
        try:
            # Use a raw SQL query with timeout to count newlines directly in the database
            result = db.execute(text("""
                SELECT 
                    LENGTH(content) - LENGTH(REPLACE(CONVERT_FROM(content, 'UTF8'), E'\\n', '')) + 1 as line_count
                FROM file 
                WHERE file_id = :file_id AND user_id = :user_id
            """), {"file_id": file_id, "user_id": current_user.userId}).fetchone()
            
            if result and result[0]:
                total_lines = result[0]
                data_lines = total_lines - 1 if is_csi else total_lines  # Subtract header for CSI
                return {
                    "success": True,
                    "file_id": file_id,
                    "filename": filename,
                    "total_lines": total_lines,
                    "data_lines": max(0, data_lines),
                    "is_csi": is_csi
                }
        except Exception as sql_err:
            # If raw SQL fails, fall back to estimation
            if file_size:
                avg_line_size = 500 if is_csi else 100
                estimated_lines = file_size // avg_line_size
                return {
                    "success": True,
                    "file_id": file_id,
                    "filename": filename,
                    "total_lines": estimated_lines,
                    "data_lines": estimated_lines - 1 if is_csi else estimated_lines,
                    "is_csi": is_csi,
                    "estimated": True,
                    "note": f"Estimated (query failed: {str(sql_err)[:50]})"
                }
        
        return {
            "success": True,
            "file_id": file_id,
            "filename": filename,
            "total_lines": 0,
            "data_lines": 0,
            "error": "Could not determine line count"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get line count: {str(e)}")


# ============================================================================
# CLOUD TRAINING ENDPOINTS (must be before /{dataset_id} catch-all)
# ============================================================================

def run_cloud_training(job_id: str, db_url: str):
    """Run real PyTorch training for IMU model in background."""
    import asyncio
    import traceback
    import sys
    
    print(f"[TRAINING-DEBUG] ========================================")
    print(f"[TRAINING-DEBUG] run_cloud_training CALLED for job {job_id}")
    print(f"[TRAINING-DEBUG] db_url: {db_url[:50]}...")
    print(f"[TRAINING-DEBUG] Thread: {__import__('threading').current_thread().name}")
    print(f"[TRAINING-DEBUG] ========================================")
    sys.stdout.flush()
    
    loop = None
    try:
        # Create new event loop for this thread
        print(f"[TRAINING-DEBUG] Creating new event loop...")
        sys.stdout.flush()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        print(f"[TRAINING-DEBUG] Event loop created: {loop}")
        sys.stdout.flush()
        
        print(f"[TRAINING-DEBUG] Starting run_until_complete...")
        sys.stdout.flush()
        loop.run_until_complete(_run_cloud_training_async(job_id, db_url))
        print(f"[TRAINING-DEBUG] run_until_complete finished successfully")
        sys.stdout.flush()
    except Exception as e:
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        print(f"[TRAINING-ERROR] Exception in run_cloud_training: {error_msg}")
        print(f"[TRAINING-ERROR] Traceback: {error_traceback}")
        sys.stdout.flush()
        
        # Mark job as failed in database
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            engine = create_engine(db_url)
            SessionLocal = sessionmaker(bind=engine)
            db = SessionLocal()
            job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
            if job and job.status in ["pending", "running"]:
                job.status = "failed"
                job.error_message = f"Training crashed: {error_msg}"
                job.completed_at = datetime.utcnow()
                db.commit()
                print(f"[TRAINING-DEBUG] Job {job_id} marked as failed in database")
            db.close()
        except Exception as db_error:
            print(f"[TRAINING-ERROR] Failed to update job status in DB: {db_error}")
        sys.stdout.flush()
    finally:
        if loop:
            print(f"[TRAINING-DEBUG] Closing event loop...")
            sys.stdout.flush()
            loop.close()
            print(f"[TRAINING-DEBUG] Event loop closed")
            sys.stdout.flush()


async def _run_cloud_training_async(job_id: str, db_url: str):
    """Async implementation of cloud training."""
    import traceback
    import sys
    import time
    
    # Timing tracking for each pipeline stage
    timing = {
        'total_start': time.time(),
        'preprocessing_start': None,
        'preprocessing_end': None,
        'training_start': None,
        'training_end': None,
        'evaluation_start': None,
        'evaluation_end': None,
        'model_save_start': None,
        'model_save_end': None,
    }
    
    print(f"[TRAINING-DEBUG] _run_cloud_training_async started for job {job_id}")
    sys.stdout.flush()
    
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import sessionmaker
    from server.ml_training import run_full_training
    
    print(f"[TRAINING-DEBUG] Imports successful")
    sys.stdout.flush()
    
    print(f"[TRAINING-DEBUG] Creating database engine with extended timeouts...")
    sys.stdout.flush()
    
    # Create engine with extended pool timeout for large file operations
    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_recycle=3600,
        connect_args={
            "connect_timeout": 300,  # 5 minutes connection timeout
            "options": "-c statement_timeout=300000"  # 5 minutes statement timeout
        }
    )
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    print(f"[TRAINING-DEBUG] Database session created with extended timeouts")
    sys.stdout.flush()
    
    try:
        print(f"[TRAINING-DEBUG] Querying job {job_id}...")
        sys.stdout.flush()
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if not job:
            print(f"[TRAINING-ERROR] Job {job_id} not found in database!")
            sys.stdout.flush()
            return
        
        print(f"[TRAINING-DEBUG] Job found: dataset_id={job.dataset_id}, model_type={job.model_type}")
        sys.stdout.flush()
        
        print(f"[TRAINING-DEBUG] Setting job status to 'running'...")
        sys.stdout.flush()
        job.status = "running"
        job.started_at = datetime.utcnow()
        db.commit()
        print(f"[TRAINING-DEBUG] Job status updated to 'running'")
        sys.stdout.flush()
        
        config = json.loads(job.config) if job.config else {}
        print(f"[TRAINING-DEBUG] Config loaded: {list(config.keys())}")
        sys.stdout.flush()
        
        # Define update callback for progress tracking
        async def update_callback(status, epoch, total, metrics):
            nonlocal job, db
            try:
                stage = metrics.get('stage', '') if metrics else ''
                print(f"[TRAINING-DEBUG] Update callback: status={status}, epoch={epoch}, stage={stage}")
                sys.stdout.flush()
                job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
                if job:
                    job.status = status
                    if epoch > 0:
                        job.current_epoch = epoch
                    # Store stage info in config for ML models
                    if stage and metrics:
                        try:
                            config_data = json.loads(job.config) if job.config else {}
                            config_data['current_stage'] = stage
                            if 'train_accuracy' in metrics:
                                config_data['current_train_accuracy'] = metrics['train_accuracy']
                            job.config = json.dumps(config_data)
                        except:
                            pass
                    db.commit()
            except Exception as e:
                print(f"[TRAINING-WARNING] Failed to update job status: {e}")
                sys.stdout.flush()
        
        # Run real PyTorch training
        print(f"[TRAINING-DEBUG] ========================================")
        print(f"[TRAINING-DEBUG] CALLING run_full_training...")
        print(f"[TRAINING-DEBUG] job_id={job_id}")
        print(f"[TRAINING-DEBUG] dataset_id={job.dataset_id}")
        print(f"[TRAINING-DEBUG] model_type={job.model_type}")
        print(f"[TRAINING-DEBUG] ========================================")
        sys.stdout.flush()
        
        timing['training_start'] = time.time()
        results = await run_full_training(
            job_id=job_id,
            dataset_id=job.dataset_id,
            db_session=db,
            model_type=job.model_type,
            config=config,
            update_callback=update_callback
        )
        timing['training_end'] = time.time()
        
        # Calculate timing durations
        timing['total_end'] = time.time()
        timing_summary = {
            'total_seconds': timing['total_end'] - timing['total_start'],
            'training_seconds': timing['training_end'] - timing['training_start'] if timing['training_start'] else 0,
        }
        # Add timing from results if available (preprocessing, evaluation)
        if results.get('timing'):
            timing_summary.update(results['timing'])
        
        print(f"[TIMING] Total: {timing_summary['total_seconds']:.2f}s, Training: {timing_summary['training_seconds']:.2f}s")
        sys.stdout.flush()
        
        # Store Bayesian trials if available
        if results.get('bayesian_trials_data'):
            config['bayesian_trials_results'] = results['bayesian_trials_data']
        
        # Store timing in config
        config['timing'] = timing_summary
        job.config = json.dumps(config)
        
        # Store model bytes
        model_bytes = results.get('model_bytes')
        class_names = results.get('class_names', [])
        
        job.metrics = json.dumps({
            "loss": results["train_losses"],
            "accuracy": results["train_accuracies"],
            "val_loss": results["val_losses"],
            "val_accuracy": results["val_accuracies"]
        })
        
        best_metrics = {
            "val_accuracy": results["best_val_accuracy"],
            "best_epoch": results["best_epoch"],
            "per_class_metrics": results.get("per_class_metrics", {}),
            "confusion_matrix": results.get("confusion_matrix", []),
            "class_names": results.get("class_names", []),
            "roc_curves": results.get("roc_curves", {}),
            "pr_curves": results.get("pr_curves", {}),
            "model_architecture": results.get("model_architecture", {}),
            "learning_rate": results.get("model_architecture", {}).get("learning_rate", 0.001),
            "batch_size": results.get("model_architecture", {}).get("batch_size", 32),
            "optimizer": results.get("model_architecture", {}).get("optimizer", "adam")
        }
        
        if results.get("test_results"):
            best_metrics["test_accuracy"] = results["test_results"]["test_accuracy"]
            best_metrics["test_loss"] = results["test_results"]["test_loss"]
            best_metrics["test_dataset_id"] = results["test_results"]["test_dataset_id"]
        
        job.best_metrics = json.dumps(best_metrics)
        
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        # For ML models, set current_epoch = total_epochs (1) to show 100% progress
        job.current_epoch = job.total_epochs
        
        # Save trained model (for both mock and real training)
        try:
            from sqlalchemy import text
            try:
                db.execute(text("""
                    CREATE TABLE IF NOT EXISTS trained_model (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        job_id VARCHAR(255),
                        name VARCHAR(255) NOT NULL,
                        architecture VARCHAR(50),
                        accuracy FLOAT,
                        size_bytes BIGINT,
                        model_data BYTEA,
                        config TEXT,
                        is_pinned BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                db.commit()
                print(f"[INFO] Ensured trained_model table exists")
            except Exception as table_error:
                print(f"[WARNING] Could not create trained_model table: {table_error}")
            
            existing_models = db.query(TrainedModel).filter(
                TrainedModel.user_id == job.user_id,
                TrainedModel.is_pinned == False
            ).order_by(TrainedModel.created_at.desc()).all()
            
            if len(existing_models) >= 10:
                models_to_delete = existing_models[9:]
                for old_model in models_to_delete:
                    db.delete(old_model)
                    print(f"[INFO] Deleted old model {old_model.name} to maintain 10-model limit")
            
            # Use custom name from config if provided
            model_name = config.get("model_name") or f"{job.model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Get real model bytes from training
            if not model_bytes:
                raise ValueError("No model weights returned from training")
            
            actual_model_bytes = model_bytes
            actual_size_bytes = len(model_bytes)
            print(f"[INFO] Saving real model weights: {actual_size_bytes} bytes")
            
            # Convert accuracy from decimal (0-1) to percentage (0-100) for storage
            accuracy_pct = float(results["best_val_accuracy"]) * 100
            
            trained_model = TrainedModel(
                user_id=job.user_id,
                job_id=job_id,
                name=model_name,
                architecture=job.model_type,
                accuracy=accuracy_pct,
                size_bytes=actual_size_bytes,
                model_data=actual_model_bytes,
                config=json.dumps({
                    "training_results": {
                        "total_epochs": job.total_epochs,
                        "best_epoch": results["best_epoch"],
                        "final_train_loss": results["train_losses"][-1] if results["train_losses"] else 0,
                        "final_val_loss": results["val_losses"][-1] if results["val_losses"] else 0,
                        "final_train_acc": results["train_accuracies"][-1] if results["train_accuracies"] else 0,
                        "final_val_acc": results["val_accuracies"][-1] if results["val_accuracies"] else 0,
                        "best_val_acc": results["best_val_accuracy"],
                        "test_results": results.get("test_results"),
                        "model_type": job.model_type,
                        "num_train_samples": results.get("num_train_samples"),
                        "num_val_samples": results.get("num_val_samples")
                    }
                })
            )
            db.add(trained_model)
            db.commit()
            print(f"[INFO] Created trained model {model_name} with {accuracy_pct:.2f}% accuracy, size: {actual_size_bytes/1024/1024:.2f} MB")
            sys.stdout.flush()
        except Exception as e:
            print(f"[ERROR] Failed to create trained model: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
        
        db.commit()
        
        print(f"[INFO] Training job {job_id} completed successfully")
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[ERROR] Training job {job_id} failed: {str(e)}\n{error_details}")
        
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()


@router.post("/train/cloud", response_model=Dict[str, Any])
async def start_cloud_training(
    request: CloudTrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Start a cloud training job with a labeled dataset."""
    try:
        dataset = db.query(TrainingDataset).filter(
            TrainingDataset.id == request.dataset_id,
            TrainingDataset.user_id == current_user.userId
        ).first()
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if not dataset.files or len(dataset.files) == 0:
            raise HTTPException(status_code=400, detail="Dataset has no files")
        
        labels = set(f.label for f in dataset.files)
        if len(labels) < 2:
            raise HTTPException(status_code=400, detail="Dataset needs at least 2 different labels for classification")
        
        job_id = str(uuid.uuid4())
        config = {
            "model_type": request.model_type,
            "model_architecture": request.model_architecture,
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "learning_rate": request.learning_rate,
            "validation_split": request.validation_split,
            "test_split": request.test_split,
            "model_name": request.model_name,
            "window_size": request.window_size,
            "num_classes": len(labels),
            "labels": list(labels),
            # Preprocessing pipeline
            "preprocessing_pipeline_id": request.preprocessing_pipeline_id,
            "preprocessing_blocks": request.preprocessing_blocks or [],
            # CSI-specific preprocessing options
            "data_type": request.data_type,
            "include_phase": request.include_phase,
            "filter_subcarriers": request.filter_subcarriers,
            "subcarrier_start": request.subcarrier_start,
            "subcarrier_end": request.subcarrier_end,
            "output_shape": request.output_shape,
            # ML-specific hyperparameters
            "ml_params": request.ml_params or {},
            # Optimization
            "grid_search": request.grid_search or {},
            # Bayesian optimization settings
            "use_bayesian_optimization": request.use_bayesian_optimization,
            "bayesian_trials": request.bayesian_trials,
            "bayesian_epochs_per_trial": request.bayesian_epochs_per_trial,
            "bayesian_lr_min": request.bayesian_lr_min,
            "bayesian_lr_max": request.bayesian_lr_max,
            "bayesian_lr_scale": request.bayesian_lr_scale,
            "bayesian_batch_sizes": request.bayesian_batch_sizes,
            "bayesian_weight_decay_min": request.bayesian_weight_decay_min,
            "bayesian_weight_decay_max": request.bayesian_weight_decay_max,
            "bayesian_optimizers": request.bayesian_optimizers,
            "bayesian_exploration_rate": request.bayesian_exploration_rate,
            "bayesian_search_architecture": request.bayesian_search_architecture
        }
        
        # ML models don't have epochs - set total_epochs=1 for progress tracking
        is_ml_model = request.model_type in ['adaboost', 'knn', 'svc', 'xgboost', 'random_forest']
        total_epochs = 1 if is_ml_model else request.epochs
        
        job = TrainingJob(
            job_id=job_id,
            user_id=current_user.userId,
            dataset_id=request.dataset_id,
            test_dataset_id=request.test_dataset_id,
            preprocessing_pipeline_id=request.preprocessing_pipeline_id,
            model_type=request.model_type,
            training_mode="cloud",
            config=json.dumps(config),
            status="pending",
            total_epochs=total_epochs
        )
        db.add(job)
        db.commit()
        
        import os
        import sys
        db_url = os.environ.get("DATABASE_URL", "postgresql+psycopg2://lms_user:lms_password@localhost:5432/thoth")
        
        print(f"[TRAINING-DEBUG] ========================================")
        print(f"[TRAINING-DEBUG] start_cloud_training endpoint called")
        print(f"[TRAINING-DEBUG] job_id: {job_id}")
        print(f"[TRAINING-DEBUG] dataset_id: {request.dataset_id}")
        print(f"[TRAINING-DEBUG] model_type: {request.model_type}")
        print(f"[TRAINING-DEBUG] use_bayesian_optimization: {request.use_bayesian_optimization}")
        print(f"[TRAINING-DEBUG] db_url: {db_url[:50]}...")
        print(f"[TRAINING-DEBUG] background_tasks object: {background_tasks}")
        print(f"[TRAINING-DEBUG] ========================================")
        sys.stdout.flush()
        
        print(f"[TRAINING-DEBUG] Adding background task...")
        sys.stdout.flush()
        background_tasks.add_task(run_cloud_training, job_id, db_url)
        print(f"[TRAINING-DEBUG] Background task ADDED for job {job_id}")
        print(f"[TRAINING-DEBUG] background_tasks.tasks count: {len(background_tasks.tasks) if hasattr(background_tasks, 'tasks') else 'N/A'}")
        sys.stdout.flush()
        
        return {
            "success": True,
            "message": f"Cloud training job started",
            "job": {
                "job_id": job_id,
                "dataset_id": request.dataset_id,
                "dataset_name": dataset.name,
                "model_type": request.model_type,
                "epochs": request.epochs,
                "num_classes": len(labels),
                "labels": list(labels),
                "status": "pending"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")


@router.get("/train/jobs", response_model=Dict[str, Any])
async def list_training_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum jobs to return"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """List all training jobs for the current user."""
    try:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[JOBS] Starting jobs query for user {current_user.userId}, status={status}, limit={limit}")

        query = db.query(
            TrainingJob.job_id,
            TrainingJob.dataset_id,
            TrainingDataset.name.label("dataset_name"),
            TrainingJob.model_type,
            TrainingJob.training_mode,
            TrainingJob.status,
            TrainingJob.current_epoch,
            TrainingJob.total_epochs,
            TrainingJob.metrics,
            TrainingJob.best_metrics,
            TrainingJob.config,
            TrainingJob.created_at,
            TrainingJob.started_at,
            TrainingJob.completed_at,
            TrainingJob.error_message,
        ).outerjoin(
            TrainingDataset,
            TrainingDataset.id == TrainingJob.dataset_id,
        ).filter(TrainingJob.user_id == current_user.userId)
        
        if status:
            query = query.filter(TrainingJob.status == status)
        
        jobs = query.order_by(TrainingJob.created_at.desc()).limit(limit).all()
        
        logger.info(f"[JOBS] Query completed, returned {len(jobs)} jobs")
        
        # Convert to dict efficiently
        job_list = []
        for j in jobs:
            progress = 0.0
            if j.total_epochs and j.total_epochs > 0:
                progress = (j.current_epoch / j.total_epochs) * 100

            metrics = {}
            if j.metrics:
                try:
                    metrics = json.loads(j.metrics)
                except Exception:
                    metrics = {}

            best_metrics = {}
            if j.best_metrics:
                try:
                    best_metrics = json.loads(j.best_metrics)
                except Exception:
                    best_metrics = {}

            config = {}
            if j.config:
                try:
                    config = json.loads(j.config)
                except Exception:
                    config = {}

            job_list.append({
                "job_id": j.job_id,
                "dataset_id": j.dataset_id,
                "dataset_name": j.dataset_name,
                "model_type": j.model_type,
                "training_mode": j.training_mode,
                "status": j.status,
                "current_epoch": j.current_epoch,
                "total_epochs": j.total_epochs,
                "progress": progress,
                "metrics": metrics,
                "best_metrics": best_metrics,
                "config": config,
                "created_at": j.created_at.isoformat() if j.created_at else None,
                "started_at": j.started_at.isoformat() if j.started_at else None,
                "completed_at": j.completed_at.isoformat() if j.completed_at else None,
                "error_message": j.error_message,
            })
        
        return {
            "success": True,
            "jobs": job_list,
            "total": len(job_list),
            "limit": limit,
            "operation": "list_training_jobs",
            "status": "completed"
        }
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"[JOBS] Error listing jobs: {str(e)}")
        # Return a consistent error response structure
        return {
            "success": False,
            "jobs": [],
            "total": 0,
            "limit": limit,
            "operation": "list_training_jobs",
            "status": "error",
            "error": str(e)
        }


@router.get("/train/jobs/{job_id}", response_model=Dict[str, Any])
async def get_training_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get details of a specific training job."""
    try:
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()

        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")

        if job.user_id != current_user.userId:
            logger.warning(
                "[JOBS] User %s attempted to access job %s owned by user %s",
                current_user.userId,
                job_id,
                job.user_id,
            )
            raise HTTPException(status_code=403, detail="Training job not owned by user")
        
        return {
            "success": True,
            "job": job.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job: {str(e)}")


@router.post("/train/jobs/{job_id}/cancel", response_model=StandardResponse)
async def cancel_training_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Cancel a running training job."""
    try:
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()

        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")

        if job.user_id != current_user.userId:
            logger.warning(
                "[JOBS] User %s attempted to cancel job %s owned by user %s",
                current_user.userId,
                job_id,
                job.user_id,
            )
            raise HTTPException(status_code=403, detail="Training job not owned by user")
        
        if job.status not in ["pending", "running"]:
            raise HTTPException(status_code=400, detail=f"Cannot cancel job with status '{job.status}'")
        
        job.status = "cancelled"
        job.completed_at = datetime.utcnow()
        db.commit()
        
        return StandardResponse(
            success=True,
            message="Training job cancelled"
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")


@router.delete("/train/jobs/{job_id}", response_model=StandardResponse)
async def delete_training_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Delete a specific training job."""
    try:
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()

        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")

        if job.user_id != current_user.userId:
            logger.warning(
                "[JOBS] User %s attempted to delete job %s owned by user %s",
                current_user.userId,
                job_id,
                job.user_id,
            )
            raise HTTPException(status_code=403, detail="Training job not owned by user")
        
        db.delete(job)
        db.commit()
        
        return StandardResponse(
            success=True,
            message="Training job deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete job: {str(e)}")


@router.get("/models", response_model=Dict[str, Any])
async def list_trained_models(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """List all trained models for the current user."""
    try:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[MODELS] Starting models query for user {current_user.userId}")
        
        # Query only necessary columns, exclude model_data to avoid loading large binaries
        logger.info("[MODELS] Executing models query")
        models = db.query(
            TrainedModel.id,
            TrainedModel.job_id,
            TrainedModel.name,
            TrainedModel.architecture,
            TrainedModel.accuracy,
            TrainedModel.size_bytes,
            TrainedModel.config,
            TrainedModel.is_pinned,
            TrainedModel.created_at
        ).filter(
            TrainedModel.user_id == current_user.userId
        ).order_by(TrainedModel.created_at.desc()).limit(50).all()
        
        logger.info(f"[MODELS] Query returned {len(models)} models")
        
        # Convert to dict manually since we're not loading the full model
        model_list = []
        for m in models:
            model_list.append({
                "id": m.id,
                "job_id": m.job_id,
                "name": m.name,
                "architecture": m.architecture,
                "accuracy": round(m.accuracy, 2) if m.accuracy else None,
                "size_mb": m.size_bytes / (1024 * 1024) if m.size_bytes else None,
                "config": json.loads(m.config) if m.config else {},
                "is_pinned": m.is_pinned,
                "created_at": m.created_at.isoformat() if m.created_at else None
            })
        
        logger.info(f"[MODELS] Successfully processed {len(model_list)} models")
        
        return {
            "success": True,
            "models": model_list,
            "total": len(model_list),
            "operation": "list_models",
            "status": "completed"
        }
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"[MODELS] Error listing models: {str(e)}")
        # Return a consistent error response structure
        return {
            "success": False,
            "models": [],
            "total": 0,
            "operation": "list_models",
            "status": "error",
            "error": str(e)
        }


@router.delete("/models/{model_id}", response_model=StandardResponse)
async def delete_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Delete a trained model."""
    try:
        model = db.query(TrainedModel).filter(
            TrainedModel.id == model_id,
            TrainedModel.user_id == current_user.userId
        ).first()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        db.delete(model)
        db.commit()
        
        return StandardResponse(
            success=True,
            message="Model deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


@router.post("/models/{model_id}/pin", response_model=StandardResponse)
async def pin_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Pin a trained model to prevent auto-deletion."""
    try:
        model = db.query(TrainedModel).filter(
            TrainedModel.id == model_id,
            TrainedModel.user_id == current_user.userId
        ).first()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model.is_pinned = True
        db.commit()
        
        return StandardResponse(
            success=True,
            message="Model pinned successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to pin model: {str(e)}")


@router.post("/models/{model_id}/unpin", response_model=StandardResponse)
async def unpin_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Unpin a trained model to allow auto-deletion."""
    try:
        model = db.query(TrainedModel).filter(
            TrainedModel.id == model_id,
            TrainedModel.user_id == current_user.userId
        ).first()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model.is_pinned = False
        db.commit()
        
        return StandardResponse(
            success=True,
            message="Model unpinned successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to unpin model: {str(e)}")


class RenameModelRequest(BaseModel):
    """Request to rename a trained model."""
    name: str


@router.put("/models/{model_id}/rename", response_model=StandardResponse)
async def rename_model(
    model_id: int,
    request: RenameModelRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Rename a trained model."""
    try:
        model = db.query(TrainedModel).filter(
            TrainedModel.id == model_id,
            TrainedModel.user_id == current_user.userId
        ).first()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        if not request.name or len(request.name.strip()) == 0:
            raise HTTPException(status_code=400, detail="Model name cannot be empty")
        
        model.name = request.name.strip()
        db.commit()
        
        return StandardResponse(
            success=True,
            message="Model renamed successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to rename model: {str(e)}")


@router.get("/models/{model_id}/download")
async def download_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Download a trained model."""
    from fastapi.responses import Response
    try:
        model = db.query(TrainedModel).filter(
            TrainedModel.id == model_id,
            TrainedModel.user_id == current_user.userId
        ).first()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # If model_data exists, return it
        if model.model_data:
            return Response(
                content=model.model_data,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename={model.name}.pth"
                }
            )
        else:
            # No model weights available
            raise HTTPException(
                status_code=404, 
                detail=f"Model weights not found for model '{model.name}'. The model may not have been trained yet."
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download model: {str(e)}")


# ============================================================================
# MODEL DEPLOYMENT ENDPOINTS
# ============================================================================

class DeployModelRequest(BaseModel):
    """Request to deploy a trained model to a device."""
    model_id: int
    device_id: str  # device_uuid
    config: Optional[Dict[str, Any]] = None  # trigger config, thresholds, etc.


@router.post("/models/{model_id}/deploy")
async def deploy_model_to_device(
    model_id: int,
    request: DeployModelRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Queue a trained model for deployment to a specific device.

    Uses a pull-based model: the deployment payload is stored in the DB and
    the device picks it up on its next register/heartbeat cycle.  This works
    correctly even when the device is behind NAT or a private network.
    """
    from ..db import Device, DeviceDeployment
    import base64

    try:
        # Validate model
        model = db.query(TrainedModel).filter(
            TrainedModel.id == model_id,
            TrainedModel.user_id == current_user.userId
        ).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        if not model.model_data:
            raise HTTPException(status_code=400, detail="Model has no weights to deploy")

        # Validate device
        device = db.query(Device).filter(
            Device.device_uuid == request.device_id,
            Device.userId == current_user.userId
        ).first()
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")

        # Build deployment config
        deployment_id = str(uuid.uuid4())
        deploy_config = request.config or {}
        deploy_config.update({
            "deployment_id": deployment_id,
            "model_name": model.name,
            "model_type": model.architecture or "unknown",
            "deployed_at": datetime.utcnow().isoformat(),
        })

        # Enrich with training job preprocessing info
        if model.job_id:
            job = db.query(TrainingJob).filter(TrainingJob.job_id == model.job_id).first()
            if job and job.config:
                try:
                    job_config = json.loads(job.config) if isinstance(job.config, str) else job.config
                    deploy_config["preprocessing"] = {
                        "data_type": job_config.get("data_type"),
                        "window_size": job_config.get("window_size"),
                        "output_shape": job_config.get("output_shape"),
                    }
                    deploy_config["class_names"] = job_config.get("class_names", [])
                except (json.JSONDecodeError, TypeError):
                    pass

        # Build full payload (model weights encoded as base64)
        full_payload = {
            "deployment_id": deployment_id,
            "model_name": model.name,
            "model_type": model.architecture or "unknown",
            "model_data": base64.b64encode(model.model_data).decode("utf-8"),
            "config": deploy_config,
        }

        # Store in DB — device will pick it up on next register/heartbeat
        record = DeviceDeployment(
            deployment_id=deployment_id,
            device_uuid=request.device_id,
            model_id=model_id,
            user_id=current_user.userId,
            payload=json.dumps(full_payload),
            status="pending",
        )
        db.add(record)
        db.commit()

        logger.info(f"Deployment {deployment_id} queued for device {request.device_id}")
        return {
            "success": True,
            "deployment_id": deployment_id,
            "model_name": model.name,
            "device_name": device.device_name,
            "device_id": device.device_uuid,
            "message": f"'{model.name}' queued for deployment — device will receive it on next sync"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to queue deployment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to queue deployment: {str(e)}")


@router.get("/models/pending-deployments")
async def list_pending_deployments(
    current_user = Depends(get_current_user)
):
    """Return pending model deployment requests for the current user.
    
    Deployments are pushed directly to devices; no async queue exists yet.
    This endpoint returns an empty list so the portal stops 405-erroring.
    """
    return {"success": True, "deployments": []}


@router.post("/models/deployments/{deployment_id}/confirm")
async def confirm_deployment(
    deployment_id: str,
    body: Dict[str, Any] = Body(default={}),
    current_user = Depends(get_current_user)
):
    """Confirm (accept/decline) a pending model deployment request."""
    accepted = body.get("accepted", True)
    return {
        "success": True,
        "deployment_id": deployment_id,
        "accepted": accepted,
        "message": "Deployment accepted" if accepted else "Deployment declined"
    }


# ============================================================================
# DATASET DETAIL ENDPOINTS (catch-all routes must come last)
# ============================================================================

@router.get("/{dataset_id}", response_model=Dict[str, Any])
async def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get dataset details including all files and labels."""
    try:
        dataset = db.query(TrainingDataset).options(
            selectinload(TrainingDataset.files).selectinload(DatasetFile.file).load_only(
                File.fileId,
                File.filename,
                File.size,
                File.content_type,
                File.uploaded_at,
            )
        ).filter(
            TrainingDataset.id == dataset_id,
            TrainingDataset.user_id == current_user.userId,
        ).first()
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        files_with_details = []
        for df in dataset.files or []:
            try:
                file_info = df.to_dict() if hasattr(df, "to_dict") else {
                    "id": getattr(df, "id", None),
                    "dataset_id": getattr(df, "dataset_id", None),
                    "file_id": getattr(df, "file_id", None),
                    "filename": None,
                    "label": getattr(df, "label", None),
                    "created_at": getattr(df, "created_at", None).isoformat() if getattr(df, "created_at", None) else None,
                }

                if df.file:
                    file_info["filename"] = df.file.filename
                    file_info["size"] = df.file.size
                    file_info["content_type"] = df.file.content_type
                    file_info["file_missing"] = False
                else:
                    file_info["file_missing"] = True

                files_with_details.append(file_info)
            except Exception as file_err:
                files_with_details.append({
                    "id": getattr(df, "id", None),
                    "dataset_id": getattr(df, "dataset_id", None),
                    "file_id": getattr(df, "file_id", None),
                    "filename": None,
                    "label": getattr(df, "label", None),
                    "created_at": getattr(df, "created_at", None).isoformat() if getattr(df, "created_at", None) else None,
                    "file_missing": True,
                    "error": str(file_err),
                })
        
        # Calculate label distribution
        label_counts = {}
        for df in dataset.files or []:
            try:
                if df.label is None:
                    continue
                label_counts[df.label] = label_counts.get(df.label, 0) + 1
            except Exception:
                continue
        
        result = dataset.to_dict()
        result["files"] = files_with_details
        result["label_distribution"] = label_counts
        
        return {
            "success": True,
            "dataset": result
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[DATASET] Failed to get dataset {dataset_id} for user {getattr(current_user, 'userId', None)}: {e}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to get dataset: {str(e)}")


@router.post("/{dataset_id}/files", response_model=Dict[str, Any])
async def add_files_to_dataset(
    dataset_id: int,
    request: AddFilesToDatasetRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Add files with labels to a dataset."""
    import time
    start_time = time.time()
    logger.info(f"[ADD_FILES] Starting to add {len(request.files)} files to dataset {dataset_id} for user {current_user.userId}")
    
    try:
        # Query dataset
        query_start = time.time()
        dataset = db.query(TrainingDataset).filter(
            TrainingDataset.id == dataset_id,
            TrainingDataset.user_id == current_user.userId
        ).first()
        logger.info(f"[ADD_FILES] Dataset query took {(time.time() - query_start)*1000:.2f}ms")
        
        if not dataset:
            logger.warning(f"[ADD_FILES] Dataset {dataset_id} not found for user {current_user.userId}")
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        added_count = 0
        errors = []
        
        # Batch query only file IDs (avoid selecting large columns like file.content)
        file_ids = sorted({f.get("file_id") for f in request.files if f.get("file_id")})
        logger.info(f"[ADD_FILES] Querying {len(file_ids)} files in batch")
        
        query_start = time.time()
        existing_file_ids = set()
        if file_ids:
            rows = db.query(File.fileId).filter(
                File.fileId.in_(file_ids),
                File.userId == current_user.userId
            ).all()
            existing_file_ids = {r[0] for r in rows}
        logger.info(f"[ADD_FILES] Batch file-id query took {(time.time() - query_start)*1000:.2f}ms, found {len(existing_file_ids)} files")
        
        # Query file labels for concatenation
        file_labels_map = {}
        if file_ids:
            file_rows = db.query(File.fileId, File.labels).filter(
                File.fileId.in_(file_ids),
                File.userId == current_user.userId
            ).all()
            for fid, labels_json in file_rows:
                if labels_json:
                    try:
                        file_labels_map[fid] = json.loads(labels_json)
                    except:
                        file_labels_map[fid] = []
                else:
                    file_labels_map[fid] = []
        
        # Process files
        dataset_files_to_add = []
        all_labels_used = set()
        
        for file_entry in request.files:
            file_id = file_entry.get("file_id")
            label = file_entry.get("label")
            
            if not file_id or not label:
                errors.append(f"Missing file_id or label in entry")
                continue
            
            # Check if file exists in our batch query results
            if file_id not in existing_file_ids:
                errors.append(f"File {file_id} not found")
                logger.warning(f"[ADD_FILES] File {file_id} not found for user {current_user.userId}")
                continue
            
            # Get file's existing labels and create concatenated label if multiple
            file_labels = file_labels_map.get(file_id, [])
            final_label = label
            
            # If file has multiple labels, create concatenated label (label1_label2)
            if len(file_labels) > 1:
                # Concatenate all labels with underscore
                concatenated = "_".join(file_labels)
                final_label = concatenated
                all_labels_used.add(concatenated)
            elif len(file_labels) == 1:
                final_label = file_labels[0]
                all_labels_used.add(file_labels[0])
            else:
                all_labels_used.add(label)
            
            # Prepare dataset file entry
            dataset_file = DatasetFile(
                dataset_id=dataset_id,
                file_id=file_id,
                label=final_label
            )
            dataset_files_to_add.append(dataset_file)
            added_count += 1
        
        # Bulk add all dataset files
        if dataset_files_to_add:
            logger.info(f"[ADD_FILES] Adding {len(dataset_files_to_add)} dataset file entries")
            add_start = time.time()
            db.bulk_save_objects(dataset_files_to_add)
            logger.info(f"[ADD_FILES] Bulk add took {(time.time() - add_start)*1000:.2f}ms")
        
        # Commit transaction
        commit_start = time.time()
        db.commit()
        logger.info(f"[ADD_FILES] Commit took {(time.time() - commit_start)*1000:.2f}ms")
        
        total_time = (time.time() - start_time) * 1000
        logger.info(f"[ADD_FILES] Successfully added {added_count} files to dataset {dataset_id} in {total_time:.2f}ms")
        
        return {
            "success": True,
            "message": f"Added {added_count} files to dataset",
            "added_count": added_count,
            "errors": errors if errors else None
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"[ADD_FILES] Failed to add files to dataset {dataset_id}: {str(e)}\n{error_details}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to add files: {str(e)}")


@router.delete("/{dataset_id}/files/{file_id}", response_model=StandardResponse)
async def remove_file_from_dataset(
    dataset_id: int,
    file_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Remove a file from a dataset."""
    try:
        dataset = db.query(TrainingDataset).filter(
            TrainingDataset.id == dataset_id,
            TrainingDataset.user_id == current_user.userId
        ).first()
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset_file = db.query(DatasetFile).filter(
            DatasetFile.dataset_id == dataset_id,
            DatasetFile.file_id == file_id
        ).first()
        
        if not dataset_file:
            raise HTTPException(status_code=404, detail="File not in dataset")
        
        db.delete(dataset_file)
        db.commit()
        
        return StandardResponse(
            success=True,
            message="File removed from dataset"
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to remove file: {str(e)}")


@router.put("/{dataset_id}/files/{file_id}/label", response_model=StandardResponse)
async def update_file_label(
    dataset_id: int,
    file_id: int,
    request: UpdateFileLabelRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Update a file's label in a dataset."""
    try:
        dataset = db.query(TrainingDataset).filter(
            TrainingDataset.id == dataset_id,
            TrainingDataset.user_id == current_user.userId
        ).first()
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset_file = db.query(DatasetFile).filter(
            DatasetFile.dataset_id == dataset_id,
            DatasetFile.file_id == file_id
        ).first()
        
        if not dataset_file:
            raise HTTPException(status_code=404, detail="File not in dataset")
        
        dataset_file.label = request.label
        db.commit()
        
        return StandardResponse(
            success=True,
            message=f"Label updated to '{request.label}'"
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update label: {str(e)}")


@router.delete("/{dataset_id}", response_model=StandardResponse)
async def delete_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Delete a dataset."""
    try:
        dataset = db.query(TrainingDataset).filter(
            TrainingDataset.id == dataset_id,
            TrainingDataset.user_id == current_user.userId
        ).first()
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        db.delete(dataset)
        db.commit()
        
        return StandardResponse(
            success=True,
            message="Dataset deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")
