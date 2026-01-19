"""Figure Export Endpoints.

Provides endpoints for generating and exporting publication-ready figures:
- Training curves (loss, accuracy)
- Confusion matrix
- ROC curves
- Precision-Recall curves
- Class distribution charts
- Cross-validation box plots
- Model comparison charts

Supports export formats:
- PDF (vector, preferred for publications)
- PNG (high-DPI raster)
- LaTeX tables (IEEE format)
"""

from fastapi import APIRouter, HTTPException, Query, Depends, Response
from fastapi.responses import StreamingResponse
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import json
import io
import logging
import base64

from ..db import get_db, TrainingJob, TrainedModel
from ..auth import get_current_user
from ..figure_export import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_pr_curves,
    plot_class_distribution,
    plot_cv_boxplot,
    plot_model_comparison,
    generate_latex_table,
    generate_confusion_matrix_latex,
    export_figure,
    figure_to_base64,
    export_all_training_figures,
)

router = APIRouter(prefix="/figures", tags=["figures"])
logger = logging.getLogger(__name__)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class FigureExportRequest(BaseModel):
    """Request to export a figure."""
    job_id: str
    figure_type: str = Field(..., description="Type: training_curves, confusion_matrix, roc_curves, pr_curves")
    format: str = Field(default="pdf", description="Export format: pdf, png, svg, eps")
    width: str = Field(default="single", description="Column width: single, double")
    dpi: int = Field(default=300, description="DPI for raster formats")


class FigurePreviewResponse(BaseModel):
    """Response with base64-encoded figure preview."""
    figure_type: str
    format: str
    data_base64: str
    width_inches: float
    height_inches: float


class AllFiguresExportRequest(BaseModel):
    """Request to export all figures for a training job."""
    job_id: str
    formats: List[str] = Field(default=["pdf", "png"], description="Export formats")
    width: str = Field(default="single", description="Column width: single, double")


class LatexExportRequest(BaseModel):
    """Request to export LaTeX tables."""
    job_id: str
    table_type: str = Field(..., description="Type: results, confusion_matrix, comparison")
    caption: Optional[str] = None
    label: Optional[str] = None


class ModelComparisonRequest(BaseModel):
    """Request to compare multiple models."""
    job_ids: List[str]
    metrics: List[str] = Field(default=["accuracy", "precision", "recall", "f1"])
    format: str = Field(default="pdf")
    width: str = Field(default="double")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_training_results(job_id: str, db: Session, user_id: int) -> Dict[str, Any]:
    """Get training results for a job."""
    job = db.query(TrainingJob).filter(
        TrainingJob.job_id == job_id,
        TrainingJob.user_id == user_id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Training job is not completed (status: {job.status})")
    
    # Parse stored metrics
    results = {}
    
    if job.metrics:
        metrics = json.loads(job.metrics)
        results['train_losses'] = metrics.get('loss', [])
        results['val_losses'] = metrics.get('val_loss', [])
        results['train_accuracies'] = metrics.get('accuracy', [])
        results['val_accuracies'] = metrics.get('val_accuracy', [])
    
    if job.best_metrics:
        best = json.loads(job.best_metrics)
        results['best_epoch'] = best.get('best_epoch', 1)
        results['best_val_accuracy'] = best.get('val_accuracy', 0)
        results['confusion_matrix'] = best.get('confusion_matrix', [])
        results['per_class_metrics'] = best.get('per_class_metrics', {})
        results['roc_curves'] = best.get('roc_curves', {})
        results['pr_curves'] = best.get('pr_curves', {})
        results['class_names'] = best.get('class_names', [])
    
    if job.config:
        config = json.loads(job.config)
        results['model_type'] = job.model_type
        results['config'] = config
    
    return results


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/preview/{figure_type}")
async def get_figure_preview(
    figure_type: str,
    request: FigureExportRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
) -> FigurePreviewResponse:
    """Get a base64-encoded preview of a figure for display in the UI."""
    import matplotlib.pyplot as plt
    
    results = get_training_results(request.job_id, db, current_user.userId)
    
    fig = None
    try:
        if figure_type == "training_curves":
            if not all(k in results for k in ['train_losses', 'val_losses', 'train_accuracies', 'val_accuracies']):
                raise HTTPException(status_code=400, detail="Training metrics not available")
            fig = plot_training_curves(
                results['train_losses'],
                results['val_losses'],
                results['train_accuracies'],
                results['val_accuracies'],
                width=request.width,
                best_epoch=results.get('best_epoch')
            )
        
        elif figure_type == "confusion_matrix":
            if 'confusion_matrix' not in results or 'class_names' not in results:
                raise HTTPException(status_code=400, detail="Confusion matrix not available")
            fig = plot_confusion_matrix(
                results['confusion_matrix'],
                results['class_names'],
                width=request.width
            )
        
        elif figure_type == "roc_curves":
            if 'roc_curves' not in results:
                raise HTTPException(status_code=400, detail="ROC curves not available")
            fig = plot_roc_curves(results['roc_curves'], width=request.width)
        
        elif figure_type == "pr_curves":
            if 'pr_curves' not in results:
                raise HTTPException(status_code=400, detail="PR curves not available")
            fig = plot_pr_curves(results['pr_curves'], width=request.width)
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown figure type: {figure_type}")
        
        # Get figure dimensions
        width_inches, height_inches = fig.get_size_inches()
        
        # Export to base64
        data_base64 = figure_to_base64(fig, format='png', dpi=150)  # Lower DPI for preview
        
        return FigurePreviewResponse(
            figure_type=figure_type,
            format='png',
            data_base64=data_base64,
            width_inches=width_inches,
            height_inches=height_inches
        )
    
    finally:
        if fig:
            plt.close(fig)


@router.post("/export/{figure_type}")
async def export_figure_endpoint(
    figure_type: str,
    request: FigureExportRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Export a figure in the requested format (PDF, PNG, SVG, EPS)."""
    import matplotlib.pyplot as plt
    
    results = get_training_results(request.job_id, db, current_user.userId)
    
    fig = None
    try:
        if figure_type == "training_curves":
            fig = plot_training_curves(
                results['train_losses'],
                results['val_losses'],
                results['train_accuracies'],
                results['val_accuracies'],
                width=request.width,
                best_epoch=results.get('best_epoch')
            )
        
        elif figure_type == "confusion_matrix":
            fig = plot_confusion_matrix(
                results['confusion_matrix'],
                results['class_names'],
                width=request.width
            )
        
        elif figure_type == "roc_curves":
            fig = plot_roc_curves(results['roc_curves'], width=request.width)
        
        elif figure_type == "pr_curves":
            fig = plot_pr_curves(results['pr_curves'], width=request.width)
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown figure type: {figure_type}")
        
        # Export to bytes
        data = export_figure(fig, format=request.format, dpi=request.dpi)
        
        # Determine content type
        content_types = {
            'pdf': 'application/pdf',
            'png': 'image/png',
            'svg': 'image/svg+xml',
            'eps': 'application/postscript'
        }
        content_type = content_types.get(request.format, 'application/octet-stream')
        
        # Return as downloadable file
        filename = f"{figure_type}_{request.job_id[:8]}.{request.format}"
        
        return Response(
            content=data,
            media_type=content_type,
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"'
            }
        )
    
    finally:
        if fig:
            plt.close(fig)


@router.post("/export-all")
async def export_all_figures(
    request: AllFiguresExportRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """Export all available figures for a training job.
    
    Returns a dictionary with base64-encoded figures in each requested format.
    """
    import matplotlib.pyplot as plt
    
    results = get_training_results(request.job_id, db, current_user.userId)
    
    exported = {}
    
    # Training curves
    if all(k in results for k in ['train_losses', 'val_losses', 'train_accuracies', 'val_accuracies']):
        fig = plot_training_curves(
            results['train_losses'],
            results['val_losses'],
            results['train_accuracies'],
            results['val_accuracies'],
            width=request.width,
            best_epoch=results.get('best_epoch')
        )
        exported['training_curves'] = {
            fmt: base64.b64encode(export_figure(fig, format=fmt)).decode('utf-8')
            for fmt in request.formats
        }
        plt.close(fig)
    
    # Confusion matrix
    if 'confusion_matrix' in results and 'class_names' in results:
        fig = plot_confusion_matrix(
            results['confusion_matrix'],
            results['class_names'],
            width=request.width
        )
        exported['confusion_matrix'] = {
            fmt: base64.b64encode(export_figure(fig, format=fmt)).decode('utf-8')
            for fmt in request.formats
        }
        # Add LaTeX
        exported['confusion_matrix']['latex'] = generate_confusion_matrix_latex(
            results['confusion_matrix'],
            results['class_names']
        )
        plt.close(fig)
    
    # ROC curves
    if 'roc_curves' in results:
        fig = plot_roc_curves(results['roc_curves'], width=request.width)
        exported['roc_curves'] = {
            fmt: base64.b64encode(export_figure(fig, format=fmt)).decode('utf-8')
            for fmt in request.formats
        }
        plt.close(fig)
    
    # PR curves
    if 'pr_curves' in results:
        fig = plot_pr_curves(results['pr_curves'], width=request.width)
        exported['pr_curves'] = {
            fmt: base64.b64encode(export_figure(fig, format=fmt)).decode('utf-8')
            for fmt in request.formats
        }
        plt.close(fig)
    
    return {
        'job_id': request.job_id,
        'formats': request.formats,
        'width': request.width,
        'figures': exported
    }


@router.post("/latex/results-table")
async def export_latex_results_table(
    request: LatexExportRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
) -> Dict[str, str]:
    """Export results as a LaTeX table (IEEE format)."""
    results = get_training_results(request.job_id, db, current_user.userId)
    
    if request.table_type == "confusion_matrix":
        if 'confusion_matrix' not in results or 'class_names' not in results:
            raise HTTPException(status_code=400, detail="Confusion matrix not available")
        
        latex = generate_confusion_matrix_latex(
            results['confusion_matrix'],
            results['class_names'],
            caption=request.caption or "Confusion Matrix",
            label=request.label or "tab:confusion"
        )
    
    elif request.table_type == "results":
        if 'per_class_metrics' not in results:
            raise HTTPException(status_code=400, detail="Per-class metrics not available")
        
        # Format per-class metrics for table
        data = {}
        for class_name, metrics in results['per_class_metrics'].items():
            data[class_name] = {
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1': metrics.get('f1', 0),
                'support': metrics.get('support', 0)
            }
        
        latex = generate_latex_table(
            data,
            metrics=['precision', 'recall', 'f1'],
            caption=request.caption or "Per-Class Performance Metrics",
            label=request.label or "tab:per_class",
            show_std=False
        )
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown table type: {request.table_type}")
    
    return {
        'table_type': request.table_type,
        'latex': latex
    }


@router.post("/compare-models")
async def compare_models_figure(
    request: ModelComparisonRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Generate a comparison figure for multiple trained models."""
    import matplotlib.pyplot as plt
    
    if len(request.job_ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 models to compare")
    
    # Collect results from all jobs
    comparison_data = {}
    
    for job_id in request.job_ids:
        try:
            results = get_training_results(job_id, db, current_user.userId)
            model_name = results.get('model_type', job_id[:8])
            
            # Get overall metrics
            if 'per_class_metrics' in results:
                # Calculate macro averages
                metrics_list = list(results['per_class_metrics'].values())
                if metrics_list:
                    comparison_data[model_name] = {
                        'accuracy': results.get('best_val_accuracy', 0),
                        'precision': sum(m.get('precision', 0) for m in metrics_list) / len(metrics_list),
                        'recall': sum(m.get('recall', 0) for m in metrics_list) / len(metrics_list),
                        'f1': sum(m.get('f1', 0) for m in metrics_list) / len(metrics_list),
                    }
        except Exception as e:
            logger.warning(f"Failed to get results for job {job_id}: {e}")
            continue
    
    if len(comparison_data) < 2:
        raise HTTPException(status_code=400, detail="Could not retrieve results for enough models")
    
    # Generate comparison figure
    fig = plot_model_comparison(
        comparison_data,
        metrics=request.metrics,
        width=request.width,
        show_error_bars=False
    )
    
    try:
        data = export_figure(fig, format=request.format)
        
        content_types = {
            'pdf': 'application/pdf',
            'png': 'image/png',
            'svg': 'image/svg+xml',
        }
        content_type = content_types.get(request.format, 'application/octet-stream')
        
        filename = f"model_comparison.{request.format}"
        
        return Response(
            content=data,
            media_type=content_type,
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"'
            }
        )
    finally:
        plt.close(fig)


@router.post("/latex/model-comparison")
async def export_latex_model_comparison(
    request: ModelComparisonRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
) -> Dict[str, str]:
    """Export model comparison as a LaTeX table."""
    if len(request.job_ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 models to compare")
    
    comparison_data = {}
    
    for job_id in request.job_ids:
        try:
            results = get_training_results(job_id, db, current_user.userId)
            model_name = results.get('model_type', job_id[:8])
            
            if 'per_class_metrics' in results:
                metrics_list = list(results['per_class_metrics'].values())
                if metrics_list:
                    comparison_data[model_name] = {
                        'accuracy': results.get('best_val_accuracy', 0),
                        'precision': sum(m.get('precision', 0) for m in metrics_list) / len(metrics_list),
                        'recall': sum(m.get('recall', 0) for m in metrics_list) / len(metrics_list),
                        'f1': sum(m.get('f1', 0) for m in metrics_list) / len(metrics_list),
                    }
        except Exception as e:
            logger.warning(f"Failed to get results for job {job_id}: {e}")
            continue
    
    if len(comparison_data) < 2:
        raise HTTPException(status_code=400, detail="Could not retrieve results for enough models")
    
    latex = generate_latex_table(
        comparison_data,
        metrics=request.metrics,
        caption="Model Performance Comparison",
        label="tab:comparison",
        highlight_best=True,
        show_std=False
    )
    
    return {
        'latex': latex,
        'models': list(comparison_data.keys())
    }


@router.get("/available/{job_id}")
async def get_available_figures(
    job_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get list of available figures for a training job."""
    results = get_training_results(job_id, db, current_user.userId)
    
    available = []
    
    if all(k in results for k in ['train_losses', 'val_losses', 'train_accuracies', 'val_accuracies']):
        available.append({
            'type': 'training_curves',
            'name': 'Training Curves',
            'description': 'Loss and accuracy over epochs'
        })
    
    if 'confusion_matrix' in results and 'class_names' in results:
        available.append({
            'type': 'confusion_matrix',
            'name': 'Confusion Matrix',
            'description': 'True vs predicted labels heatmap',
            'has_latex': True
        })
    
    if 'roc_curves' in results:
        available.append({
            'type': 'roc_curves',
            'name': 'ROC Curves',
            'description': 'Receiver Operating Characteristic curves per class'
        })
    
    if 'pr_curves' in results:
        available.append({
            'type': 'pr_curves',
            'name': 'Precision-Recall Curves',
            'description': 'Precision vs Recall curves per class'
        })
    
    return {
        'job_id': job_id,
        'model_type': results.get('model_type', 'unknown'),
        'available_figures': available,
        'export_formats': ['pdf', 'png', 'svg', 'eps'],
        'column_widths': ['single', 'double']
    }
