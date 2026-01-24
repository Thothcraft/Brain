"""API endpoints for the plotting system.

Provides REST API for:
- Generating plots with customization
- Exporting plots in multiple formats
- Getting LaTeX code for figures
- Listing available plot types
"""

import io
import base64
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/plotting", tags=["plotting"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class PlotConfigRequest(BaseModel):
    """Plot configuration from frontend."""
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    figsize: List[float] = Field(default=[8, 6])
    dpi: int = 150
    grid: bool = True
    legend: bool = True
    legendLoc: str = "best"
    lineWidth: float = 2.0
    markerSize: float = 6
    alpha: float = 0.8
    showCI: bool = True
    ciLevel: float = 0.95
    annotateValues: bool = False
    annotateBest: bool = True


class ExportConfigRequest(BaseModel):
    """Export configuration from frontend."""
    format: str = "pdf"
    dpi: int = 300
    transparent: bool = False
    width: str = "single"
    customWidth: Optional[float] = None


class GeneratePlotRequest(BaseModel):
    """Request to generate a plot."""
    job_id: Optional[str] = None
    plot_type: str
    theme: str = "neurips"
    config: Optional[PlotConfigRequest] = None
    data: Optional[Dict[str, Any]] = None


class ExportPlotRequest(BaseModel):
    """Request to export a plot."""
    job_id: Optional[str] = None
    plot_type: str
    theme: str = "neurips"
    config: Optional[PlotConfigRequest] = None
    export_config: Optional[ExportConfigRequest] = None
    data: Optional[Dict[str, Any]] = None


class LaTeXCodeRequest(BaseModel):
    """Request for LaTeX code."""
    plot_type: str
    theme: str = "neurips"
    width: str = "single"
    caption: Optional[str] = None
    label: Optional[str] = None


class PlotTypeInfo(BaseModel):
    """Information about a plot type."""
    id: str
    name: str
    description: str
    category: str
    requires_data: List[str]


class PlotCategoryInfo(BaseModel):
    """Information about a plot category."""
    id: str
    name: str
    plots: List[PlotTypeInfo]


# ============================================================================
# PLOT TYPE DEFINITIONS
# ============================================================================

PLOT_TYPES = {
    # Performance
    "learning_curves": PlotTypeInfo(
        id="learning_curves", name="Learning Curves",
        description="Loss and accuracy over epochs",
        category="performance", requires_data=["train_loss", "val_loss"]
    ),
    "metrics_bar": PlotTypeInfo(
        id="metrics_bar", name="Metrics Bar Chart",
        description="Final metrics comparison",
        category="performance", requires_data=["accuracy", "precision", "recall", "f1"]
    ),
    "confusion_matrix": PlotTypeInfo(
        id="confusion_matrix", name="Confusion Matrix",
        description="Classification confusion matrix",
        category="performance", requires_data=["confusion_matrix"]
    ),
    "roc_curves": PlotTypeInfo(
        id="roc_curves", name="ROC Curves",
        description="Receiver Operating Characteristic",
        category="performance", requires_data=["roc_data"]
    ),
    "pr_curves": PlotTypeInfo(
        id="pr_curves", name="PR Curves",
        description="Precision-Recall curves",
        category="performance", requires_data=["pr_data"]
    ),
    "class_metrics": PlotTypeInfo(
        id="class_metrics", name="Per-Class Metrics",
        description="Metrics breakdown by class",
        category="performance", requires_data=["class_metrics"]
    ),
    # Resources
    "resource_timeline": PlotTypeInfo(
        id="resource_timeline", name="Resource Timeline",
        description="CPU, GPU, Memory over time",
        category="resources", requires_data=["cpu_percent", "memory_percent"]
    ),
    "cpu_usage": PlotTypeInfo(
        id="cpu_usage", name="CPU Usage",
        description="CPU utilization by core",
        category="resources", requires_data=["cpu_per_core"]
    ),
    "gpu_usage": PlotTypeInfo(
        id="gpu_usage", name="GPU Usage",
        description="GPU utilization and memory",
        category="resources", requires_data=["gpu_data"]
    ),
    "memory_usage": PlotTypeInfo(
        id="memory_usage", name="Memory Usage",
        description="Memory breakdown and timeline",
        category="resources", requires_data=["memory_data"]
    ),
    # Timing
    "timing_breakdown": PlotTypeInfo(
        id="timing_breakdown", name="Timing Breakdown",
        description="Time distribution by component",
        category="timing", requires_data=["timing_components"]
    ),
    "epoch_timing": PlotTypeInfo(
        id="epoch_timing", name="Epoch Timing",
        description="Per-epoch time analysis",
        category="timing", requires_data=["epoch_times"]
    ),
    "throughput": PlotTypeInfo(
        id="throughput", name="Throughput",
        description="Samples/batches per second",
        category="timing", requires_data=["throughput_data"]
    ),
    # Model
    "parameter_distribution": PlotTypeInfo(
        id="parameter_distribution", name="Parameter Distribution",
        description="Parameters by layer",
        category="model", requires_data=["layer_info"]
    ),
    "layer_complexity": PlotTypeInfo(
        id="layer_complexity", name="Layer Complexity",
        description="FLOPs and memory by layer",
        category="model", requires_data=["layer_complexity"]
    ),
    "gradient_norm": PlotTypeInfo(
        id="gradient_norm", name="Gradient Norms",
        description="Gradient magnitude over training",
        category="model", requires_data=["grad_norms"]
    ),
    "gradient_flow": PlotTypeInfo(
        id="gradient_flow", name="Gradient Flow",
        description="Gradient flow through layers",
        category="model", requires_data=["layer_gradients"]
    ),
    # Data
    "class_distribution": PlotTypeInfo(
        id="class_distribution", name="Class Distribution",
        description="Sample count per class",
        category="data", requires_data=["class_counts"]
    ),
    "feature_correlation": PlotTypeInfo(
        id="feature_correlation", name="Feature Correlation",
        description="Feature correlation matrix",
        category="data", requires_data=["correlation_matrix"]
    ),
    "embedding_2d": PlotTypeInfo(
        id="embedding_2d", name="2D Embedding",
        description="t-SNE/UMAP/PCA visualization",
        category="data", requires_data=["embeddings"]
    ),
    # Comparison
    "trial_comparison": PlotTypeInfo(
        id="trial_comparison", name="Trial Comparison",
        description="Compare multiple trials",
        category="comparison", requires_data=["trials"]
    ),
    "model_comparison": PlotTypeInfo(
        id="model_comparison", name="Model Comparison",
        description="Compare different models",
        category="comparison", requires_data=["models"]
    ),
    "ablation_study": PlotTypeInfo(
        id="ablation_study", name="Ablation Study",
        description="Component contribution analysis",
        category="comparison", requires_data=["ablation_data"]
    ),
    "pareto_front": PlotTypeInfo(
        id="pareto_front", name="Pareto Front",
        description="Multi-objective optimization",
        category="comparison", requires_data=["pareto_points"]
    ),
    # Calibration
    "reliability_diagram": PlotTypeInfo(
        id="reliability_diagram", name="Reliability Diagram",
        description="Model calibration analysis",
        category="calibration", requires_data=["calibration_data"]
    ),
}

PLOT_CATEGORIES = [
    PlotCategoryInfo(id="performance", name="Performance", plots=[
        PLOT_TYPES[k] for k in ["learning_curves", "metrics_bar", "confusion_matrix", "roc_curves", "pr_curves", "class_metrics"]
    ]),
    PlotCategoryInfo(id="resources", name="Resources", plots=[
        PLOT_TYPES[k] for k in ["resource_timeline", "cpu_usage", "gpu_usage", "memory_usage"]
    ]),
    PlotCategoryInfo(id="timing", name="Timing", plots=[
        PLOT_TYPES[k] for k in ["timing_breakdown", "epoch_timing", "throughput"]
    ]),
    PlotCategoryInfo(id="model", name="Model Analysis", plots=[
        PLOT_TYPES[k] for k in ["parameter_distribution", "layer_complexity", "gradient_norm", "gradient_flow"]
    ]),
    PlotCategoryInfo(id="data", name="Data Analysis", plots=[
        PLOT_TYPES[k] for k in ["class_distribution", "feature_correlation", "embedding_2d"]
    ]),
    PlotCategoryInfo(id="comparison", name="Comparison", plots=[
        PLOT_TYPES[k] for k in ["trial_comparison", "model_comparison", "ablation_study", "pareto_front"]
    ]),
    PlotCategoryInfo(id="calibration", name="Calibration", plots=[
        PLOT_TYPES[k] for k in ["reliability_diagram"]
    ]),
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_plotter(plot_type: str, theme: str):
    """Get appropriate plotter for plot type."""
    try:
        from server.plotting import create_plotter, AdvancedPlotter
        from server.plotting.factory import PlotterFactory
        
        # Determine plotter type based on plot category
        plot_info = PLOT_TYPES.get(plot_type)
        if not plot_info:
            return AdvancedPlotter(theme=theme)
        
        category = plot_info.category
        if category in ["performance"]:
            return create_plotter("dl", theme=theme)
        elif category in ["resources", "timing", "model", "data", "calibration"]:
            return AdvancedPlotter(theme=theme)
        elif category in ["comparison"]:
            return create_plotter("comparison", theme=theme)
        else:
            return AdvancedPlotter(theme=theme)
    except ImportError as e:
        logger.error(f"Failed to import plotting module: {e}")
        raise HTTPException(status_code=500, detail="Plotting module not available")


def convert_config(config: PlotConfigRequest) -> dict:
    """Convert frontend config to backend PlotConfig."""
    return {
        "title": config.title,
        "xlabel": config.xlabel,
        "ylabel": config.ylabel,
        "figsize": tuple(config.figsize),
        "dpi": config.dpi,
        "grid": config.grid,
        "legend": config.legend,
        "legend_loc": config.legendLoc,
        "line_width": config.lineWidth,
        "marker_size": config.markerSize,
        "alpha": config.alpha,
        "show_ci": config.showCI,
        "ci_level": config.ciLevel,
        "annotate_values": config.annotateValues,
        "annotate_best": config.annotateBest,
    }


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.get("/types")
async def get_plot_types():
    """Get all available plot types."""
    return {
        "categories": [cat.dict() for cat in PLOT_CATEGORIES],
        "types": {k: v.dict() for k, v in PLOT_TYPES.items()},
    }


@router.get("/themes")
async def get_themes():
    """Get available themes."""
    return {
        "themes": [
            {"id": "default", "name": "Default", "description": "Standard styling"},
            {"id": "neurips", "name": "NeurIPS", "description": "NeurIPS conference style"},
            {"id": "icml", "name": "ICML", "description": "ICML conference style"},
            {"id": "cvpr", "name": "CVPR", "description": "CVPR conference style"},
            {"id": "ieee", "name": "IEEE", "description": "IEEE journal style"},
            {"id": "nature", "name": "Nature", "description": "Nature journal style"},
            {"id": "presentation", "name": "Presentation", "description": "Large fonts for slides"},
            {"id": "poster", "name": "Poster", "description": "Extra large for posters"},
        ]
    }


@router.post("/generate")
async def generate_plot(request: GeneratePlotRequest):
    """Generate a plot and return as base64."""
    try:
        plotter = get_plotter(request.plot_type, request.theme)
        
        # Prepare data
        data = request.data or {}
        
        # Generate plot
        config = None
        if request.config:
            from server.plotting.base import PlotConfig, PlotType
            config = PlotConfig(
                plot_type=PlotType.MODEL_COMPARISON,
                **convert_config(request.config)
            )
        
        # Map plot type to plotter method
        fig, base64_str = plotter.create_plot(request.plot_type, data, config)
        
        return {
            "success": True,
            "plot_type": request.plot_type,
            "base64": base64_str,
            "theme": request.theme,
        }
        
    except Exception as e:
        logger.error(f"Error generating plot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export")
async def export_plot(request: ExportPlotRequest):
    """Export a plot to file format."""
    try:
        from fastapi.responses import StreamingResponse
        
        plotter = get_plotter(request.plot_type, request.theme)
        
        # Prepare data
        data = request.data or {}
        
        # Generate plot
        config = None
        if request.config:
            from server.plotting.base import PlotConfig, PlotType
            config = PlotConfig(
                plot_type=PlotType.MODEL_COMPARISON,
                **convert_config(request.config)
            )
        
        fig, _ = plotter.create_plot(request.plot_type, data, config)
        
        # Export to requested format
        export_config = request.export_config or ExportConfigRequest()
        
        buf = io.BytesIO()
        
        # Determine width
        if export_config.width == "single":
            width = 3.5
        elif export_config.width == "double":
            width = 7.16
        else:
            width = export_config.customWidth or 6
        
        # Resize figure
        height = fig.get_figheight() * (width / fig.get_figwidth())
        fig.set_size_inches(width, height)
        
        # Save to buffer
        fig.savefig(
            buf,
            format=export_config.format,
            dpi=export_config.dpi,
            transparent=export_config.transparent,
            bbox_inches='tight',
            pad_inches=0.05,
        )
        buf.seek(0)
        
        # Determine content type
        content_types = {
            "png": "image/png",
            "pdf": "application/pdf",
            "svg": "image/svg+xml",
            "eps": "application/postscript",
        }
        content_type = content_types.get(export_config.format, "application/octet-stream")
        
        return StreamingResponse(
            buf,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename={request.plot_type}.{export_config.format}"
            }
        )
        
    except Exception as e:
        logger.error(f"Error exporting plot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/latex-code")
async def get_latex_code(request: LaTeXCodeRequest):
    """Get LaTeX code for including a figure."""
    try:
        from server.plotting.export import LaTeXHelper
        
        # Determine width
        if request.width == "single":
            width = "0.48\\textwidth"
        elif request.width == "double":
            width = "\\textwidth"
        else:
            width = "0.8\\textwidth"
        
        # Generate LaTeX code
        figure_path = f"figures/{request.plot_type}"
        caption = request.caption or f"{request.plot_type.replace('_', ' ').title()}"
        label = request.label or request.plot_type.replace('_', '-')
        
        latex = LaTeXHelper.generate_figure_code(
            figure_path=figure_path,
            caption=caption,
            label=label,
            width=width,
        )
        
        return {
            "latex": latex,
            "preamble": LaTeXHelper.generate_preamble(),
        }
        
    except Exception as e:
        logger.error(f"Error generating LaTeX: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-export")
async def batch_export(
    job_id: str,
    plot_types: List[str],
    theme: str = "neurips",
    format: str = "pdf",
    width: str = "single",
):
    """Export multiple plots at once."""
    try:
        results = {}
        
        for plot_type in plot_types:
            try:
                plotter = get_plotter(plot_type, theme)
                fig, base64_str = plotter.create_plot(plot_type, {})
                results[plot_type] = {
                    "success": True,
                    "base64": base64_str,
                }
            except Exception as e:
                results[plot_type] = {
                    "success": False,
                    "error": str(e),
                }
        
        return {
            "job_id": job_id,
            "results": results,
        }
        
    except Exception as e:
        logger.error(f"Error in batch export: {e}")
        raise HTTPException(status_code=500, detail=str(e))
