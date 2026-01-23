"""Report Endpoints for Training Reports and Shareable Views.

This module provides API endpoints for:
- Viewing training reports
- Generating shareable read-only links
- Downloading report data and plots
- Public view mode for reproducible research
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from server.db import get_db, User, TrainingJob
from server.auth import get_current_user
from server.training_report import (
    TrainingReport,
    ReportGenerator,
    ShareableReportManager,
    ReportPlotter,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reports", tags=["reports"])

# Initialize managers
report_manager = ShareableReportManager()
report_generator = ReportGenerator()


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class CreateReportRequest(BaseModel):
    job_id: str
    include_plots: bool = True


class ShareReportRequest(BaseModel):
    report_id: str
    is_public: bool = True


class ReportSummary(BaseModel):
    report_id: str
    job_id: str
    model_type: str
    training_mode: str
    best_accuracy: float
    created_at: str
    share_token: Optional[str] = None
    share_url: Optional[str] = None


# ============================================================================
# AUTHENTICATED ENDPOINTS
# ============================================================================

@router.get("/list", response_model=Dict[str, Any])
async def list_reports(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all training reports for the current user."""
    try:
        reports = report_manager.list_user_reports(current_user.userId)
        
        # Apply pagination
        total = len(reports)
        reports = reports[offset:offset + limit]
        
        # Add share URLs
        for report in reports:
            if report.get("share_token"):
                report["share_url"] = f"/reports/view/{report['share_token']}"
        
        return {
            "success": True,
            "reports": reports,
            "total": total,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total,
            }
        }
    except Exception as e:
        logger.error(f"Error listing reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{report_id}", response_model=Dict[str, Any])
async def get_report(
    report_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get a specific training report by ID."""
    try:
        report = report_manager.get_report_by_id(report_id)
        
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        # Check ownership
        if report.user_id != current_user.userId:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return {
            "success": True,
            "report": report.to_dict(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{report_id}/download")
async def download_report(
    report_id: str,
    format: str = Query("json", regex="^(json|html)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Download report in JSON or HTML format."""
    try:
        report = report_manager.get_report_by_id(report_id)
        
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        if report.user_id != current_user.userId:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if format == "json":
            content = report_generator.export_to_json(report)
            return Response(
                content=content,
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=report_{report_id}.json"
                }
            )
        else:
            content = report_generator.export_to_html(report)
            return Response(
                content=content,
                media_type="text/html",
                headers={
                    "Content-Disposition": f"attachment; filename=report_{report_id}.html"
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/share", response_model=Dict[str, Any])
async def share_report(
    request: ShareReportRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate or update shareable link for a report."""
    try:
        report = report_manager.get_report_by_id(request.report_id)
        
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        if report.user_id != current_user.userId:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Update public status
        report.is_public = request.is_public
        
        # Save updated report
        share_url = report_manager.save_report(report)
        
        return {
            "success": True,
            "share_token": report.share_token,
            "share_url": share_url,
            "is_public": report.is_public,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sharing report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{report_id}/plot/{plot_name}")
async def get_plot(
    report_id: str,
    plot_name: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get a specific plot from a report as PNG image."""
    try:
        report = report_manager.get_report_by_id(report_id)
        
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        if report.user_id != current_user.userId:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if plot_name not in report.plots:
            raise HTTPException(status_code=404, detail=f"Plot '{plot_name}' not found")
        
        import base64
        plot_data = base64.b64decode(report.plots[plot_name])
        
        return Response(
            content=plot_data,
            media_type="image/png",
            headers={
                "Content-Disposition": f"inline; filename={plot_name}.png"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting plot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PUBLIC VIEW ENDPOINTS (No authentication required)
# ============================================================================

@router.get("/view/{share_token}", response_class=HTMLResponse)
async def view_shared_report(
    share_token: str,
    db: Session = Depends(get_db),
):
    """View a shared report in read-only mode (no authentication required)."""
    try:
        report = report_manager.get_report_by_token(share_token)
        
        if not report:
            return HTMLResponse(
                content="""
                <html>
                <head><title>Report Not Found</title></head>
                <body style="font-family: sans-serif; text-align: center; padding: 50px;">
                    <h1>Report Not Found</h1>
                    <p>This report may have been deleted or the link is invalid.</p>
                </body>
                </html>
                """,
                status_code=404
            )
        
        if not report.is_public:
            return HTMLResponse(
                content="""
                <html>
                <head><title>Access Denied</title></head>
                <body style="font-family: sans-serif; text-align: center; padding: 50px;">
                    <h1>Access Denied</h1>
                    <p>This report is not publicly shared.</p>
                </body>
                </html>
                """,
                status_code=403
            )
        
        # Generate read-only HTML view
        html = _generate_readonly_view(report)
        return HTMLResponse(content=html)
        
    except Exception as e:
        logger.error(f"Error viewing shared report: {e}")
        return HTMLResponse(
            content=f"""
            <html>
            <head><title>Error</title></head>
            <body style="font-family: sans-serif; text-align: center; padding: 50px;">
                <h1>Error</h1>
                <p>An error occurred while loading the report.</p>
            </body>
            </html>
            """,
            status_code=500
        )


@router.get("/view/{share_token}/data", response_model=Dict[str, Any])
async def get_shared_report_data(
    share_token: str,
    db: Session = Depends(get_db),
):
    """Get shared report data as JSON (for downloading in view mode)."""
    try:
        report = report_manager.get_report_by_token(share_token)
        
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        if not report.is_public:
            raise HTTPException(status_code=403, detail="Report is not public")
        
        # Return report data without plots (for smaller response)
        data = report.to_dict()
        data.pop("plots", None)  # Remove plots from JSON response
        
        return {
            "success": True,
            "report": data,
            "download_available": True,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting shared report data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/view/{share_token}/download")
async def download_shared_report(
    share_token: str,
    format: str = Query("json", regex="^(json|html|csv)$"),
    db: Session = Depends(get_db),
):
    """Download shared report data (available in view mode)."""
    try:
        report = report_manager.get_report_by_token(share_token)
        
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        if not report.is_public:
            raise HTTPException(status_code=403, detail="Report is not public")
        
        if format == "json":
            content = report_generator.export_to_json(report)
            return Response(
                content=content,
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=report_{report.job_id}.json"
                }
            )
        elif format == "html":
            content = report_generator.export_to_html(report)
            return Response(
                content=content,
                media_type="text/html",
                headers={
                    "Content-Disposition": f"attachment; filename=report_{report.job_id}.html"
                }
            )
        elif format == "csv":
            # Export epoch metrics as CSV
            content = _export_metrics_csv(report)
            return Response(
                content=content,
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=metrics_{report.job_id}.csv"
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading shared report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _generate_readonly_view(report: TrainingReport) -> str:
    """Generate read-only HTML view with hidden edit buttons."""
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Report - {report.job_id} (View Only)</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            margin: 0; 
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .header {{
            background: rgba(255,255,255,0.95);
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 20px;
            color: #333;
        }}
        .view-badge {{
            background: #28a745;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }}
        .container {{ 
            max-width: 1200px; 
            margin: 20px auto; 
            padding: 0 20px;
        }}
        .card {{
            background: white; 
            padding: 25px; 
            border-radius: 12px; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            margin-bottom: 20px;
        }}
        h2 {{ 
            color: #333; 
            margin-top: 0;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .metrics-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); 
            gap: 15px; 
            margin: 20px 0; 
        }}
        .metric-card {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px; 
            border-radius: 10px; 
            text-align: center;
            color: white;
        }}
        .metric-value {{ 
            font-size: 28px; 
            font-weight: bold; 
        }}
        .metric-label {{ 
            font-size: 12px; 
            opacity: 0.9;
            margin-top: 5px; 
        }}
        .plot {{ 
            margin: 20px 0; 
            text-align: center; 
        }}
        .plot img {{ 
            max-width: 100%; 
            height: auto; 
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin: 20px 0; 
        }}
        th, td {{ 
            padding: 12px; 
            text-align: left; 
            border-bottom: 1px solid #eee; 
        }}
        th {{ 
            background: #f8f9fa; 
            font-weight: 600;
        }}
        .download-section {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }}
        .download-btn {{
            display: inline-block;
            padding: 10px 20px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 6px;
            margin-right: 10px;
            font-size: 14px;
        }}
        .download-btn:hover {{
            background: #5a6fd6;
        }}
        .footer {{ 
            text-align: center;
            padding: 20px; 
            color: rgba(255,255,255,0.8);
            font-size: 12px; 
        }}
        .info-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 15px;
        }}
        .info-item {{
            flex: 1;
            min-width: 200px;
        }}
        .info-label {{
            font-size: 12px;
            color: #666;
            margin-bottom: 3px;
        }}
        .info-value {{
            font-weight: 600;
            color: #333;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Training Report</h1>
        <span class="view-badge">📖 VIEW ONLY</span>
    </div>
    
    <div class="container">
        <div class="card">
            <h2>Overview</h2>
            <div class="info-row">
                <div class="info-item">
                    <div class="info-label">Job ID</div>
                    <div class="info-value">{report.job_id}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Model</div>
                    <div class="info-value">{report.model_type}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Training Mode</div>
                    <div class="info-value">{report.training_mode.upper()}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Dataset</div>
                    <div class="info-value">{report.dataset_name}</div>
                </div>
            </div>
            <div class="info-row">
                <div class="info-item">
                    <div class="info-label">Classes</div>
                    <div class="info-value">{report.num_classes} ({', '.join(report.class_names[:5])}{'...' if len(report.class_names) > 5 else ''})</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Created</div>
                    <div class="info-value">{report.created_at}</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Performance Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{report.best_val_accuracy:.2f}%</div>
                    <div class="metric-label">Best Validation Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{report.final_train_accuracy:.2f}%</div>
                    <div class="metric-label">Final Train Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{report.epochs}</div>
                    <div class="metric-label">Epochs</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{report.total_time_seconds:.1f}s</div>
                    <div class="metric-label">Total Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{report.train_samples}</div>
                    <div class="metric-label">Training Samples</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{report.device}</div>
                    <div class="metric-label">Device</div>
                </div>
            </div>
        </div>
"""
    
    # Add plots
    for plot_name, plot_data in report.plots.items():
        title = plot_name.replace('_', ' ').title()
        html += f"""
        <div class="card">
            <h2>{title}</h2>
            <div class="plot">
                <img src="data:image/png;base64,{plot_data}" alt="{title}">
            </div>
        </div>
"""
    
    # Add class metrics table
    if report.class_metrics:
        html += """
        <div class="card">
            <h2>Per-Class Metrics</h2>
            <table>
                <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
"""
        for m in report.class_metrics:
            html += f"                <tr><td>{m.class_name}</td><td>{m.precision:.3f}</td><td>{m.recall:.3f}</td><td>{m.f1_score:.3f}</td><td>{m.support}</td></tr>\n"
        html += """
            </table>
        </div>
"""
    
    # Download section
    html += f"""
        <div class="card">
            <h2>Download Data</h2>
            <div class="download-section">
                <p>Download this report for reproducible research:</p>
                <a href="/reports/view/{report.share_token}/download?format=json" class="download-btn">📄 Download JSON</a>
                <a href="/reports/view/{report.share_token}/download?format=html" class="download-btn">🌐 Download HTML</a>
                <a href="/reports/view/{report.share_token}/download?format=csv" class="download-btn">📊 Download Metrics CSV</a>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>Report ID: {report.report_id}</p>
        <p>Generated by Thoth ML Platform • Verifiable & Reproducible ML Research</p>
    </div>
</body>
</html>
"""
    return html


def _export_metrics_csv(report: TrainingReport) -> str:
    """Export epoch metrics as CSV."""
    lines = ["epoch,train_loss,train_accuracy,val_loss,val_accuracy,learning_rate"]
    
    for m in report.epoch_metrics:
        val_loss = m.val_loss if m.val_loss is not None else ""
        val_acc = m.val_accuracy if m.val_accuracy is not None else ""
        lines.append(f"{m.epoch},{m.train_loss},{m.train_accuracy},{val_loss},{val_acc},{m.learning_rate}")
    
    return "\n".join(lines)
