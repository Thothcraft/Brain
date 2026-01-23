"""Metrics Exporter - Export training metrics in various formats.

Supports export to JSON, CSV, PDF, and LaTeX for publication.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import io

logger = logging.getLogger(__name__)


class MetricsExporter:
    """Export training metrics in various formats.
    
    Supported formats:
    - JSON: Complete metrics data
    - CSV: Tabular epoch/round metrics
    - PDF: Publication-quality report
    - LaTeX: Tables for academic papers
    - HTML: Interactive report
    """
    
    def __init__(self, tracker: 'MetricsTracker', visualizer: 'MetricsVisualizer' = None):
        """Initialize exporter.
        
        Args:
            tracker: MetricsTracker with logged metrics
            visualizer: Optional MetricsVisualizer for plots
        """
        self.tracker = tracker
        self.visualizer = visualizer
    
    def export_json(self, filepath: str) -> str:
        """Export all metrics to JSON file.
        
        Args:
            filepath: Output file path
            
        Returns:
            Path to exported file
        """
        data = self.tracker.to_dict()
        data["exported_at"] = datetime.now().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported metrics to JSON: {filepath}")
        return filepath
    
    def export_csv(self, filepath: str, include_batches: bool = False) -> str:
        """Export epoch metrics to CSV file.
        
        Args:
            filepath: Output file path
            include_batches: Include batch-level metrics
            
        Returns:
            Path to exported file
        """
        import csv
        
        with open(filepath, 'w', newline='') as f:
            if self.tracker.epoch_metrics:
                # Epoch metrics
                fieldnames = [
                    'epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy',
                    'learning_rate', 'epoch_time_seconds', 'train_precision', 'train_recall',
                    'train_f1', 'val_precision', 'val_recall', 'val_f1'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for m in self.tracker.epoch_metrics:
                    writer.writerow(m.to_dict())
            
            elif self.tracker.fl_round_metrics:
                # FL round metrics
                fieldnames = [
                    'round_num', 'global_loss', 'global_accuracy', 'num_clients',
                    'aggregation_time_ms', 'avg_client_loss', 'avg_client_accuracy',
                    'min_client_accuracy', 'max_client_accuracy', 'client_variance'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for m in self.tracker.fl_round_metrics:
                    writer.writerow(m.to_dict())
        
        logger.info(f"Exported metrics to CSV: {filepath}")
        return filepath
    
    def export_latex_tables(self, filepath: str) -> str:
        """Export metrics as LaTeX tables for academic papers.
        
        Args:
            filepath: Output file path
            
        Returns:
            Path to exported file
        """
        latex_content = []
        
        # Header
        latex_content.append("% Training Metrics Tables")
        latex_content.append(f"% Generated: {datetime.now().isoformat()}")
        latex_content.append(f"% Job ID: {self.tracker.job_id}")
        latex_content.append("")
        
        # Summary table
        summary = self.tracker.get_summary()
        latex_content.append("% Summary Table")
        latex_content.append("\\begin{table}[h]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{Training Summary}")
        latex_content.append("\\begin{tabular}{ll}")
        latex_content.append("\\toprule")
        latex_content.append("Metric & Value \\\\")
        latex_content.append("\\midrule")
        latex_content.append(f"Best Validation Accuracy & {summary['best_val_accuracy']:.4f} \\\\")
        latex_content.append(f"Best Epoch & {summary['best_val_epoch']} \\\\")
        latex_content.append(f"Total Epochs & {summary['total_epochs']} \\\\")
        latex_content.append(f"Training Time (s) & {summary['total_training_time_seconds']:.1f} \\\\")
        latex_content.append(f"Device & {summary['device']} \\\\")
        latex_content.append("\\bottomrule")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\label{tab:training_summary}")
        latex_content.append("\\end{table}")
        latex_content.append("")
        
        # Per-class metrics table
        if self.tracker.class_metrics:
            latex_content.append("% Per-Class Metrics Table")
            latex_content.append("\\begin{table}[h]")
            latex_content.append("\\centering")
            latex_content.append("\\caption{Per-Class Performance Metrics}")
            latex_content.append("\\begin{tabular}{lcccc}")
            latex_content.append("\\toprule")
            latex_content.append("Class & Precision & Recall & F1-Score & Support \\\\")
            latex_content.append("\\midrule")
            
            for c in self.tracker.class_metrics:
                latex_content.append(
                    f"{c.class_name} & {c.precision:.4f} & {c.recall:.4f} & {c.f1_score:.4f} & {c.support} \\\\"
                )
            
            latex_content.append("\\bottomrule")
            latex_content.append("\\end{tabular}")
            latex_content.append("\\label{tab:class_metrics}")
            latex_content.append("\\end{table}")
            latex_content.append("")
        
        # Epoch metrics table (first and last 5)
        if self.tracker.epoch_metrics:
            latex_content.append("% Epoch Metrics Table (Selected)")
            latex_content.append("\\begin{table}[h]")
            latex_content.append("\\centering")
            latex_content.append("\\caption{Training Progress (Selected Epochs)}")
            latex_content.append("\\begin{tabular}{ccccc}")
            latex_content.append("\\toprule")
            latex_content.append("Epoch & Train Loss & Train Acc & Val Loss & Val Acc \\\\")
            latex_content.append("\\midrule")
            
            # Select epochs to show
            epochs = self.tracker.epoch_metrics
            if len(epochs) <= 10:
                selected = epochs
            else:
                selected = epochs[:5] + epochs[-5:]
            
            for m in selected:
                val_loss = f"{m.val_loss:.4f}" if m.val_loss is not None else "-"
                val_acc = f"{m.val_accuracy:.4f}" if m.val_accuracy is not None else "-"
                latex_content.append(
                    f"{m.epoch} & {m.train_loss:.4f} & {m.train_accuracy:.4f} & {val_loss} & {val_acc} \\\\"
                )
            
            latex_content.append("\\bottomrule")
            latex_content.append("\\end{tabular}")
            latex_content.append("\\label{tab:epoch_metrics}")
            latex_content.append("\\end{table}")
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(latex_content))
        
        logger.info(f"Exported LaTeX tables: {filepath}")
        return filepath
    
    def export_html_report(self, filepath: str) -> str:
        """Export interactive HTML report.
        
        Args:
            filepath: Output file path
            
        Returns:
            Path to exported file
        """
        summary = self.tracker.get_summary()
        
        # Generate plots if visualizer available
        plots = {}
        if self.visualizer:
            plots = self.visualizer.generate_all_plots()
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Report - {self.tracker.job_id}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #2E86AB, #1a5276); color: white; padding: 40px; border-radius: 10px; margin-bottom: 30px; }}
        .header h1 {{ font-size: 2em; margin-bottom: 10px; }}
        .header p {{ opacity: 0.9; }}
        .card {{ background: white; border-radius: 10px; padding: 25px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .card h2 {{ color: #2E86AB; margin-bottom: 20px; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
        .metric-box {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-box .value {{ font-size: 2em; font-weight: bold; color: #2E86AB; }}
        .metric-box .label {{ color: #666; margin-top: 5px; }}
        .plot-container {{ text-align: center; margin: 20px 0; }}
        .plot-container img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .footer {{ text-align: center; padding: 20px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Training Report</h1>
            <p>Job ID: {self.tracker.job_id} | Mode: {self.tracker.training_mode} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="card">
            <h2>Summary</h2>
            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="value">{summary['best_val_accuracy']:.4f}</div>
                    <div class="label">Best Validation Accuracy</div>
                </div>
                <div class="metric-box">
                    <div class="value">{summary['best_val_epoch']}</div>
                    <div class="label">Best Epoch</div>
                </div>
                <div class="metric-box">
                    <div class="value">{summary['total_epochs']}</div>
                    <div class="label">Total Epochs</div>
                </div>
                <div class="metric-box">
                    <div class="value">{summary['total_training_time_seconds']:.1f}s</div>
                    <div class="label">Training Time</div>
                </div>
                <div class="metric-box">
                    <div class="value">{summary['num_classes']}</div>
                    <div class="label">Classes</div>
                </div>
                <div class="metric-box">
                    <div class="value">{summary['device']}</div>
                    <div class="label">Device</div>
                </div>
            </div>
        </div>
"""
        
        # Add plots
        if plots.get("training_curves"):
            html_content += f"""
        <div class="card">
            <h2>Training Curves</h2>
            <div class="plot-container">
                <img src="data:image/png;base64,{plots['training_curves']}" alt="Training Curves">
            </div>
        </div>
"""
        
        if plots.get("confusion_matrix"):
            html_content += f"""
        <div class="card">
            <h2>Confusion Matrix</h2>
            <div class="plot-container">
                <img src="data:image/png;base64,{plots['confusion_matrix']}" alt="Confusion Matrix">
            </div>
        </div>
"""
        
        if plots.get("class_performance"):
            html_content += f"""
        <div class="card">
            <h2>Per-Class Performance</h2>
            <div class="plot-container">
                <img src="data:image/png;base64,{plots['class_performance']}" alt="Class Performance">
            </div>
        </div>
"""
        
        if plots.get("roc_curves"):
            html_content += f"""
        <div class="card">
            <h2>ROC Curves</h2>
            <div class="plot-container">
                <img src="data:image/png;base64,{plots['roc_curves']}" alt="ROC Curves">
            </div>
        </div>
"""
        
        # Class metrics table
        if self.tracker.class_metrics:
            html_content += """
        <div class="card">
            <h2>Per-Class Metrics</h2>
            <table>
                <thead>
                    <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
                </thead>
                <tbody>
"""
            for c in self.tracker.class_metrics:
                html_content += f"""
                    <tr><td>{c.class_name}</td><td>{c.precision:.4f}</td><td>{c.recall:.4f}</td><td>{c.f1_score:.4f}</td><td>{c.support}</td></tr>
"""
            html_content += """
                </tbody>
            </table>
        </div>
"""
        
        html_content += """
        <div class="footer">
            <p>Generated by Thoth ML Platform</p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Exported HTML report: {filepath}")
        return filepath
    
    def export_all(self, output_dir: str, base_name: str = None) -> Dict[str, str]:
        """Export metrics in all formats.
        
        Args:
            output_dir: Output directory
            base_name: Base filename (default: job_id)
            
        Returns:
            Dictionary mapping format to file path
        """
        os.makedirs(output_dir, exist_ok=True)
        base_name = base_name or self.tracker.job_id
        
        exports = {}
        
        # JSON
        json_path = os.path.join(output_dir, f"{base_name}_metrics.json")
        exports["json"] = self.export_json(json_path)
        
        # CSV
        csv_path = os.path.join(output_dir, f"{base_name}_metrics.csv")
        exports["csv"] = self.export_csv(csv_path)
        
        # LaTeX
        latex_path = os.path.join(output_dir, f"{base_name}_tables.tex")
        exports["latex"] = self.export_latex_tables(latex_path)
        
        # HTML
        html_path = os.path.join(output_dir, f"{base_name}_report.html")
        exports["html"] = self.export_html_report(html_path)
        
        logger.info(f"Exported all formats to: {output_dir}")
        return exports
