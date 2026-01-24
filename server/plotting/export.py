"""Export system for publication-quality figure output.

Supports multiple formats for different use cases:
- PNG/JPEG: Web and presentations
- PDF: LaTeX documents and print
- EPS: Legacy LaTeX and journals
- SVG: Scalable web graphics
- TikZ/PGFPlots: Native LaTeX figures
"""

import os
import io
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from .base import ExportConfig, ExportFormat

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Figure = Any


class ExportManager:
    """Manages figure export to multiple formats."""
    
    @classmethod
    def export_figure(cls, fig: Figure, name: str, 
                     config: ExportConfig) -> Dict[str, str]:
        """Export a figure to multiple formats.
        
        Args:
            fig: Matplotlib figure to export
            name: Base filename (without extension)
            config: Export configuration
        
        Returns:
            Dictionary mapping format to file path
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib not available for export")
            return {}
        
        # Create output directory
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build filename
        filename = name
        if config.prefix:
            filename = f"{config.prefix}_{filename}"
        if config.suffix:
            filename = f"{filename}_{config.suffix}"
        if config.timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{ts}"
        
        saved_files = {}
        
        for fmt in config.formats:
            if isinstance(fmt, str):
                fmt = ExportFormat(fmt)
            
            filepath = output_dir / f"{filename}.{fmt.value}"
            
            try:
                if fmt == ExportFormat.TIKZ:
                    TikZExporter.export(fig, str(filepath), config)
                elif fmt == ExportFormat.PGFPLOTS:
                    TikZExporter.export(fig, str(filepath), config, pgfplots=True)
                else:
                    cls._export_matplotlib(fig, str(filepath), fmt, config)
                
                saved_files[fmt.value] = str(filepath)
                logger.info(f"Exported {fmt.value}: {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to export {fmt.value}: {e}")
        
        return saved_files
    
    @classmethod
    def _export_matplotlib(cls, fig: Figure, filepath: str,
                          fmt: ExportFormat, config: ExportConfig):
        """Export using matplotlib's savefig."""
        save_kwargs = {
            'dpi': config.dpi,
            'bbox_inches': config.bbox_inches,
            'pad_inches': config.pad_inches,
            'transparent': config.transparent,
        }
        
        # Format-specific options
        if fmt in [ExportFormat.PDF, ExportFormat.EPS]:
            if config.embed_fonts:
                save_kwargs['metadata'] = {'Creator': 'Thoth Plotting Library'}
        
        fig.savefig(filepath, format=fmt.value, **save_kwargs)
    
    @classmethod
    def export_all(cls, figures: Dict[str, Figure],
                  config: ExportConfig) -> Dict[str, Dict[str, str]]:
        """Export multiple figures.
        
        Args:
            figures: Dictionary mapping names to figures
            config: Export configuration
        
        Returns:
            Nested dictionary: {figure_name: {format: filepath}}
        """
        results = {}
        for name, fig in figures.items():
            results[name] = cls.export_figure(fig, name, config)
        return results


class TikZExporter:
    """Export matplotlib figures to TikZ/PGFPlots format.
    
    Generates native LaTeX code that can be compiled directly,
    ensuring perfect font matching and scalability.
    """
    
    @classmethod
    def export(cls, fig: Figure, filepath: str, config: ExportConfig,
              pgfplots: bool = False):
        """Export figure to TikZ format.
        
        Args:
            fig: Matplotlib figure
            filepath: Output file path
            config: Export configuration
            pgfplots: Use PGFPlots format instead of basic TikZ
        """
        try:
            # Try to use tikzplotlib if available
            import tikzplotlib
            
            tikz_code = tikzplotlib.get_tikz_code(
                fig,
                standalone=config.tikz_standalone,
                extra_axis_parameters=[
                    f"width={config.tikz_width}",
                    "height=0.75\\textwidth",
                ],
            )
            
            with open(filepath, 'w') as f:
                f.write(tikz_code)
                
        except ImportError:
            # Fallback: generate basic TikZ manually
            logger.warning("tikzplotlib not available, using basic TikZ export")
            tikz_code = cls._generate_basic_tikz(fig, config, pgfplots)
            
            with open(filepath, 'w') as f:
                f.write(tikz_code)
    
    @classmethod
    def _generate_basic_tikz(cls, fig: Figure, config: ExportConfig,
                            pgfplots: bool = False) -> str:
        """Generate basic TikZ code from figure."""
        lines = []
        
        if config.tikz_standalone:
            lines.append("\\documentclass{standalone}")
            lines.append("\\usepackage{tikz}")
            if pgfplots:
                lines.append("\\usepackage{pgfplots}")
                lines.append("\\pgfplotsset{compat=1.18}")
            lines.append("\\begin{document}")
        
        lines.append("\\begin{tikzpicture}")
        
        if pgfplots:
            lines.append("\\begin{axis}[")
            lines.append(f"    width={config.tikz_width},")
            lines.append("    height=0.75\\textwidth,")
            lines.append("    grid=major,")
            lines.append("    grid style={dashed, gray!30},")
            lines.append("]")
            
            # Extract data from figure axes
            for ax in fig.axes:
                for line in ax.get_lines():
                    xdata = line.get_xdata()
                    ydata = line.get_ydata()
                    label = line.get_label()
                    color = line.get_color()
                    
                    if len(xdata) > 0 and not label.startswith('_'):
                        lines.append(f"\\addplot[color={cls._mpl_to_tikz_color(color)}, thick] coordinates {{")
                        for x, y in zip(xdata, ydata):
                            lines.append(f"    ({x}, {y})")
                        lines.append("};")
                        if label:
                            lines.append(f"\\addlegendentry{{{label}}}")
            
            lines.append("\\end{axis}")
        else:
            lines.append("% Basic TikZ - consider using pgfplots for better results")
            lines.append("\\draw[help lines] (0,0) grid (10,10);")
        
        lines.append("\\end{tikzpicture}")
        
        if config.tikz_standalone:
            lines.append("\\end{document}")
        
        return "\n".join(lines)
    
    @staticmethod
    def _mpl_to_tikz_color(color: str) -> str:
        """Convert matplotlib color to TikZ color."""
        color_map = {
            '#0072B2': 'blue',
            '#D55E00': 'orange',
            '#009E73': 'green!50!black',
            '#CC79A7': 'magenta',
            '#F0E442': 'yellow',
            '#56B4E9': 'cyan',
            '#E69F00': 'orange!80!red',
            'b': 'blue',
            'g': 'green',
            'r': 'red',
            'c': 'cyan',
            'm': 'magenta',
            'y': 'yellow',
            'k': 'black',
            'w': 'white',
        }
        return color_map.get(color, 'black')


class HTMLExporter:
    """Export figures to interactive HTML."""
    
    @classmethod
    def export(cls, figures: Dict[str, str], title: str = "Training Report",
              output_path: str = None) -> str:
        """Export figures to standalone HTML.
        
        Args:
            figures: Dictionary mapping names to base64 PNG strings
            title: HTML page title
            output_path: Optional file path to save HTML
        
        Returns:
            HTML string
        """
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #0072B2;
            padding-bottom: 10px;
        }}
        .figure {{
            margin: 30px 0;
            text-align: center;
        }}
        .figure img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .figure-title {{
            font-size: 14px;
            color: #666;
            margin-top: 10px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 12px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="grid">
"""
        
        for name, b64 in figures.items():
            display_name = name.replace('_', ' ').title()
            html += f"""
            <div class="figure">
                <img src="data:image/png;base64,{b64}" alt="{display_name}">
                <div class="figure-title">{display_name}</div>
            </div>
"""
        
        html += f"""
        </div>
        <div class="footer">
            Generated by Thoth Plotting Library | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html)
        
        return html


class LaTeXHelper:
    """Helper for generating LaTeX code to include figures."""
    
    @classmethod
    def generate_figure_code(cls, figure_path: str, caption: str = "",
                            label: str = "", width: str = "\\textwidth",
                            position: str = "htbp") -> str:
        """Generate LaTeX code for including a figure.
        
        Args:
            figure_path: Path to figure file
            caption: Figure caption
            label: Figure label for referencing
            width: Figure width
            position: Float position specifier
        
        Returns:
            LaTeX code string
        """
        if not label:
            label = Path(figure_path).stem.replace('_', '-')
        
        return f"""\\begin{{figure}}[{position}]
    \\centering
    \\includegraphics[width={width}]{{{figure_path}}}
    \\caption{{{caption}}}
    \\label{{fig:{label}}}
\\end{{figure}}
"""
    
    @classmethod
    def generate_subfigures(cls, figures: List[Dict[str, str]],
                           caption: str = "", label: str = "",
                           columns: int = 2) -> str:
        """Generate LaTeX code for subfigures.
        
        Args:
            figures: List of dicts with 'path', 'caption', 'label' keys
            caption: Main figure caption
            label: Main figure label
            columns: Number of columns
        
        Returns:
            LaTeX code string
        """
        width = f"{0.95/columns:.2f}\\textwidth"
        
        lines = ["\\begin{figure}[htbp]", "    \\centering"]
        
        for i, fig in enumerate(figures):
            lines.append(f"    \\begin{{subfigure}}[b]{{{width}}}")
            lines.append(f"        \\centering")
            lines.append(f"        \\includegraphics[width=\\textwidth]{{{fig['path']}}}")
            lines.append(f"        \\caption{{{fig.get('caption', '')}}}")
            lines.append(f"        \\label{{fig:{fig.get('label', f'sub{i+1}')}}}")
            lines.append(f"    \\end{{subfigure}}")
            
            if (i + 1) % columns == 0 and i < len(figures) - 1:
                lines.append("    \\\\")
            else:
                lines.append("    \\hfill")
        
        lines.append(f"    \\caption{{{caption}}}")
        lines.append(f"    \\label{{fig:{label}}}")
        lines.append("\\end{figure}")
        
        return "\n".join(lines)
    
    @classmethod
    def generate_preamble(cls) -> str:
        """Generate recommended LaTeX preamble for figures."""
        return """% Recommended preamble for Thoth figures
\\usepackage{graphicx}
\\usepackage{subcaption}
\\usepackage{booktabs}
\\usepackage{xcolor}

% For TikZ/PGFPlots figures
\\usepackage{tikz}
\\usepackage{pgfplots}
\\pgfplotsset{compat=1.18}

% Color definitions matching Thoth palette
\\definecolor{thothblue}{HTML}{0072B2}
\\definecolor{thothorange}{HTML}{D55E00}
\\definecolor{thothgreen}{HTML}{009E73}
\\definecolor{thothpurple}{HTML}{CC79A7}
"""
