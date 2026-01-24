"""Theme system for publication-quality plots.

Provides pre-configured themes for major ML/AI publication venues:
- NeurIPS, ICML, ICLR (ML conferences)
- CVPR, ICCV, ECCV (Computer vision)
- ACL, EMNLP (NLP)
- JMLR, TMLR (Journals)
- IEEE (Transactions)

Each theme follows the specific formatting guidelines of the venue.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


class VenueType(str, Enum):
    """Publication venue types."""
    NEURIPS = "neurips"
    ICML = "icml"
    ICLR = "iclr"
    CVPR = "cvpr"
    ICCV = "iccv"
    ACL = "acl"
    JMLR = "jmlr"
    IEEE = "ieee"
    NATURE = "nature"
    DEFAULT = "default"
    PRESENTATION = "presentation"
    POSTER = "poster"


@dataclass
class ColorPalette:
    """Color palette for plots."""
    name: str
    primary: List[str]
    secondary: List[str]
    diverging: List[str]
    sequential: List[str]
    
    # Semantic colors
    positive: str = "#2ecc71"  # Green
    negative: str = "#e74c3c"  # Red
    neutral: str = "#95a5a6"   # Gray
    highlight: str = "#f39c12"  # Orange


# Colorblind-friendly palettes
COLORBLIND_PALETTE = ColorPalette(
    name="colorblind",
    primary=[
        '#0072B2',  # Blue
        '#D55E00',  # Vermillion
        '#009E73',  # Bluish green
        '#CC79A7',  # Reddish purple
        '#F0E442',  # Yellow
        '#56B4E9',  # Sky blue
        '#E69F00',  # Orange
        '#000000',  # Black
    ],
    secondary=[
        '#4477AA',  # Blue
        '#EE6677',  # Red
        '#228833',  # Green
        '#CCBB44',  # Yellow
        '#66CCEE',  # Cyan
        '#AA3377',  # Purple
        '#BBBBBB',  # Gray
    ],
    diverging=[
        '#313695', '#4575b4', '#74add1', '#abd9e9',
        '#e0f3f8', '#ffffbf', '#fee090', '#fdae61',
        '#f46d43', '#d73027', '#a50026'
    ],
    sequential=[
        '#f7fbff', '#deebf7', '#c6dbef', '#9ecae1',
        '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'
    ]
)

VIBRANT_PALETTE = ColorPalette(
    name="vibrant",
    primary=[
        '#EE7733',  # Orange
        '#0077BB',  # Blue
        '#33BBEE',  # Cyan
        '#EE3377',  # Magenta
        '#CC3311',  # Red
        '#009988',  # Teal
        '#BBBBBB',  # Gray
    ],
    secondary=[
        '#332288',  # Indigo
        '#88CCEE',  # Cyan
        '#44AA99',  # Teal
        '#117733',  # Green
        '#999933',  # Olive
        '#DDCC77',  # Sand
        '#CC6677',  # Rose
        '#882255',  # Wine
        '#AA4499',  # Purple
    ],
    diverging=COLORBLIND_PALETTE.diverging,
    sequential=COLORBLIND_PALETTE.sequential
)


@dataclass
class PublicationTheme:
    """Complete theme configuration for a publication venue."""
    name: str
    venue: VenueType
    
    # Figure dimensions (in inches)
    figure_width: float = 5.5
    figure_height: float = 4.0
    column_width: float = 3.25  # Single column
    text_width: float = 6.875   # Full text width
    
    # Font settings
    font_family: str = "serif"
    font_size: int = 10
    title_size: int = 11
    label_size: int = 10
    tick_size: int = 9
    legend_size: int = 9
    
    # Line settings
    line_width: float = 1.5
    marker_size: float = 5
    
    # Axes settings
    axes_linewidth: float = 0.8
    grid_alpha: float = 0.3
    grid_linewidth: float = 0.5
    
    # Spine settings
    spine_top: bool = False
    spine_right: bool = False
    
    # Color palette
    palette: ColorPalette = field(default_factory=lambda: COLORBLIND_PALETTE)
    
    # LaTeX settings
    use_latex: bool = False
    latex_preamble: str = ""
    
    # DPI settings
    display_dpi: int = 150
    export_dpi: int = 300
    
    def to_rcparams(self) -> Dict[str, Any]:
        """Convert theme to matplotlib rcParams."""
        params = {
            # Figure
            'figure.figsize': (self.figure_width, self.figure_height),
            'figure.dpi': self.display_dpi,
            'savefig.dpi': self.export_dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
            
            # Font
            'font.family': self.font_family,
            'font.size': self.font_size,
            'axes.titlesize': self.title_size,
            'axes.labelsize': self.label_size,
            'xtick.labelsize': self.tick_size,
            'ytick.labelsize': self.tick_size,
            'legend.fontsize': self.legend_size,
            
            # Lines
            'lines.linewidth': self.line_width,
            'lines.markersize': self.marker_size,
            
            # Axes
            'axes.linewidth': self.axes_linewidth,
            'axes.grid': True,
            'grid.alpha': self.grid_alpha,
            'grid.linewidth': self.grid_linewidth,
            'axes.spines.top': self.spine_top,
            'axes.spines.right': self.spine_right,
            
            # Colors
            'axes.prop_cycle': plt.cycler('color', self.palette.primary) if MATPLOTLIB_AVAILABLE else None,
        }
        
        if self.use_latex:
            params.update({
                'text.usetex': True,
                'font.family': 'serif',
                'text.latex.preamble': self.latex_preamble,
            })
        
        return {k: v for k, v in params.items() if v is not None}


# Pre-defined themes for major venues
THEMES: Dict[str, PublicationTheme] = {
    'default': PublicationTheme(
        name="Default",
        venue=VenueType.DEFAULT,
        figure_width=8,
        figure_height=6,
        font_family="sans-serif",
        font_size=12,
        title_size=14,
        label_size=12,
        tick_size=10,
        legend_size=10,
        line_width=2.0,
        marker_size=6,
    ),
    
    'neurips': PublicationTheme(
        name="NeurIPS",
        venue=VenueType.NEURIPS,
        figure_width=5.5,
        figure_height=4.0,
        column_width=5.5,
        text_width=5.5,
        font_family="serif",
        font_size=10,
        title_size=11,
        label_size=10,
        tick_size=9,
        legend_size=9,
        line_width=1.5,
        marker_size=5,
        use_latex=False,  # Set True if LaTeX available
    ),
    
    'icml': PublicationTheme(
        name="ICML",
        venue=VenueType.ICML,
        figure_width=6.0,
        figure_height=4.0,
        column_width=3.25,
        text_width=6.875,
        font_family="serif",
        font_size=10,
        title_size=11,
        label_size=10,
        tick_size=9,
        legend_size=9,
        line_width=1.5,
        marker_size=5,
    ),
    
    'iclr': PublicationTheme(
        name="ICLR",
        venue=VenueType.ICLR,
        figure_width=5.5,
        figure_height=4.0,
        font_family="serif",
        font_size=10,
        title_size=11,
        label_size=10,
        tick_size=9,
        legend_size=9,
        line_width=1.5,
        marker_size=5,
    ),
    
    'cvpr': PublicationTheme(
        name="CVPR",
        venue=VenueType.CVPR,
        figure_width=6.5,
        figure_height=4.5,
        column_width=3.25,
        text_width=6.875,
        font_family="serif",
        font_size=11,
        title_size=12,
        label_size=11,
        tick_size=10,
        legend_size=10,
        line_width=2.0,
        marker_size=6,
    ),
    
    'ieee': PublicationTheme(
        name="IEEE",
        venue=VenueType.IEEE,
        figure_width=3.5,
        figure_height=2.5,
        column_width=3.5,
        text_width=7.16,
        font_family="serif",
        font_size=8,
        title_size=9,
        label_size=8,
        tick_size=7,
        legend_size=7,
        line_width=1.0,
        marker_size=4,
        axes_linewidth=0.5,
    ),
    
    'jmlr': PublicationTheme(
        name="JMLR",
        venue=VenueType.JMLR,
        figure_width=6.0,
        figure_height=4.5,
        font_family="serif",
        font_size=11,
        title_size=12,
        label_size=11,
        tick_size=10,
        legend_size=10,
        line_width=1.5,
        marker_size=5,
    ),
    
    'nature': PublicationTheme(
        name="Nature",
        venue=VenueType.NATURE,
        figure_width=3.5,
        figure_height=3.0,
        column_width=3.5,
        text_width=7.2,
        font_family="sans-serif",
        font_size=7,
        title_size=8,
        label_size=7,
        tick_size=6,
        legend_size=6,
        line_width=1.0,
        marker_size=3,
        axes_linewidth=0.5,
        palette=VIBRANT_PALETTE,
    ),
    
    'presentation': PublicationTheme(
        name="Presentation",
        venue=VenueType.PRESENTATION,
        figure_width=10,
        figure_height=7,
        font_family="sans-serif",
        font_size=16,
        title_size=20,
        label_size=16,
        tick_size=14,
        legend_size=14,
        line_width=3.0,
        marker_size=10,
        axes_linewidth=1.5,
    ),
    
    'poster': PublicationTheme(
        name="Poster",
        venue=VenueType.POSTER,
        figure_width=12,
        figure_height=9,
        font_family="sans-serif",
        font_size=20,
        title_size=24,
        label_size=20,
        tick_size=18,
        legend_size=18,
        line_width=4.0,
        marker_size=12,
        axes_linewidth=2.0,
    ),
}


class ThemeManager:
    """Manages and applies plotting themes."""
    
    _current_theme: Optional[PublicationTheme] = None
    _custom_themes: Dict[str, PublicationTheme] = {}
    
    @classmethod
    def get_theme(cls, name: str) -> PublicationTheme:
        """Get a theme by name.
        
        Args:
            name: Theme name
        
        Returns:
            PublicationTheme instance
        """
        name_lower = name.lower()
        
        if name_lower in cls._custom_themes:
            return cls._custom_themes[name_lower]
        
        if name_lower in THEMES:
            return THEMES[name_lower]
        
        logger.warning(f"Theme '{name}' not found, using default")
        return THEMES['default']
    
    @classmethod
    def register_theme(cls, name: str, theme: PublicationTheme):
        """Register a custom theme.
        
        Args:
            name: Theme name
            theme: PublicationTheme instance
        """
        cls._custom_themes[name.lower()] = theme
        logger.info(f"Registered custom theme: {name}")
    
    @classmethod
    def apply_theme(cls, name: str):
        """Apply a theme to matplotlib.
        
        Args:
            name: Theme name
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available, cannot apply theme")
            return
        
        theme = cls.get_theme(name)
        cls._current_theme = theme
        
        # Reset to defaults first
        plt.rcdefaults()
        
        # Apply theme rcParams
        plt.rcParams.update(theme.to_rcparams())
        
        # Apply seaborn style if available
        if SEABORN_AVAILABLE:
            sns.set_style("whitegrid")
            sns.set_palette(theme.palette.primary)
        
        logger.debug(f"Applied theme: {theme.name}")
    
    @classmethod
    def get_current_theme(cls) -> Optional[PublicationTheme]:
        """Get the currently applied theme."""
        return cls._current_theme
    
    @classmethod
    def list_themes(cls) -> List[str]:
        """List all available themes."""
        return list(THEMES.keys()) + list(cls._custom_themes.keys())
    
    @classmethod
    def get_colors(cls, n: int = None) -> List[str]:
        """Get colors from current theme palette.
        
        Args:
            n: Number of colors to return (None for all)
        
        Returns:
            List of color hex codes
        """
        theme = cls._current_theme or THEMES['default']
        colors = theme.palette.primary
        
        if n is not None:
            # Cycle colors if n > len(colors)
            return [colors[i % len(colors)] for i in range(n)]
        
        return colors
    
    @classmethod
    def get_markers(cls, n: int = None) -> List[str]:
        """Get marker styles.
        
        Args:
            n: Number of markers to return
        
        Returns:
            List of marker styles
        """
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']
        
        if n is not None:
            return [markers[i % len(markers)] for i in range(n)]
        
        return markers
    
    @classmethod
    def get_linestyles(cls, n: int = None) -> List[str]:
        """Get line styles.
        
        Args:
            n: Number of line styles to return
        
        Returns:
            List of line styles
        """
        styles = ['-', '--', '-.', ':']
        
        if n is not None:
            return [styles[i % len(styles)] for i in range(n)]
        
        return styles
