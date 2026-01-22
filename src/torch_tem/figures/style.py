"""Centralized styling configuration for TEM figures.

Provides consistent matplotlib styling across all figure modules, including:
- Figure DPI and size defaults
- Font sizes and families
- Colormap choices
- Axis formatting conventions

All style settings are configurable via a single StyleConfig dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt


@dataclass
class StyleConfig:
    """Matplotlib style configuration for TEM figures.
    
    Attributes:
        dpi: Dots per inch for rasterization (PDF is vector, but affects preview).
        font_size: Base font size in points.
        font_family: Font family (e.g., 'sans-serif', 'serif').
        colormap_sequential: Default colormap for sequential data (rates, probabilities).
        colormap_diverging: Default colormap for diverging data (errors, differences).
        colormap_qualitative: Default colormap for categorical data (actions, discrete labels).
        figure_facecolor: Background color for the figure.
        axes_facecolor: Background color for axes.
    """
    dpi: int = 150
    font_size: int = 10
    font_family: str = "sans-serif"
    colormap_sequential: str = "viridis"
    colormap_diverging: str = "RdBu_r"
    colormap_qualitative: str = "Pastel1"
    figure_facecolor: str = "white"
    axes_facecolor: str = "white"
    
    def apply(self) -> None:
        """Apply this style configuration to matplotlib rcParams.
        
        This modifies the global matplotlib configuration. Call this once
        before generating figures, or use as a context manager via apply_context().
        """
        mpl.rcParams['figure.dpi'] = self.dpi
        mpl.rcParams['font.size'] = self.font_size
        mpl.rcParams['font.family'] = self.font_family
        mpl.rcParams['figure.facecolor'] = self.figure_facecolor
        mpl.rcParams['axes.facecolor'] = self.axes_facecolor
    
    def apply_context(self):
        """Return a context manager that applies this style temporarily.
        
        Example:
            >>> style = StyleConfig(dpi=200)
            >>> with style.apply_context():
            ...     fig = make_figure(...)  # Uses dpi=200
            >>> # Outside context, original rcParams restored
        """
        return plt.rc_context({
            'figure.dpi': self.dpi,
            'font.size': self.font_size,
            'font.family': self.font_family,
            'figure.facecolor': self.figure_facecolor,
            'axes.facecolor': self.axes_facecolor,
        })


# Default style instance for consistent usage across modules
DEFAULT_STYLE = StyleConfig()
