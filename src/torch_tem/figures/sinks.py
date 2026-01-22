"""Figure persistence and logging utilities.

Provides sinks for saving figures as PDF files and logging preview images
to TensorBoard. Implements the dual-output pattern: canonical PDF artifacts
on disk + rasterized previews for interactive monitoring.

Key Functions:
    save_pdf: Save a matplotlib Figure as a vector PDF.
    log_tensorboard_figure: Log a rasterized figure preview to TensorBoard.
    make_figure_path: Generate deterministic output paths for figures.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def save_pdf(
    fig: Figure,
    path: Path,
    *,
    bbox_inches: str = "tight",
    dpi: Optional[int] = None,
) -> None:
    """Save a matplotlib Figure as a PDF file.
    
    Args:
        fig: The matplotlib Figure to save.
        path: Output path for the PDF file. Parent directories are created if needed.
        bbox_inches: Bounding box mode ("tight" removes whitespace, None keeps default).
        dpi: DPI for embedded raster elements (does not affect vector elements).
            If None, uses the figure's current DPI setting.
    
    Raises:
        OSError: If the output directory cannot be created or the file cannot be written.
    
    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3])
        >>> save_pdf(fig, Path("output/my_figure.pdf"))
    """
    # Ensure output directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as PDF (vector format)
    fig.savefig(path, format="pdf", bbox_inches=bbox_inches, dpi=dpi)


def save_png(
    fig: Figure,
    path: Path,
    *,
    bbox_inches: str = "tight",
    dpi: int = 150,
) -> None:
    """Save a matplotlib Figure as a PNG file.
    
    Args:
        fig: The matplotlib Figure to save.
        path: Output path for the PNG file. Parent directories are created if needed.
        bbox_inches: Bounding box mode ("tight" removes whitespace, None keeps default).
        dpi: Resolution in dots per inch.
    
    Raises:
        OSError: If the output directory cannot be created or the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", bbox_inches=bbox_inches, dpi=dpi)


def log_tensorboard_figure(
    logger,
    tag: str,
    fig: Figure,
    global_step: int,
    *,
    close: bool = True,
    dpi: int = 150,
) -> None:
    """Log a figure preview to TensorBoard.
    
    Renders the matplotlib Figure to a raster image and logs it to TensorBoard's
    Images tab. The figure is converted to a NumPy array (HWC format) and logged
    via the Lightning logger's experiment (SummaryWriter).
    
    Args:
        logger: PyTorch Lightning TensorBoardLogger instance.
        tag: Tag name for the TensorBoard image (e.g., "figures/tem_overview").
        fig: The matplotlib Figure to log.
        global_step: Training step for x-axis alignment in TensorBoard.
        close: Whether to close the figure after logging (recommended to prevent
            memory leaks in long training runs).
        dpi: Resolution for rasterization.
    
    Raises:
        AttributeError: If logger is None or does not have an `experiment` attribute.
    
    Example:
        >>> from lightning.pytorch.loggers import TensorBoardLogger
        >>> logger = TensorBoardLogger("logs", name="my_run")
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3])
        >>> log_tensorboard_figure(logger, "figures/test", fig, global_step=100)
    """
    if logger is None:
        raise AttributeError("logger is None; cannot log to TensorBoard")
    
    if not hasattr(logger, 'experiment'):
        raise AttributeError(f"logger {type(logger).__name__} does not have 'experiment' attribute")
    
    # Get the SummaryWriter from the Lightning logger
    writer = logger.experiment
    
    # Render figure to a buffer as PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    buf.seek(0)
    
    # Load image as NumPy array and convert to CHW format for TensorBoard
    from PIL import Image
    img = np.array(Image.open(buf))
    
    # TensorBoard expects CHW (channels, height, width), but PIL gives HWC
    if img.ndim == 3:
        img = img.transpose(2, 0, 1)  # HWC -> CHW
    elif img.ndim == 2:
        img = img[np.newaxis, ...]  # Add channel dimension for grayscale
    
    # Log to TensorBoard
    writer.add_image(tag, img, global_step=global_step)
    
    # Close buffer and optionally close figure
    buf.close()
    if close:
        plt.close(fig)


def make_figure_path(
    base_dir: Path,
    figure_name: str,
    step: Optional[int] = None,
    version: Optional[str] = None,
    extension: str = "pdf",
) -> Path:
    """Generate a deterministic file path for a figure.
    
    Creates a path following the pattern:
        <base_dir>/figures/<figure_name>_[step<step>_][version_<version>].<extension>
    
    Args:
        base_dir: Base directory (typically logger.log_dir).
        figure_name: Descriptive name for the figure (e.g., "tem_overview").
        step: Optional training step to include in filename.
        version: Optional version/run identifier.
        extension: File extension ("pdf" or "png").
    
    Returns:
        Complete Path object for the figure file.
    
    Example:
        >>> make_figure_path(Path("logs/run_0"), "tem_overview", step=1000)
        PosixPath('logs/run_0/figures/tem_overview_step1000.pdf')
    """
    parts = [figure_name]
    if step is not None:
        parts.append(f"step{step}")
    if version is not None:
        parts.append(f"version_{version}")
    
    filename = "_".join(parts) + f".{extension}"
    return base_dir / "figures" / filename
