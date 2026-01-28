"""Figure composition and layout helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.gridspec as mpl_gridspec
import matplotlib.pyplot as plt

from torch_tem.figures.figures.base import GuidePolicy, PanelCallable
from torch_tem.figures.figures.styles import Theme, get_theme
from torch_tem.figures.figures.templates import Template, get_template


def make_grid(
    *,
    nrows: int,
    ncols: int,
    size: Optional[Tuple[float, float]] = None,
    sharex: bool = False,
    sharey: bool = False,
    template: Optional[str] = None,
) -> Tuple[plt.Figure, Any]:
    """Create a figure and axes grid.

    Args:
            nrows: Number of rows.
            ncols: Number of columns.
            size: Optional figure size.
            sharex: Share x-axis across panels.
            sharey: Share y-axis across panels.
            template: Optional template name.

    Returns:
            Tuple of Figure and axes array.
    """
    template_obj = get_template(template)
    figsize = size or template_obj.size
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        constrained_layout=template_obj.constrained_layout,
    )
    return fig, axes


def compose(
    *,
    panels: Sequence[PanelCallable],
    layout: Tuple[int, int],
    theme: Optional[str | Theme] = None,
    template: Optional[str | Template] = None,
    legend: str = "shared",
    colorbar: str = "shared",
    sharex: bool = False,
    sharey: bool = False,
    share_color_norm: bool = False,
    size: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Compose a multi-panel figure.

    Args:
            panels: Panel callables.
            layout: Tuple of (nrows, ncols).
            theme: Theme name or instance.
            template: Template name or instance.
            legend: Legend policy (none, per-axes, shared).
            colorbar: Colorbar policy (none, per-axes, shared).
            sharex: Share x-axis across panels.
            sharey: Share y-axis across panels.
            share_color_norm: Share color normalization across mappables.
            size: Optional figure size override.

    Returns:
            Matplotlib Figure instance.
    """
    nrows, ncols = layout
    if nrows * ncols != len(panels):
        raise ValueError("layout does not match number of panels")
    template_obj = template if isinstance(template, Template) else get_template(template)
    theme_obj = theme if isinstance(theme, Theme) else get_theme(theme)

    fig, axes = make_grid(
        nrows=nrows,
        ncols=ncols,
        size=size or template_obj.size,
        sharex=sharex,
        sharey=sharey,
        template=template_obj.name,
    )
    axes_list = _flatten_axes(axes)
    results: List[Any] = []
    panel_results: List[List[Any]] = []
    for panel, ax in zip(panels, axes_list):
        panel_output = list(_iter_results(_call_panel(panel, ax, theme_obj)))
        panel_results.append(panel_output)
        results.extend(panel_output)

    guides = GuidePolicy(legend=legend, colorbar=colorbar)
    if guides.legend == "per-axes":
        for ax in axes_list:
            ax.legend()
    if guides.legend == "shared":
        handles, labels = _collect_legend_items(results)
        if handles:
            fig.legend(handles, labels, loc="upper right")

    mappables = _collect_mappables(results)
    if share_color_norm and mappables:
        _apply_shared_norm(results, mappables)
    if guides.colorbar == "per-axes":
        for ax, panel_output in zip(axes_list, panel_results):
            mappable = _first_mappable(panel_output)
            if mappable is not None:
                fig.colorbar(mappable, ax=ax)
    if guides.colorbar == "shared" and mappables:
        fig.colorbar(mappables[0], ax=axes_list)
    return fig


def compose_gridspec(
    *,
    panels: Sequence[PanelCallable],
    layout: Tuple[int, int],
    theme: Optional[str | Theme] = None,
    template: Optional[str | Template] = None,
    legend: str = "shared",
    colorbar: str = "grouped",
    sharex: bool = False,
    sharey: bool = False,
    share_color_norm: bool = False,
    size: Optional[Tuple[float, float]] = None,
    gridspec_layout: Optional[Dict[str, Any]] = None,
    content_positions: Optional[Sequence[Tuple[int, int]]] = None,
    colorbar_groups: Optional[Dict[str, Tuple[int, int]]] = None,
) -> plt.Figure:
    """Compose a multi-panel figure using GridSpec and grouped guides.

    Args:
            panels: Panel callables.
            layout: Tuple of (nrows, ncols) for content panels.
            theme: Theme name or instance.
            template: Template name or instance.
            legend: Legend policy (none, per-axes, shared).
            colorbar: Colorbar policy (none, per-axes, shared, grouped).
            sharex: Share x-axis across content panels.
            sharey: Share y-axis across content panels.
            share_color_norm: Share color normalization across mappables.
            size: Optional figure size override.
            gridspec_layout: GridSpec layout overrides (nrows, ncols, ratios).
            content_positions: GridSpec positions for content panels.
            colorbar_groups: Map group name to GridSpec positions.

    Returns:
            Matplotlib Figure instance.
    """
    nrows, ncols = layout
    if nrows * ncols != len(panels):
        raise ValueError("layout does not match number of panels")

    template_obj = template if isinstance(template, Template) else get_template(template)
    theme_obj = theme if isinstance(theme, Theme) else get_theme(theme)

    gs_config = dict(gridspec_layout or {})
    gs_rows = int(gs_config.pop("nrows", nrows))
    gs_cols = int(gs_config.pop("ncols", ncols))
    fig = plt.figure(figsize=size or template_obj.size)
    grid = mpl_gridspec.GridSpec(gs_rows, gs_cols, figure=fig, **gs_config)

    content_positions = content_positions or _default_positions(nrows, ncols)
    if len(content_positions) != len(panels):
        raise ValueError("content_positions do not match number of panels")

    axes_list: List[Any] = []
    first_ax: Optional[Any] = None
    for row, col in content_positions:
        sharex_ax = first_ax if sharex else None
        sharey_ax = first_ax if sharey else None
        ax = fig.add_subplot(grid[row, col], sharex=sharex_ax, sharey=sharey_ax)
        if first_ax is None:
            first_ax = ax
        axes_list.append(ax)

    guide_axes: Dict[str, Any] = {}
    if colorbar_groups:
        for group, (row, col) in colorbar_groups.items():
            guide_axes[group] = fig.add_subplot(grid[row, col])

    results: List[Any] = []
    panel_results: List[List[Any]] = []
    for panel, ax in zip(panels, axes_list):
        panel_output = list(_iter_results(_call_panel(panel, ax, theme_obj)))
        panel_results.append(panel_output)
        results.extend(panel_output)

    guides = GuidePolicy(legend=legend, colorbar=colorbar)
    if guides.legend == "per-axes":
        for ax in axes_list:
            ax.legend()
    if guides.legend == "shared":
        handles, labels = _collect_legend_items(results)
        if handles:
            fig.legend(handles, labels, loc="upper right")

    if guides.colorbar == "per-axes":
        for ax, panel_output in zip(axes_list, panel_results):
            mappable = _first_mappable(panel_output)
            if mappable is not None:
                fig.colorbar(mappable, ax=ax)
    if guides.colorbar == "shared":
        mappables = _collect_mappables(results)
        if share_color_norm and mappables:
            _apply_shared_norm(results, mappables)
        if mappables:
            fig.colorbar(mappables[0], ax=axes_list)
    if guides.colorbar == "grouped":
        grouped = _collect_grouped_mappables(results)
        for group, mappables in grouped.items():
            if share_color_norm and mappables:
                _apply_group_shared_norm(results, mappables, group)
            if not mappables:
                continue
            cax = guide_axes.get(group)
            if cax is None:
                continue
            fig.colorbar(mappables[0], cax=cax)
    return fig


def _call_panel(panel: PanelCallable, ax: Any, theme: Theme) -> Any:
    """Invoke a panel callable with fallback for older signatures."""
    try:
        return panel(ax, theme=theme)
    except TypeError:
        return panel(ax)


def _flatten_axes(axes: Any) -> List[Any]:
    """Flatten Matplotlib axes into a list."""
    if hasattr(axes, "ravel"):
        return list(axes.ravel())
    return [axes]


def _iter_results(value: Any) -> Iterable[Any]:
    """Normalize panel output into an iterable of results."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return value
    return [value]


def _collect_legend_items(results: Sequence[Any]) -> Tuple[List[Any], List[str]]:
    """Collect legend handles and labels from results."""
    handles: List[Any] = []
    labels: List[str] = []
    seen = set()
    for result in results:
        handle, label = _legend_from_result(result)
        if handle is None or label is None:
            continue
        if label in seen:
            continue
        seen.add(label)
        handles.append(handle)
        labels.append(label)
    return handles, labels


def _legend_from_result(result: Any) -> Tuple[Optional[Any], Optional[str]]:
    """Extract legend handle and label from a result."""
    if hasattr(result, "legend_handle"):
        handle = getattr(result, "legend_handle")
        label = getattr(result, "label", None)
        return handle, label
    if hasattr(result, "get_label"):
        label = result.get_label()
        if label and not label.startswith("_"):
            return result, label
    return None, None


def _collect_mappables(results: Sequence[Any]) -> List[Any]:
    """Collect mappables for colorbar composition."""
    mappables: List[Any] = []
    for result in results:
        mappable = _first_mappable(_iter_results(result))
        if mappable is not None:
            mappables.append(mappable)
    return mappables


def _collect_grouped_mappables(results: Sequence[Any]) -> Dict[str, List[Any]]:
    """Collect mappables grouped by colorbar group."""
    grouped: Dict[str, List[Any]] = {}
    for result in results:
        group = _colorbar_group_from_result(result)
        if not group:
            continue
        mappable = _first_mappable(_iter_results(result))
        if mappable is None:
            continue
        grouped.setdefault(group, []).append(mappable)
    return grouped


def _colorbar_group_from_result(result: Any) -> Optional[str]:
    """Extract colorbar group from a result if present."""
    if hasattr(result, "colorbar_group"):
        return getattr(result, "colorbar_group")
    return None


def _apply_group_shared_norm(
    results: Sequence[Any],
    mappables: Sequence[Any],
    group: str,
) -> None:
    """Apply shared normalization to mappables within a group."""
    vmins = []
    vmaxs = []
    for result in results:
        if _colorbar_group_from_result(result) != group:
            continue
        if hasattr(result, "vmin") and getattr(result, "vmin") is not None:
            vmins.append(getattr(result, "vmin"))
        if hasattr(result, "vmax") and getattr(result, "vmax") is not None:
            vmaxs.append(getattr(result, "vmax"))
    if not vmins or not vmaxs:
        return
    vmin = min(vmins)
    vmax = max(vmaxs)
    for mappable in mappables:
        if hasattr(mappable, "set_clim"):
            mappable.set_clim(vmin=vmin, vmax=vmax)


def _default_positions(nrows: int, ncols: int) -> List[Tuple[int, int]]:
    """Return row-major GridSpec positions for a content grid."""
    return [(row, col) for row in range(nrows) for col in range(ncols)]


def _first_mappable(results: Iterable[Any]) -> Optional[Any]:
    """Return the first mappable found in results."""
    for result in results:
        if hasattr(result, "mappable") and getattr(result, "mappable") is not None:
            return getattr(result, "mappable")
    return None


def _apply_shared_norm(results: Sequence[Any], mappables: Sequence[Any]) -> None:
    """Apply shared normalization to all mappables if available."""
    vmins = []
    vmaxs = []
    for result in results:
        if hasattr(result, "vmin") and getattr(result, "vmin") is not None:
            vmins.append(getattr(result, "vmin"))
        if hasattr(result, "vmax") and getattr(result, "vmax") is not None:
            vmaxs.append(getattr(result, "vmax"))
    if not vmins or not vmaxs:
        return
    vmin = min(vmins)
    vmax = max(vmaxs)
    for mappable in mappables:
        if hasattr(mappable, "set_clim"):
            mappable.set_clim(vmin=vmin, vmax=vmax)
