"""Shared helpers for figure-level orchestration."""

from __future__ import annotations

from contextlib import nullcontext

from torch_tem.figures.registry import FigureContext


def style_context(ctx: FigureContext):
    """Return a scoped style context manager for figure generation.

    Args:
        ctx: Figure context containing an optional ``style``.

    Returns:
        A context manager that applies the style, or a no-op context.
    """
    style = getattr(ctx, "style", None)
    if style is None:
        return nullcontext()
    return style.apply_context()
