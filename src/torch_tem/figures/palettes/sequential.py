"""Sequential colormap utilities."""

from __future__ import annotations

from typing import Dict

import matplotlib.cm as cm
from matplotlib.colors import Colormap

_COLORMAPS: Dict[str, str] = {
    "default": "viridis",
    "heat": "inferno",
    "cool": "plasma",
    "blue": "Blues",
}


def get_colormap(name: str | None) -> Colormap:
    """Return a Matplotlib colormap.

    Args:
            name: Colormap name or None for default.

    Returns:
            Matplotlib Colormap instance.
    """
    key = name or "default"
    cmap_name = _COLORMAPS.get(key, _COLORMAPS["default"])
    return cm.get_cmap(cmap_name)
