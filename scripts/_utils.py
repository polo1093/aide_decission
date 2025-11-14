"""Compatibility layer for calibration helpers used by legacy scripts."""
from __future__ import annotations

from objet.utils.calibration import (
    Region,
    clamp_bbox,
    clamp_top_left,
    coerce_int,
    extract_patch,
    extract_region_images,
    load_coordinates,
    resolve_templates,
)

__all__ = [
    "Region",
    "coerce_int",
    "clamp_bbox",
    "clamp_top_left",
    "resolve_templates",
    "load_coordinates",
    "extract_patch",
    "extract_region_images",
]
