"""Compatibility layer for calibration helpers used by legacy scripts."""
from __future__ import annotations

from objet.utils.calibration import (
    CardPatch,
    Region,
    clamp_bbox,
    clamp_top_left,
    coerce_int,
    extract_patch,
    collect_card_patches,
    load_coordinates,
    resolve_templates,
)

__all__ = [
    "CardPatch",
    "Region",
    "coerce_int",
    "clamp_bbox",
    "clamp_top_left",
    "resolve_templates",
    "load_coordinates",
    "extract_patch",
    "collect_card_patches",
]
