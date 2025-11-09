"""Common helpers for calibration scripts.

This module centralises the JSON loading/parsing logic shared by the
calibration utilities as well as a couple of small image helpers.  The
functions are intentionally dependency-light so they can be imported from
Tk/CLI tools alike.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from PIL import Image

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


@dataclass(frozen=True)
class Region:
    """Simple container describing a rectangular capture zone."""

    key: str
    group: str
    top_left: Tuple[int, int]
    size: Tuple[int, int]
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return the JSON-compatible representation of the region."""

        payload = {"group": self.group, "top_left": list(self.top_left)}
        payload.update(self.meta)
        return payload


def coerce_int(value: Any, default: int = 0) -> int:
    """Convert *value* to an int, falling back to *default* on failure."""

    try:
        return int(round(float(value)))
    except Exception:
        return default


def clamp_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Tuple[int, int, int, int]:
    """Clamp a bounding box to the image boundaries."""

    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def clamp_top_left(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int]:
    """Ensure the rectangle starting at (x, y) stays inside (W, H)."""

    if W <= 0 or H <= 0:
        return x, y
    x = max(0, min(x, max(0, W - w)))
    y = max(0, min(y, max(0, H - h)))
    return x, y


def resolve_templates(templates: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Resolve *templates* aliases and expose a uniform mapping."""

    def get_size(name: str, seen: Optional[set] = None) -> Tuple[int, int]:
        if seen is None:
            seen = set()
        if name in seen:
            return 0, 0
        seen.add(name)
        tpl = templates.get(name, {})
        if "size" in tpl:
            w, h = tpl.get("size", [0, 0])
            return coerce_int(w), coerce_int(h)
        alias = tpl.get("alias_of")
        if alias:
            return get_size(str(alias), seen)
        return 0, 0

    def get_type(name: str, seen: Optional[set] = None) -> str:
        if seen is None:
            seen = set()
        if name in seen:
            return ""
        seen.add(name)
        tpl = templates.get(name, {})
        typ = tpl.get("type")
        if typ:
            return str(typ)
        alias = tpl.get("alias_of")
        if alias:
            return get_type(str(alias), seen)
        return ""

    def get_layout(name: str, seen: Optional[set] = None) -> Dict[str, Any]:
        if seen is None:
            seen = set()
        if name in seen:
            return {}
        seen.add(name)
        tpl = templates.get(name, {})
        layout = tpl.get("layout")
        if isinstance(layout, Mapping):
            return dict(layout)
        alias = tpl.get("alias_of")
        if alias:
            return get_layout(str(alias), seen)
        return {}

    resolved: Dict[str, Dict[str, Any]] = {}
    for group in templates.keys():
        size = get_size(group)
        typ = get_type(group)
        layout = get_layout(group)
        payload = {"size": [size[0], size[1]], "type": typ}
        if layout:
            payload["layout"] = layout
        resolved[group] = payload
    return resolved


def _normalise_region_entry(key: str, raw: Mapping[str, Any], templates: Mapping[str, Dict[str, Any]]) -> Region:
    group = str(raw.get("group", ""))
    top_left = raw.get("top_left", [0, 0])
    size = templates.get(group, {}).get("size", [0, 0])
    meta = {k: v for k, v in raw.items() if k not in {"group", "top_left"}}
    return Region(
        key=key,
        group=group,
        top_left=(coerce_int(top_left[0]), coerce_int(top_left[1])),
        size=(coerce_int(size[0]), coerce_int(size[1])),
        meta=dict(meta),
    )


def load_coordinates(path: Path | str) -> Tuple[Dict[str, Region], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Load a coordinates.json file.

    Returns ``(regions, templates_resolved, table_capture)`` where ``regions``
    maps keys to :class:`Region` instances.
    """

    coord_path = Path(path)
    with coord_path.open("r", encoding="utf-8") as fh:
        payload: Dict[str, Any] = json.load(fh)

    templates = payload.get("templates", {})
    resolved = resolve_templates(templates)
    raw_regions = payload.get("regions", {})
    regions = {
        key: _normalise_region_entry(key, raw, resolved)
        for key, raw in raw_regions.items()
    }
    table_capture = payload.get("table_capture", {})
    return regions, resolved, table_capture


def extract_patch(image: Image.Image, top_left: Tuple[int, int], size: Tuple[int, int], pad: int = 4) -> Image.Image:
    """Crop ``image`` around ``top_left``/``size`` with a soft *pad*."""

    x, y = map(int, top_left)
    w, h = map(int, size)
    width, height = image.size
    x1, y1 = x - pad, y - pad
    x2, y2 = x + w + pad, y + h + pad
    x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, width, height)
    return image.crop((x1, y1, x2, y2))


def extract_region_images(
    table_img: Image.Image,
    regions: Mapping[str, Region | Mapping[str, Any]],
    *,
    pad: int = 4,
    groups_numbers: Tuple[str, ...] = ("player_card_number", "board_card_number"),
    groups_suits: Tuple[str, ...] = ("player_card_symbol", "board_card_symbol"),
) -> Dict[str, Tuple[Image.Image, Image.Image]]:
    """Return ``{base_key: (number_patch, suit_patch)}`` for cards regions."""

    def group_of(region: Region | Mapping[str, Any]) -> str:
        return region.group if isinstance(region, Region) else str(region.get("group", ""))

    def top_left_of(region: Region | Mapping[str, Any]) -> Tuple[int, int]:
        if isinstance(region, Region):
            return region.top_left
        top_left = region.get("top_left", [0, 0])
        return coerce_int(top_left[0]), coerce_int(top_left[1])

    def size_of(region: Region | Mapping[str, Any]) -> Tuple[int, int]:
        if isinstance(region, Region):
            return region.size
        size = region.get("size")
        if isinstance(size, Iterable):
            values = list(size)
            if len(values) >= 2:
                return coerce_int(values[0]), coerce_int(values[1])
        return 0, 0

    pairs: Dict[str, Dict[str, Image.Image]] = {}

    for key, region in regions.items():
        if group_of(region) in groups_numbers:
            patch = extract_patch(table_img, top_left_of(region), size_of(region), pad)
            base = key.replace("_number", "")
            pairs.setdefault(base, {})["number"] = patch

    for key, region in regions.items():
        if group_of(region) in groups_suits:
            patch = extract_patch(table_img, top_left_of(region), size_of(region), pad)
            base = key.replace("_symbol", "")
            pairs.setdefault(base, {})["symbol"] = patch

    out: Dict[str, Tuple[Image.Image, Image.Image]] = {}
    for base, mapping in pairs.items():
        if "number" in mapping and "symbol" in mapping:
            out[base] = (mapping["number"], mapping["symbol"])
    return out
