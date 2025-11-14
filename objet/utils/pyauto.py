"""Helpers built around :mod:`pyautogui` usable from app code and scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import pyautogui
from PIL import Image

HaystackType = Union[np.ndarray, Image.Image]
NeedleType = Union[str, Path, Image.Image]

__all__ = ["locate_in_image"]


def _to_pil_rgb(img: HaystackType, assume_bgr: bool = False) -> Image.Image:
    """Convert ``img`` into a :class:`PIL.Image.Image` in RGB mode."""

    if isinstance(img, Image.Image):
        return img.convert("RGB")

    if not isinstance(img, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(img)}")

    arr = img
    if arr.ndim == 2:
        # grayscale -> RGB
        return Image.fromarray(arr).convert("RGB")

    if arr.ndim == 3 and arr.shape[2] == 3:
        if assume_bgr:
            arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        else:
            arr_rgb = arr
        return Image.fromarray(arr_rgb)

    raise ValueError(f"Unsupported array shape: {arr.shape}")


def locate_in_image(
    haystack: HaystackType,
    needle: NeedleType,
    *,
    assume_bgr: bool = False,
    grayscale: bool = False,
    confidence: float = 0.9,
) -> Optional[Tuple[int, int, int, int]]:
    """Locate ``needle`` inside ``haystack`` using ``pyautogui.locate``."""

    haystack_pil = _to_pil_rgb(haystack, assume_bgr=assume_bgr)

    if isinstance(needle, (str, Path)):
        needle_pil = Image.open(needle).convert("RGB")
    elif isinstance(needle, Image.Image):
        needle_pil = needle.convert("RGB")
    else:
        raise TypeError(f"Unsupported needle type: {type(needle)}")

    box = pyautogui.locate(
        needle_pil,
        haystack_pil,
        grayscale=grayscale,
        confidence=confidence,
    )
    if box is None:
        return None

    return int(box.left), int(box.top), int(box.width), int(box.height)
