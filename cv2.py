"""Minimal subset of OpenCV (cv2) required for cropping scripts.

This stub implements the tiny subset used in the repository so the
code can run in environments where the official ``opencv-python`` wheel
is unavailable (for example when ``libGL`` is missing).  Only RGB↔BGR
conversion, RGB→gray conversion, basic template matching, and
``minMaxLoc`` are provided.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

# Color conversion constants (compatible values are not required, only identity)
COLOR_RGB2GRAY = 1
COLOR_BGR2RGB = 2
COLOR_RGB2BGR = 3

# Template matching method constants
TM_SQDIFF = 0
TM_SQDIFF_NORMED = 1
TM_CCORR_NORMED = 4
TM_CCOEFF = 5
TM_CCOEFF_NORMED = 6

CAP_PROP_FPS = 5

__version__ = "0.0-stub"


def _ensure_array(arr: np.ndarray) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    return np.asarray(arr)


def cvtColor(arr: np.ndarray, code: int) -> np.ndarray:
    arr = _ensure_array(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Unsupported image shape for cvtColor: {arr.shape}")
    if code == COLOR_RGB2GRAY:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray.astype(np.float32)
    if code in (COLOR_BGR2RGB, COLOR_RGB2BGR):
        return arr[..., ::-1]
    raise ValueError(f"Unsupported conversion code: {code}")


def _integral_image(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float64, copy=False)
    return arr.cumsum(axis=0).cumsum(axis=1)


def _sum_from_integral(ii: np.ndarray, h: int, w: int) -> np.ndarray:
    ii_pad = np.pad(ii, ((1, 0), (1, 0)), mode="constant")
    return (
        ii_pad[h:, w:]
        - ii_pad[:-h, w:]
        - ii_pad[h:, :-w]
        + ii_pad[:-h, :-w]
    )


def _correlate_valid(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    ih, iw = image.shape
    th, tw = template.shape
    fh = ih + th - 1
    fw = iw + tw - 1
    fft_im = np.fft.rfftn(image, s=(fh, fw))
    fft_tpl = np.fft.rfftn(np.flip(template, axis=(0, 1)), s=(fh, fw))
    conv = np.fft.irfftn(fft_im * fft_tpl, s=(fh, fw))
    y0 = th - 1
    x0 = tw - 1
    return np.real(conv[y0 : y0 + ih - th + 1, x0 : x0 + iw - tw + 1])


def _match_template_ccoeff_normed(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    template = template.astype(np.float32)
    ih, iw = image.shape
    th, tw = template.shape
    if ih < th or iw < tw:
        raise ValueError("Template larger than image")

    tpl = template - template.mean()
    tpl_norm = np.sqrt(np.sum(tpl**2))
    if tpl_norm < 1e-9:
        tpl_norm = 1e-9

    area = float(th * tw)
    corr = _correlate_valid(image, tpl)
    ii = _integral_image(image)
    ii_sq = _integral_image(image**2)
    sum_patch = _sum_from_integral(ii, th, tw)
    sum_sq_patch = _sum_from_integral(ii_sq, th, tw)
    mean_patch = sum_patch / area
    var_patch = sum_sq_patch - (sum_patch**2) / area
    denom = np.sqrt(np.maximum(var_patch, 1e-9)) * tpl_norm
    response = corr / denom
    response[var_patch <= 1e-6] = -1.0
    return response


def _match_template_sqdiff(image: np.ndarray, template: np.ndarray, normed: bool) -> np.ndarray:
    image = image.astype(np.float32)
    template = template.astype(np.float32)
    ih, iw = image.shape
    th, tw = template.shape
    if ih < th or iw < tw:
        raise ValueError("Template larger than image")

    tpl_sum = np.sum(template**2)
    if tpl_sum < 1e-9:
        tpl_sum = 1e-9

    ii = _integral_image(image)
    ii_sq = _integral_image(image**2)
    sum_patch = _sum_from_integral(ii, th, tw)
    sum_sq_patch = _sum_from_integral(ii_sq, th, tw)
    corr = _correlate_valid(image, template)
    ssd = sum_sq_patch - 2 * corr + tpl_sum
    if normed:
        denom = np.sqrt(np.maximum(sum_sq_patch, 1e-9) * tpl_sum)
        out = ssd / denom
        out[sum_sq_patch <= 1e-6] = 0.0
        return out
    return ssd


def matchTemplate(image: np.ndarray, template: np.ndarray, method: int = TM_CCOEFF_NORMED) -> np.ndarray:
    image = _ensure_array(image)
    template = _ensure_array(template)
    if image.ndim == 3:
        image = cvtColor(image, COLOR_RGB2GRAY)
    if template.ndim == 3:
        template = cvtColor(template, COLOR_RGB2GRAY)
    if method in (TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR_NORMED):
        return _match_template_ccoeff_normed(image, template)
    if method == TM_SQDIFF:
        return _match_template_sqdiff(image, template, normed=False)
    if method == TM_SQDIFF_NORMED:
        return _match_template_sqdiff(image, template, normed=True)
    raise ValueError(f"Unsupported matchTemplate method: {method}")


def minMaxLoc(mat: np.ndarray) -> Tuple[float, float, Tuple[int, int], Tuple[int, int]]:
    mat = _ensure_array(mat)
    if mat.size == 0:
        raise ValueError("Empty matrix")
    min_val = float(np.min(mat))
    max_val = float(np.max(mat))
    min_idx = int(np.argmin(mat))
    max_idx = int(np.argmax(mat))
    min_loc = (int(min_idx % mat.shape[1]), int(min_idx // mat.shape[1]))
    max_loc = (int(max_idx % mat.shape[1]), int(max_idx // mat.shape[1]))
    return min_val, max_val, min_loc, max_loc


# Provide dtype constants used by numpy conversions for compatibility
CV_8U = np.uint8
CV_32F = np.float32

__all__ = [
    "COLOR_RGB2GRAY",
    "COLOR_BGR2RGB",
    "COLOR_RGB2BGR",
    "TM_SQDIFF",
    "TM_SQDIFF_NORMED",
    "TM_CCORR_NORMED",
    "TM_CCOEFF",
    "TM_CCOEFF_NORMED",
    "cvtColor",
    "matchTemplate",
    "minMaxLoc",
    "VideoCapture",
    "CAP_PROP_FPS",
    "CV_8U",
    "CV_32F",
]


class VideoCapture:
    """Simplified VideoCapture that treats image files as single-frame videos."""

    def __init__(self, path: str):
        from pathlib import Path
        from PIL import Image

        self._path = Path(path)
        self._frame = None
        self._opened = False
        if self._path.suffix.lower() in {".png", ".jpg", ".jpeg"} and self._path.exists():
            img = Image.open(self._path).convert("RGB")
            self._frame = np.array(img)[..., ::-1]  # RGB -> BGR
            self._opened = True
            self._consumed = False
        else:
            self._opened = False
            self._consumed = True

    def isOpened(self) -> bool:
        return bool(self._opened)

    def read(self):
        if not self._opened or self._consumed:
            return False, None
        self._consumed = True
        return True, self._frame.copy()

    def release(self) -> None:
        self._opened = False

    def get(self, prop: int) -> float:
        if prop == CAP_PROP_FPS:
            return 30.0
        return 0.0