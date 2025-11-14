"""Card template helpers shared between scanner code and CLI tools."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

__all__ = [
    "CardObservation",
    "TemplateIndex",
    "is_card_present",
    "match_best",
    "recognize_number_and_suit",
]


@dataclass
class CardObservation:
    """Observation brute d'une carte (issue de la capture)."""

    value: Optional[str]
    suit: Optional[str]
    value_score: float
    suit_score: float
    source: str = "capture"


class TemplateIndex:
    """Charge les gabarits de chiffres/figures et de symboles depuis config/<game>/cards."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.numbers: Dict[str, List[np.ndarray]] = {}
        self.suits: Dict[str, List[np.ndarray]] = {}

    @staticmethod
    def _prep(gray: np.ndarray) -> np.ndarray:
        return gray

    @staticmethod
    def _imread_gray(p: Path) -> Optional[np.ndarray]:
        try:
            img = Image.open(p).convert("L")
            return np.array(img)
        except Exception:
            return None

    def _load_dir(self, sub: str) -> Dict[str, List[np.ndarray]]:
        base = self.root / sub
        out: Dict[str, List[np.ndarray]] = {}
        if not base.exists():
            return out
        for label_dir in sorted(base.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            imgs: List[np.ndarray] = []
            for f in sorted(label_dir.glob("*.png")):
                g = self._imread_gray(f)
                if g is not None:
                    imgs.append(self._prep(g))
            if imgs:
                out[label] = imgs
        return out

    def load(self) -> None:
        self.numbers = self._load_dir("numbers")
        self.suits = self._load_dir("suits")

    def check_missing(
        self,
        expect_numbers: Optional[Iterable[str]] = None,
        expect_suits: Optional[Iterable[str]] = None,
    ) -> Dict[str, List[str]]:
        miss: Dict[str, List[str]] = {"numbers": [], "suits": []}
        if expect_numbers:
            for v in expect_numbers:
                if v not in self.numbers:
                    miss["numbers"].append(v)
        if expect_suits:
            for s in expect_suits:
                if s not in self.suits:
                    miss["suits"].append(s)
        return miss


def _to_gray(img):
    """Normalise un patch en niveau de gris (ndarray 2D).

    Accepte :
    - un numpy.ndarray (BGR ou déjà en gris),
    - une image PIL,
    - au pire, tout objet convertible en ndarray.
    """
    # 1) Cas OpenCV / numpy
    if isinstance(img, np.ndarray):
        # Déjà en niveaux de gris
        if img.ndim == 2:
            return img
        # Image couleur (en pratique BGR si ça vient de cv2 / screen_crop)
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        raise ValueError(f"Format ndarray inattendu pour _to_gray: shape={img.shape}")

    # 2) Cas PIL.Image
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # 3) Fallback : on tente de convertir en ndarray
    arr = np.array(img)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # On part du principe que c’est du BGR (cas le plus probable avec OpenCV)
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    raise ValueError(f"Type d'image non supporté pour _to_gray: {type(img)}")


def is_card_present(patch: np.ndarray | Image.Image, *, threshold: int = 240, min_ratio: float = 0.08) -> bool:
    """Heuristique simple : proportion de pixels *très clairs* sur la zone."""

    if isinstance(patch, np.ndarray):
        arr = patch
        if arr.ndim == 2:
            arr_u8 = arr.astype(np.uint8, copy=False)
            white = arr_u8 >= threshold
            ratio = float(white.mean())
            return ratio >= float(min_ratio)
        if arr.ndim == 3:
            arr_u8 = arr.astype(np.uint8, copy=False)
        else:
            raise ValueError(f"Unsupported array shape for card presence: {arr.shape}")
    elif isinstance(patch, Image.Image):
        arr_u8 = np.array(patch.convert("RGB"), dtype=np.uint8)
    else:
        raise TypeError(f"Unsupported patch type for card presence: {type(patch)!r}")

    if arr_u8.ndim == 2:
        white = arr_u8 >= threshold
    else:
        white = np.all(arr_u8 >= threshold, axis=2)

    ratio = float(white.mean())
    return ratio >= float(min_ratio)


def match_best(gray_img: np.ndarray, templates: List[np.ndarray], method: int = cv2.TM_CCOEFF_NORMED) -> float:
    best = -1.0
    for tpl in templates:
        if gray_img.shape[0] < tpl.shape[0] or gray_img.shape[1] < tpl.shape[1]:
            continue
        res = cv2.matchTemplate(gray_img, tpl, method)
        _, score, _, _ = cv2.minMaxLoc(res)
        best = max(best, float(score))
    return best


def recognize_number_and_suit(
    number_patch: Image.Image,
    suit_patch: Image.Image,
    idx: TemplateIndex,
) -> Tuple[Optional[str], Optional[str], float, float]:
    """Retourne (value, suit, score_value, score_suit)."""

    g_num = _to_gray(number_patch)
    g_suit = _to_gray(suit_patch)

    best_num, best_num_score = None, -1.0
    for label, tpls in idx.numbers.items():
        score = match_best(g_num, tpls)
        if score > best_num_score:
            best_num_score = score
            best_num = label

    best_suit, best_suit_score = None, -1.0
    for label, tpls in idx.suits.items():
        score = match_best(g_suit, tpls)
        if score > best_suit_score:
            best_suit_score = score
            best_suit = label

    return best_num, best_suit, best_num_score, best_suit_score
