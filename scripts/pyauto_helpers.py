import cv2
import numpy as np
import pyautogui
from pathlib import Path
from typing import Optional, Tuple, Union
from PIL import Image, ImageOps

HaystackType = Union[np.ndarray, Image.Image]
NeedleType = Union[str, Path, Image.Image]


def _to_pil_rgb(img: HaystackType, assume_bgr: bool = False) -> Image.Image:
    """Convertit un np.ndarray ou une PIL.Image en PIL RGB.

    - np.ndarray BGR -> converti correctement en RGB
    - np.ndarray RGB -> utilisé tel quel
    - PIL.Image -> convert("RGB")
    """
    if isinstance(img, Image.Image):
        return img.convert("RGB")

    if not isinstance(img, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(img)}")

    arr = img
    if arr.ndim == 2:
        # gris -> RGB
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
    """
    Localise `needle` dans `haystack` en utilisant pyautogui.locate
    (équivalent de locateOnScreen mais sur image déjà fournie).

    Args:
        haystack: image de fond (crop) - np.ndarray ou PIL.Image.
        needle: chemin vers le template OU PIL.Image.
        assume_bgr: True si haystack est un np.ndarray OpenCV (BGR).
        grayscale: idem locateOnScreen(grayscale=...).
        confidence: idem locateOnScreen(confidence=...).

    Returns:
        (left, top, width, height) relatifs au haystack, ou None si non trouvé.
    """
    # 1) Convertir le haystack en PIL RGB
    haystack_pil = _to_pil_rgb(haystack, assume_bgr=assume_bgr)

    # 2) Charger/convertir le needle
    if isinstance(needle, (str, Path)):
        needle_pil = Image.open(needle).convert("RGB")
    elif isinstance(needle, Image.Image):
        needle_pil = needle.convert("RGB")
    else:
        raise TypeError(f"Unsupported needle type: {type(needle)}")

    # 3) Appel direct à pyautogui
    box = pyautogui.locate(
        needle_pil,
        haystack_pil,
        grayscale=grayscale,
        confidence=confidence,
    )
    if box is None:
        return None

    # box est un pyautogui.Box(left, top, width, height)
    return int(box.left), int(box.top), int(box.width), int(box.height)
