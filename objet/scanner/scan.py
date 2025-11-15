
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import ImageGrab, Image
import logging

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from objet.utils.pyauto import locate_in_image
from objet.utils.calibration import bbox_from_region, load_coordinates
from objet.scanner.cards_recognition import (
    TemplateIndex,
    is_cover_me_cards,
    is_card_present,
    recognize_number_and_suit,
)

DEFAULT_COORD_PATH = Path("config/PMU/coordinates.json")
DEFAULT_ANCHOR_PATH = Path("config/PMU/anchor.png")
DEFAULT_CARDS_ROOT = Path("config/PMU/Cards")


class ScanTable:
    """Scan de la table PMU basé sur capture écran + pyautogui.

    - Localisation de la fenêtre via un template d'ancre (me.png) avec locate_in_image().
    - ``screen_array`` conserve la capture plein écran en BGR (convention OpenCV).
    """

    def __init__(self, *, value_threshold: float = 0.75, suit_threshold: float = 0.75) -> None:
        # --- Config / calibration ---
        self.coord_path = DEFAULT_COORD_PATH

        # Gabarit de référence (ancre) utilisé par pyautogui/locate
        self.reference_pil: Image.Image = Image.open(DEFAULT_ANCHOR_PATH).convert("RGB")

        # --- État runtime ---
        self.screen_array: Optional[np.ndarray] = None     # plein écran, BGR
        self.anchor_box: Optional[Tuple[int, int, int, int]] = None
        self.scan_string: str = "init"
        self.cards_root = DEFAULT_CARDS_ROOT
        self.template_index = TemplateIndex(self.cards_root)
        self.template_index.load()
        self.player_state_boxes: Dict[str, Tuple[int, int, int, int]] = {}
        self._load_player_state_regions()
        self.set_thresholds(value_threshold=value_threshold, suit_threshold=suit_threshold)
        # Première capture
        self.screen_refresh()


    
    def test_scan(self) -> bool:
        self.screen_refresh()
        return self.find_table()
    
    def screen_refresh(self) -> None:
        """Capture plein écran dans self.screen_array (numpy BGR)."""
        grab = ImageGrab.grab()               # PIL RGB
        rgb = np.array(grab)                  # numpy RGB
        self.screen_array = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # numpy BGR
        self.anchor_box = None

    # ------------------------------------------------------------------
    # Localisation de la table via pyautogui
    # ------------------------------------------------------------------
    def find_table(self, *, grayscale: bool = True, confidence: float = 0.9) -> bool:
        """Localise la table via l'ancre.

        Met à jour :
          - ``self.anchor_box`` avec la bounding box de l'ancre (left, top, w, h)
          - ``self.scan_string`` : 'ok' ou "don't find".
        """
        if self.screen_array is None:
            self.scan_string = "no_screen"
            self.anchor_box = None
            return False

        # 1) Localiser l'ancre dans le plein écran via pyautogui
        box = locate_in_image(
            haystack=self.screen_array,
            needle=self.reference_pil,
            assume_bgr=True,
            grayscale=grayscale,
            confidence=confidence,
        )

        if box is None:
            self.scan_string = "don't find"
            self.anchor_box = None
            return False

        anchor_left, anchor_top, anchor_w, anchor_h = box
        self.anchor_box = (int(anchor_left), int(anchor_top), int(anchor_w), int(anchor_h))
        self.scan_string = "ok"
        return True

    # ------------------------------------------------------------------
    # Scan des cartes directement sur la capture plein écran
    # ------------------------------------------------------------------
    def set_thresholds(self, *, value_threshold: float, suit_threshold: float) -> None:
        """Définit les seuils de similarité (0-1) utilisés pour valider valeur et symbole."""

        self.value_threshold = float(max(0.0, min(1.0, value_threshold)))
        self.suit_threshold = float(max(0.0, min(1.0, suit_threshold)))

    def scan_carte(
        self,
        position_value: Tuple[int, int, int, int],
        position_suit: Tuple[int, int, int, int],
        *,
        template_set: Optional[str] = None,
        fold_state_key: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str], float, float]:
        """
        Retourne:
            (value, suit, confidence_value, confidence_suit)

        - value, suit : str ou None
        - confidence_* : float entre 0.0 et 1.0
        """


        # crops séparés pour la valeur et le symbole
        image_card_value = self._extract_patch(position_value)
        image_card_suit = self._extract_patch(position_suit)


        if self._should_skip_for_fold(image_card_value, template_set, fold_state_key):
            return None, None, 0.0, 0.0

        if is_card_present(image_card_value):
            carte_value, carte_suit, score_value, score_suit = recognize_number_and_suit(
                image_card_value,
                image_card_suit,
                self.template_index,
                template_set=template_set,
            )

            conf_val = float(max(0.0, score_value or 0.0))
            conf_suit = float(max(0.0, score_suit or 0.0))

            value_ok = carte_value if (carte_value and conf_val >= self.value_threshold) else None
            suit_ok = carte_suit if (carte_suit and conf_suit >= self.suit_threshold) else None

            return value_ok, suit_ok, conf_val, conf_suit
        return None, None, 0.0, 0.0

    def _load_player_state_regions(self) -> None:
        try:
            regions, _, _ = load_coordinates(self.coord_path)
        except FileNotFoundError:
            logging.getLogger(__name__).warning(
                "Fichier de coordonnées introuvable pour la détection FOLD: %s",
                self.coord_path,
            )
            return
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Échec du chargement des coordonnées pour la détection FOLD: %s",
                exc,
            )
            return

        for key, region in regions.items():
            if not key.startswith("player_state"):
                continue
            box = bbox_from_region(region)
            if box:
                self.player_state_boxes[key] = box

    def _should_skip_for_fold(
        self,
        number_patch: np.ndarray,
        template_set: Optional[str],
        fold_state_key: Optional[str],
    ) -> bool:
        candidates: List[Union[np.ndarray, Image.Image]] = []

        if fold_state_key:
            box = self.player_state_boxes.get(fold_state_key)
            if box is not None:
                state_patch = self._extract_patch(box, pad=0)
                if self._patch_has_pixels(state_patch):
                    candidates.append(state_patch)

        template_hint = (template_set or "").lower()
        if not candidates and any(token in template_hint for token in ("hand", "player")):
            if self._patch_has_pixels(number_patch):
                candidates.append(number_patch)

        for patch in candidates:
            if is_cover_me_cards(patch, threshold=0.6):
                return True
        return False

    @staticmethod
    def _patch_has_pixels(patch: Union[np.ndarray, Image.Image]) -> bool:
        if isinstance(patch, np.ndarray):
            return patch.size > 0 and patch.ndim >= 2 and patch.shape[0] > 0 and patch.shape[1] > 0
        if isinstance(patch, Image.Image):
            width, height = patch.size
            return width > 0 and height > 0
        return False


 
    
    # Stubs à compléter plus tard
    def scan_pot(self, position):
        return None

    def scan_player(self, position):
        return None, None

    def scan_money_player(self, position):
        return None

    def scan_bouton(self, position):
        return None, None


    def _extract_patch(self, box: Tuple[int, int, int, int], pad: int = 3) -> np.ndarray:
        """Retourne un crop (numpy BGR) pour ``box=(x,y,w,h)`` avec padding et clamp."""

        if self.screen_array is None:
            return np.empty((0, 0), dtype=np.uint8)

        x, y, w, h = box
        x = int(round(x))
        y = int(round(y))
        w = max(0, int(round(w)))
        h = max(0, int(round(h)))

        if w == 0 or h == 0:
            return np.empty((0, 0), dtype=np.uint8)

        x0 = max(0, int(x - pad))
        y0 = max(0, int(y - pad))

        h_scr, w_scr = self.screen_array.shape[:2]
        x1 = min(w_scr, int(x + w + pad))
        y1 = min(h_scr, int(y + h + pad))

        if x1 <= x0 or y1 <= y0:
            return np.empty((0, 0), dtype=np.uint8)

        img = self.screen_array[y0:y1, x0:x1].copy()
        return img

if __name__ == "__main__":
    scan = ScanTable()
    print(scan.test_scan())
    

    import cv2
    import numpy as np
    from PIL import Image

    img = scan.screen_array  # BGR

    if isinstance(img, np.ndarray):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).show()
    elif isinstance(img, Image.Image):
        img.show()
    else:
        print("Type d'image inattendu:", type(img))
        