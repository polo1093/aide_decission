
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import ImageGrab, Image
import logging

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from objet.utils.calibration import load_coordinates
from objet.utils.pyauto import locate_in_image
from objet.scanner.cards_recognition import TemplateIndex, is_card_present, recognize_number_and_suit

DEFAULT_COORD_PATH = Path("config/PMU/coordinates.json")
DEFAULT_CARDS_ROOT = Path("config/PMU/Cards")


class ScanTable:
    """Scan de la table PMU basé sur capture écran + pyautogui.

    - Localisation de la fenêtre via un template d'ancre (me.png) avec locate_in_image().
    - Utilisation de (size, ref_offset) issus de coordinates.json pour reconstruire le crop.
    - screen_array / screen_crop en BGR (convention OpenCV).
    """

    def __init__(self) -> None:
        # --- Config / calibration ---
        self.coord_path = DEFAULT_COORD_PATH
        _, _, table_capture = load_coordinates(self.coord_path)

        size_list = table_capture.get("size")
        ref_list = table_capture.get("ref_offset")

        if not size_list or not ref_list:
            raise ValueError(f"Invalid table_capture in {self.coord_path}: {table_capture}")

        self.size_crop: Tuple[int, int] = (int(size_list[0]), int(size_list[1]))
        self.offset_ref: Tuple[int, int] = (int(ref_list[0]), int(ref_list[1]))

        # Gabarit de référence (ancre) utilisé par pyautogui/locate
        self.reference_pil: Image.Image = Image.open("config/PMU/me.png").convert("RGB")

        # --- État runtime ---
        self.screen_array: Optional[np.ndarray] = None     # plein écran, BGR
        self.screen_crop: Optional[np.ndarray] = None      # crop table, BGR
        self.table_origin: Optional[Tuple[int, int]] = None
        self.scan_string: str = "init"
        self.cards_root = DEFAULT_CARDS_ROOT
        self.template_index = TemplateIndex(self.cards_root)
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

    # ------------------------------------------------------------------
    # Localisation de la table via pyautogui
    # ------------------------------------------------------------------
    def find_table(self, *, grayscale: bool = True, confidence: float = 0.9) -> bool:
        """Localise la table via l'ancre + (size, ref_offset).

        Remplit :
          - self.screen_crop : crop couleur BGR de la table
          - self.table_origin : (x0, y0) top-left sur l'écran
          - self.scan_string : 'ok' ou "don't find".
        """
        if self.screen_array is None:
            self.scan_string = "no_screen"
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
            self.screen_crop = None
            self.table_origin = None
            return False

        anchor_left, anchor_top, anchor_w, anchor_h = box
        W, H = self.size_crop
        ox, oy = self.offset_ref

        # 2) Calcul du coin haut-gauche de la fenêtre de table
        x0 = int(anchor_left - ox)
        y0 = int(anchor_top - oy)

        # Clamp dans les bornes de l'écran
        h_scr, w_scr = self.screen_array.shape[:2]
        x0 = max(0, min(x0, w_scr - 1))
        y0 = max(0, min(y0, h_scr - 1))
        x1 = max(0, min(x0 + W, w_scr))
        y1 = max(0, min(y0 + H, h_scr))

        if x1 <= x0 or y1 <= y0:
            self.scan_string = "invalid_crop"
            self.screen_crop = None
            self.table_origin = None
            return False

        # 3) Crop BGR pour le reste du pipeline
        self.screen_crop = self.screen_array[y0:y1, x0:x1].copy()
        self.table_origin = (x0, y0)
        self.scan_string = "ok"
        return True

    # ------------------------------------------------------------------
    # Scan des cartes dans la table (identique à ta version, basé sur screen_crop)
    # ------------------------------------------------------------------
    def scan_carte(self, position_value: Tuple[int, int, int, int],position_suit ) -> Tuple[Optional[str], Optional[str], float, float]:
        """
        Retourne:
            (value, suit, confidence_value, confidence_suit)

        - value, suit : str ou None
        - confidence_* : float entre 0.0 et 1.0
        """

        h_img, w_img = self.screen_crop.shape[:2]


            

        # crops séparés pour la valeur et le symbole
        image_card_value = self._crop_box_gray(position_value)
        image_card_suit = self._crop_box_gray(position_suit)

        
        # rgb = cv2.cvtColor(image_card_value, cv2.COLOR_BGR2RGB)
        # Image.fromarray(rgb).show()
        
        if is_card_present(image_card_value):
            carte_value, carte_suit, score_value, score_suit = recognize_number_and_suit(image_card_value,image_card_suit,self.template_index) # manque un argument  sans dout pour la suit

            # Si ta fonction de reco ne retourne pas de score, on considère confidence = 1.0
            conf_val = 1.0 if carte_value is not None else 0.0
            conf_suit = 1.0 if carte_suit is not None else 0.0
            return carte_value, carte_suit, conf_val, conf_suit
        return None, None, 0.0, 0.0


 
    
    # Stubs à compléter plus tard
    def scan_pot(self, position):
        _ = self.screen_crop
        return None

    def scan_player(self, position):
        _ = self.screen_crop
        return None, None

    def scan_money_player(self, position):
        _ = self.screen_crop
        return None

    def scan_bouton(self, position):
        _ = self.screen_crop
        return None, None


    def _crop_box_gray(self,box, pad=3):
        """Retourne un crop (numpy BGR) pour box=(x,y,w,h) avec padding et clamp."""
        x, y, w, h = box
        x0 = max(0, int(x - pad))
        y0 = max(0, int(y - pad))
        x1 = min(self.screen_crop.shape[1], int(x + w + pad))
        y1 = min(self.screen_crop.shape[0], int(y + h + pad))
        img = self.screen_crop[y0:y1, x0:x1].copy()
        return img

if __name__ == "__main__":
    scan = ScanTable()
    print(scan.test_scan())
    

    import cv2
    import numpy as np
    from PIL import Image

    img = scan.screen_crop  # BGR

    if isinstance(img, np.ndarray):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).show()
    elif isinstance(img, Image.Image):
        img.show()
    else:
        print("Type d'image inattendu:", type(img))
        