from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import ImageGrab, Image
import logging

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts._utils import load_coordinates
from scripts.pyauto_helpers import locate_in_image
from scripts.capture_cards import TemplateIndex, is_card_present, recognize_number_and_suit

DEFAULT_COORD_PATH = Path("config/PMU/coordinates.json")


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
    def scan_carte(self, position):
        """
        Retourne:
            (value, suit, confidence_value, confidence_suit)

        - value, suit : str ou None
        - confidence_* : float entre 0.0 et 1.0
        """
        # Toujours renvoyer 4 valeurs
        if self.screen_crop is None or position is None:
            return None, None, 0.0, 0.0

        try:
            x, y, w, h = position
        except Exception:
            logging.warning("scan_carte: position attendue comme (x, y, w, h)")
            return None, None, 0.0, 0.0

        pad = 3
        h_img, w_img = self.screen_crop.shape[:2]

        x0 = max(0, int(x - pad))
        y0 = max(0, int(y - pad))
        x1 = min(w_img, int(x + w + pad))
        y1 = min(h_img, int(y + h + pad))

        if x0 >= x1 or y0 >= y1:
            return None, None, 0.0, 0.0

        image_card = self.screen_crop[y0:y1, x0:x1]

        # Conversion vers le gris pour la détection
        if image_card.ndim == 3:
            image_card_gray = cv2.cvtColor(image_card, cv2.COLOR_BGR2GRAY)
        else:
            image_card_gray = image_card

        # ------------------------
        # Voie principale OCR cartes
        # ------------------------
        try:
            if is_card_present(image_card_gray):
                carte_value, carte_suit = recognize_number_and_suit(image_card_gray)

                # Si ta fonction de reco ne retourne pas de score, on considère confidence = 1.0
                conf_val = 1.0 if carte_value is not None else 0.0
                conf_suit = 1.0 if carte_suit is not None else 0.0
                return carte_value, carte_suit, conf_val, conf_suit

        except Exception as e:
            logging.exception("Erreur lors de la reconnaissance de la carte: %s", e)

        # ------------------------
        # Fallback TemplateIndex
        # ------------------------
        try:
            best_val = None
            best_suit = None
            best_score_val = 0.0
            best_score_suit = 0.0

            nums = None
            suits = None
            if hasattr(TemplateIndex, "number_templates") and hasattr(TemplateIndex, "suit_templates"):
                nums = TemplateIndex.number_templates
                suits = TemplateIndex.suit_templates
            elif hasattr(TemplateIndex, "templates"):
                nums = {}
                suits = {}
                for k, tmpl in TemplateIndex.templates.items():
                    if "_" in k:
                        val, su = k.split("_", 1)
                        nums.setdefault(val, []).append(tmpl)
                        suits.setdefault(su, []).append(tmpl)
                    else:
                        nums.setdefault(k, []).append(tmpl)
            elif hasattr(TemplateIndex, "get_templates"):
                t = TemplateIndex.get_templates()
                if isinstance(t, dict):
                    nums = {}
                    suits = {}
                    for k, tmpl in t.items():
                        if "_" in k:
                            val, su = k.split("_", 1)
                            nums.setdefault(val, []).append(tmpl)
                            suits.setdefault(su, []).append(tmpl)
                        else:
                            nums.setdefault(k, []).append(tmpl)

            if not nums:
                raise RuntimeError("No templates available for fallback recognition")

            # Matcher les valeurs
            for val, tlist in nums.items():
                for tmpl in (tlist if isinstance(tlist, list) else [tlist]):
                    if tmpl is None:
                        continue
                    tmpl_gray = tmpl if tmpl.ndim == 2 else cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
                    if tmpl_gray.shape[0] > image_card_gray.shape[0] or tmpl_gray.shape[1] > image_card_gray.shape[1]:
                        continue
                    res = cv2.matchTemplate(image_card_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
                    _, maxv, _, _ = cv2.minMaxLoc(res)
                    if maxv > best_score_val:
                        best_score_val = maxv
                        best_val = val

            # Matcher les couleurs/symboles si on en a
            if suits:
                for suit, tlist in suits.items():
                    for tmpl in (tlist if isinstance(tlist, list) else [tlist]):
                        if tmpl is None:
                            continue
                        tmpl_gray = tmpl if tmpl.ndim == 2 else cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
                        if tmpl_gray.shape[0] > image_card_gray.shape[0] or tmpl_gray.shape[1] > image_card_gray.shape[1]:
                            continue
                        res = cv2.matchTemplate(image_card_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
                        _, maxv, _, _ = cv2.minMaxLoc(res)
                        if maxv > best_score_suit:
                            best_score_suit = maxv
                            best_suit = suit

            # Seuils minimaux pour accepter le fallback
            if best_val and best_suit and best_score_val > 0.4 and best_score_suit > 0.3:
                logging.warning(
                    "Fallback: carte approchée détectée %s de %s (scores %.2f / %.2f)",
                    best_val,
                    best_suit,
                    best_score_val,
                    best_score_suit,
                )
                return best_val, best_suit, float(best_score_val), float(best_score_suit)

            elif best_val and best_score_val > 0.45 and not best_suit:
                logging.warning(
                    "Fallback: valeur approchée détectée %s (score %.2f)",
                    best_val,
                    best_score_val,
                )
                return best_val, None, float(best_score_val), 0.0

        except Exception:
            logging.debug("Fallback template-matching a échoué", exc_info=True)

        # Si tout a échoué
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
        
    img = scan.screen_crop  # BGR

    if isinstance(img, np.ndarray):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).show()
    elif isinstance(img, Image.Image):
        img.show()
    else:
        print("Type d'image inattendu:", type(img))
        