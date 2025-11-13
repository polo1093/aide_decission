import cv2
import numpy as np
import logging
from typing import Dict, Tuple

from folder_tool import timer



# Import des fonctions utilitaires depuis le dossier scripts
try:
    from scripts.crop_core import find_ref_point,crop_from_size_and_offset
except ImportError:
    # Fallback au cas où le script ne serait pas dans le path
    find_ref_point = None
from scripts.capture_cards import TemplateIndex, is_card_present, recognize_number_and_suit
from scripts.crop_core import crop_from_size_and_offset

class ScanTable():
    def __init__(self):

        self.screen_array = np.array(ImageGrab.grab())
        self.screen_reference = cv2.imread('screen/launch/me.png', 0) #paht dans PMU par défaut mais  en passer paramètre
        self.screen_crop = None
        
        
        self.TIMER_SCAN_REFRESH = 0.5 
        self.timer_screen = timer.Timer(self.TIMER_SCAN_REFRESH)
        self.table_origin = None                # (x0, y0, x1, y1) absolu sur le screen global
        self.scan_string = "don t find"


    def test_scan(self,debug=False):
        while(self.screen_refresh==False):
            wait(0.1)
        if self.screen_refresh():
            if  self.find_table():
                 return True
            self.scan_string = "don't find"
            return False
       
            

        
    def screen_refresh(self):
        if self.timer_screen.is_expire():
            self.timer_screen.refresh()
            self.screen_array = np.array(ImageGrab.grab())
            return True
        return False
    
    def find_table(self):
        """Trouve la table en utilisant la fonction de template matching du module crop_core."""
        

        self.screen_crop,self.table_origin=  crop_from_size_and_offset()
        if self.table_origin is  None:
            self.scan_string = "don't find"
            return False
        return True
   
    
      

     
     
    def scan_carte(self,  position):
        """
        Trouve la carte dans un crop donné.
        - image_crop : numpy array (image) contenant la table ou la zone globale.
        - position : tuple (x, y, w, h) décrivant la position de la carte dans image_crop.
        Retourne (valeur_carte, couleur_carte) ou (None, None) si aucune carte détectée.
        #j'aimerai rajouter en sortie la confiance des reconnaissances 
        """
        if  self.screen_crop is None or position is None:
            return None, None

        try:
            x, y, w, h = position
        except Exception:
            logging.warning("scan_carte: position attendue comme (x, y, w, h)")
            return None, None

        pad = 3  # 3 pixels de marge autour de la carte
        h_img, w_img = self.screen_crop.shape[:2]

        x0 = max(0, int(x - pad))
        y0 = max(0, int(y - pad))
        x1 = min(w_img, int(x + w + pad))
        y1 = min(h_img, int(y + h + pad))

        if x0 >= x1 or y0 >= y1:
            return None, None

        image_card = self.screen_crop[y0:y1, x0:x1]

        # Convertir en gris si nécessaire (les utilitaires peuvent attendre du gris)
        if image_card.ndim == 3:
            image_card_gray = cv2.cvtColor(image_card, cv2.COLOR_BGR2GRAY)
        else:
            image_card_gray = image_card

        try:
            if is_card_present(image_card_gray):
                carte_value, carte_suit = recognize_number_and_suit(image_card_gray)
                return carte_value, carte_suit
        except Exception as e:
            logging.exception("Erreur lors de la reconnaissance de la carte: %s", e)
            # Fallback best-effort: template matching against templates in TemplateIndex (si disponible)
            try:
                best_val = None
                best_suit = None
                best_score_val = 0.0
                best_score_suit = 0.0

                # Récupère des collections possibles de templates depuis TemplateIndex (différents APIs possibles)
                nums = None
                suits = None
                if hasattr(TemplateIndex, "number_templates") and hasattr(TemplateIndex, "suit_templates"):
                    nums = TemplateIndex.number_templates
                    suits = TemplateIndex.suit_templates
                elif hasattr(TemplateIndex, "templates"):
                    # tente d'interpréter templates comme dict "VAL_SUIT" -> image ou VAL->image
                    nums = {}
                    suits = {}
                    for k, tmpl in TemplateIndex.templates.items():
                        if "_" in k:
                            val, su = k.split("_", 1)
                            nums.setdefault(val, []).append(tmpl)
                            suits.setdefault(su, []).append(tmpl)
                        else:
                            # pas d'info de suit/val ; met dans nums au cas où
                            nums.setdefault(k, []).append(tmpl)
                elif hasattr(TemplateIndex, "get_templates"):
                    t = TemplateIndex.get_templates()
                    if isinstance(t, dict):
                        # même logique que ci-dessus
                        nums = {}
                        suits = {}
                        for k, tmpl in t.items():
                            if "_" in k:
                                val, su = k.split("_", 1)
                                nums.setdefault(val, []).append(tmpl)
                                suits.setdefault(su, []).append(tmpl)
                            else:
                                nums.setdefault(k, []).append(tmpl)

                # Si on n'a pas de templates, on abandonne le fallback
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

                # Seuils minimal pour accepter le fallback (ajuster si nécessaire)
                if best_val and best_suit and best_score_val > 0.4 and best_score_suit > 0.3:
                    logging.warning("Fallback: carte approchée détectée %s de %s (scores %.2f / %.2f)", best_val, best_suit, best_score_val, best_score_suit)
                    return best_val, best_suit
                elif best_val and best_score_val > 0.45 and not best_suit:
                    # on a au moins la valeur
                    logging.warning("Fallback: valeur approchée détectée %s (score %.2f)", best_val, best_score_val)
                    return best_val, None
            except Exception:
                logging.debug("Fallback template-matching a échoué", exc_info=True)

            # Si tout échoue, on retourne None, None

        return None, None


    def scan_pot(self, image crop globale, position du pot):
        return pot_value
    
    
    def scan_player(self, image crop globale, position du joueur):  
        scan_money_player
        return player_state,player_active
    
    def scan_money_player(self, image crop globale, position de la money):
        return money_value
        
    def scan_bouton(self, image crop globale, position du bouton):
        # need ocr and traitement de texte
        return money_value, texte