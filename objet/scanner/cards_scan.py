# scanner/cards.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, Callable
import json
from pathlib import Path

try:
    # Utilitaires centralisés de calibration/extraction (si présents)
    # (cf. README: scripts/_utils.py)
    from scripts import _utils as util  # type: ignore
except Exception:
    util = None

try:
    # Si tu as une fonction de reco prête dans tes scripts:
    # ex. recognize_number_and_suit(number_img, suit_img) -> (value, suit, score_v, score_s)
    from scripts.capture_cards import recognize_number_and_suit as _rec_cards  # type: ignore
except Exception:
    _rec_cards = None

try:
    # Alternative: si identify_card expose une API similaire
    from scripts.identify_card import recognize_number_and_suit as _rec_cards_alt  # type: ignore
except Exception:
    _rec_cards_alt = None

# ---------------------------------------------------------------------
# Helpers “safe”
# ---------------------------------------------------------------------

def _load_coordinates() -> Dict[str, Any]:
    """
    Charge coordinates.json :
    - Priorité au loader commun (scripts/_utils.py) s'il existe
    - Sinon charge depuis config/coordinates.json
    """
    if util and hasattr(util, "load_coordinates"):
        return util.load_coordinates()
    # Fallback local
    coord_paths = [
        Path("config/coordinates.json"),
        Path("coordinates.json"),  # selon ce que tu utilises actuellement
    ]
    for p in coord_paths:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError("coordinates.json introuvable")

def _crop_numpy(img_np, rel_box, table_abs):
    """
    Découpe un patch numpy à partir:
    - img_np: full frame (numpy HxWxC ou PIL converti vers np)
    - rel_box: [x1, y1, x2, y2] relatif à la table (0..1 ou coord relatives pixels table ?)
    - table_abs: [X1, Y1, X2, Y2] bbox absolue de la table sur l'écran
    """
    import numpy as np
    X1, Y1, X2, Y2 = table_abs
    w = max(0, X2 - X1)
    h = max(0, Y2 - Y1)
    # Support coordonnées relatives 0..1
    if all(0.0 <= v <= 1.0 for v in rel_box):
        x1 = int(X1 + rel_box[0] * w)
        y1 = int(Y1 + rel_box[1] * h)
        x2 = int(X1 + rel_box[2] * w)
        y2 = int(Y1 + rel_box[3] * h)
    else:
        # Sinon supposées relatives en pixels table
        x1 = X1 + int(rel_box[0])
        y1 = Y1 + int(rel_box[1])
        x2 = X1 + int(rel_box[2])
        y2 = Y1 + int(rel_box[3])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
    return img_np[y1:y2, x1:x2, :]

def _to_pil(img_np):
    from PIL import Image
    return Image.fromarray(img_np)

def _grab_frame_np() -> "np.ndarray":
    from PIL import ImageGrab
    import numpy as np
    # full screen
    pil = ImageGrab.grab()
    return np.array(pil)

def _get_table_bbox(coords: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    """
    Tente de récupérer la bbox absolue de la table.
    Idéalement, tu as déjà KEY type 'table_capture' dans coordinates.json.
    Sinon, branche ici ton “find_table()”.
    """
    tcap = coords.get("table_capture")
    if not tcap:
        return None
    # Support absolu: [X1,Y1,X2,Y2]
    if len(tcap) == 4 and max(tcap) > 1.0:
        return tuple(int(v) for v in tcap)  # type: ignore
    # Support relatif à l'écran (0..1)
    from PIL import ImageGrab
    w, h = ImageGrab.grab().size
    X1 = int(tcap[0] * w); Y1 = int(tcap[1] * h); X2 = int(tcap[2] * w); Y2 = int(tcap[3] * h)
    return (X1, Y1, X2, Y2)

def _recognize(number_img_pil, symbol_img_pil) -> Tuple[Optional[str], Optional[str]]:
    """
    Appelle ta reco existante si dispo (scripts/capture_cards.py ou scripts/identify_card.py),
    sinon renvoie (None, None) proprement (pas de crash).
    """
    if _rec_cards:
        try:
            val, suit, *_scores = _rec_cards(number_img_pil, symbol_img_pil)  # type: ignore
            return (val, suit)
        except Exception:
            pass
    if _rec_cards_alt:
        try:
            val, suit, *_scores = _rec_cards_alt(number_img_pil, symbol_img_pil)  # type: ignore
            return (val, suit)
        except Exception:
            pass
    return (None, None)

# ---------------------------------------------------------------------
# Scanner Cartes minimal (adapter en wrapper Table si tu veux)
# ---------------------------------------------------------------------

CARD_KEYS = {
    "player": [
        ("player_card_1_number", "player_card_1_symbol"),
        ("player_card_2_number", "player_card_2_symbol"),
    ],
    "board": [
        ("board_card_1_number", "board_card_1_symbol"),
        ("board_card_2_number", "board_card_2_symbol"),
        ("board_card_3_number", "board_card_3_symbol"),
        ("board_card_4_number", "board_card_4_symbol"),
        ("board_card_5_number", "board_card_5_symbol"),
    ],
}

class TableScanner:
    """
    Scanner minimal qui ne gère QUE les cartes.
    - capture écran
    - crop table
    - crop patches number/symbol via coordinates.json
    - reco via scripts existants si dispos
    Retourne un scan_table dict (clés standard) ; boutons/joueurs/pot/fond restent vides.
    """
    def __init__(self):
        self.coords = _load_coordinates()

    def scan(self) -> Dict[str, Dict[str, Any]]:
        import numpy as np
        scan_table: Dict[str, Dict[str, Any]] = {}
        try:
            frame = _grab_frame_np()  # (H,W,3)
            table_bbox = _get_table_bbox(self.coords)
            if table_bbox is None:
                # pas de table -> on sort proprement
                return scan_table

            # --- Player cards ---
            for num_key, sym_key in CARD_KEYS["player"]:
                rel_num = self.coords.get(num_key)
                rel_sym = self.coords.get(sym_key)
                val = suit = None
                if rel_num and rel_sym:
                    patch_n = _crop_numpy(frame, rel_num, table_bbox)
                    patch_s = _crop_numpy(frame, rel_sym, table_bbox)
                    if patch_n.size and patch_s.size:
                        val, suit = _recognize(_to_pil(patch_n), _to_pil(patch_s))
                scan_table[num_key] = {"value": val}
                scan_table[sym_key] = {"value": suit}

            # --- Board cards ---
            for num_key, sym_key in CARD_KEYS["board"]:
                rel_num = self.coords.get(num_key)
                rel_sym = self.coords.get(sym_key)
                val = suit = None
                if rel_num and rel_sym:
                    patch_n = _crop_numpy(frame, rel_num, table_bbox)
                    patch_s = _crop_numpy(frame, rel_sym, table_bbox)
                    if patch_n.size and patch_s.size:
                        val, suit = _recognize(_to_pil(patch_n), _to_pil(patch_s))
                scan_table[num_key] = {"value": val}
                scan_table[sym_key] = {"value": suit}

            # --- Stubs neutres pour compat UI (optionnels) ---
            for i in range(1, 4):
                scan_table[f"button_{i}"] = {"value": None}
            scan_table["pot"] = {"value": None}
            scan_table["fond"] = {"value": None}
            for i in range(1, 6):
                scan_table[f"player_money_J{i}"] = {"value": None}

            return scan_table
        except Exception:
            # jamais faire planter l’UI : renvoyer un dict vide
            return {}
