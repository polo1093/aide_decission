POSITION_POINT_WHITE = [50,20]
SIZE_CARD_NUMBER= [65,55]
SIZE_CARD_SYMBOL= [50,55,55] #[largeur, decalage hauteur, hauteur]
SIZE_POT = [75,40]
SIZE_FOND = [95,40]# [largeur, hauteur]
SIZE_PLAYER_MONEY = [95, 40]      # [largeur, hauteur]
SIZE_BUTTON = [165,70]
POSITION_CARD_RELATIF = {'board': [(-274, -607), (-131, -607), (13, -607), (158, -607), (303, -607)],
                        'me_card': [(-54, -186), (80, -186)]}
POSITION_MONEY_RELATIF = {'pot': [(102, -657)], 'fond': [(17, 52)], 'bouton_1': [(284, 65)], 'bouton_2': [(521, 65)], 'bouton_3': [(764, 65)]}
POSITION_MONEY_PLAYER = {'J1': [(-684, -211)], 'J2': [(-617, -669)], 'J3': [(21, -866)], 'J4': [(660, -669)], 'J5': [(726, -211)]}
TIMER_SCAN_REFRESH = 0.5


# Chemin vers le fichier de configuration des coordonnées
import json
import os
from typing import Dict, List, Optional

COORDINATES_FILE = os.path.join(os.path.dirname(__file__), "coordinates.json")
TABLE_CAPTURE_FILE = os.path.join(os.path.dirname(__file__), "table_capture.json")


def load_coordinates(path: str = COORDINATES_FILE) -> Dict[str, Dict[str, object]]:
    """Load coordinates configuration from ``path``.

    Returns a dictionary mapping region names to their coordinate info.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def coordinate_bounds(table: Optional[Dict[str, Dict[str, object]]] = None) -> List[int]:
    """Compute the minimal bounding box covering every ``coord_rel`` entry."""

    if table is None:
        table = TABLE

    xs_min: List[int] = []
    ys_min: List[int] = []
    xs_max: List[int] = []
    ys_max: List[int] = []

    for info in table.values():
        coord_rel = info.get("coord_rel") if isinstance(info, dict) else None
        if not coord_rel:
            continue
        x1, y1, x2, y2 = coord_rel
        xs_min.append(int(x1))
        ys_min.append(int(y1))
        xs_max.append(int(x2))
        ys_max.append(int(y2))

    if not xs_min:
        return [0, 0, 0, 0]

    return [min(xs_min), min(ys_min), max(xs_max), max(ys_max)]


def load_table_capture(
    path: str = TABLE_CAPTURE_FILE,
    table: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, object]:
    """Load table capture configuration and inject sensible defaults."""

    config: Dict[str, object] = {
        "enabled": False,
        "relative_bounds": None,
        "scale": 1.0,
        "padding": 0,
    }

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        config.update(loaded)

    if config.get("relative_bounds") is None:
        config["relative_bounds"] = coordinate_bounds(table)

    return config


# Dictionnaire des régions (chargé depuis ``coordinates.json``)
TABLE = load_coordinates()

# Configuration d'extraction visuelle (permet de recadrer les captures)
TABLE_CAPTURE = load_table_capture(table=TABLE)
