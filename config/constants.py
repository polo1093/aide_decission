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
from typing import Dict, Tuple

COORDINATES_FILE = os.path.join(os.path.dirname(__file__), "coordinates.json")


def load_configuration(path: str = COORDINATES_FILE) -> Tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    """Charge la configuration complète et renvoie les régions et le crop."""

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    regions = data.get("regions", {})
    capture = data.get("table_capture", {}) or {}
    if "enabled" not in capture:
        capture["enabled"] = False

    return regions, capture


def load_coordinates(path: str = COORDINATES_FILE) -> Dict[str, Dict[str, object]]:
    """Renvoie la configuration des zones (coordonnées relatives)."""

    regions, _ = load_configuration(path)
    return regions


def load_table_capture(path: str = COORDINATES_FILE) -> Dict[str, object]:
    """Renvoie la configuration du crop de la table."""

    _, capture = load_configuration(path)
    return capture


# Dictionnaire des régions (chargé depuis ``coordinates.json``)
TABLE, TABLE_CAPTURE = load_configuration()
