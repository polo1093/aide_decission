# objet/state/cards_state.py
"""Gestion de l'état des cartes de la table.

- 5 cartes de board
- 2 cartes pour le héros
- Coordonnées injectées à la déclaration à partir de coordinates.json.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from objet.entities.card import Card
from objet.utils.calibration import Region, load_coordinates

CardBox = Tuple[int, int, int, int]

# Même défaut que dans _utils.load_coordinates
DEFAULT_COORD_PATH = Path("config/PMU/coordinates.json")


def _card_box_from_regions(regions: Dict[str, Region], base_key: str) -> Optional[CardBox]:
    """
    Calcule le bounding box global pour une carte à partir de :

      - <base_key>_number
      - <base_key>_symbol

    en fusionnant les 2 rectangles (valeur + symbole).
    """
    keys = [f"{base_key}_number", f"{base_key}_symbol"]
    boxes: List[CardBox] = []

    for key in keys:
        region = regions.get(key)
        if region is None:
            continue
        x, y = region.top_left
        w, h = region.size
        boxes.append((x, y, x + w, y + h))

    if not boxes:
        return None

    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return x1, y1, x2, y2


@dataclass
class CardsState:
    """
    Regroupe les cartes du board et du joueur, avec coordonnées injectées.

    - `coord_path` permet de surcharger le fichier de coordonnées si besoin
      (par défaut : config/PMU/coordinates.json).
    - Si `board` / `me` ne sont pas fournis, ils sont construits automatiquement
      à partir de `load_coordinates(coord_path)`.
    """

    coord_path: Path | str = DEFAULT_COORD_PATH
    board: List[Card] = field(default_factory=list)
    me: List[Card] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Cas où on injecte manuellement des cartes : on ne touche à rien.
        if self.board and self.me:
            return

        regions, _, _ = load_coordinates(self.coord_path)

        if not self.board:
            self.board = [
                Card(card_coordinates=_card_box_from_regions(regions, f"board_card_{i}"))
                for i in range(1, 6)
            ]

        if not self.me:
            self.me = [
                Card(card_coordinates=_card_box_from_regions(regions, f"player_card_{i}"))
                for i in range(1, 3)
            ]

    # --- API pratique pour le reste du code ----------------------------------

    def me_cards(self) -> List[Card]:
        """Retourne les entités Card du joueur (avec value/suit/poker_card)."""
        return self.me

    def board_cards(self) -> List[Card]:
        """Retourne les entités Card du board."""
        return self.board


        
    def reset(self) -> None:
        """Réinitialise l'état de toutes les cartes."""
        for card in self.me + self.board:
            card.reset()
            


__all__ = ["CardsState"]
