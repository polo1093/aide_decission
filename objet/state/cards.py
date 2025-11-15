"""Gestion de l'état des cartes de la table.

- 5 cartes de board
- 2 cartes pour le héros
- Coordonnées injectées à la déclaration à partir de coordinates.json.
"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from objet.entities.card import Card
from objet.utils.calibration import Region, load_coordinates , bbox_from_region

CardBox = Tuple[int, int, int, int]

# Même défaut que dans _utils.load_coordinates
DEFAULT_COORD_PATH = Path("config/PMU/coordinates.json")




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

        regions, templates_resolved, _ = load_coordinates(self.coord_path)


        if not self.board:
            self.board = [
                Card(
                    card_coordinates_value=bbox_from_region(regions.get(f"board_card_{i}_number")),
                    card_coordinates_suit=bbox_from_region(regions.get(f"board_card_{i}_symbol")),
                    template_set=_template_set_for_card(
                        regions,
                        f"board_card_{i}_number",
                        f"board_card_{i}_symbol",
                    ),
                )
                for i in range(1, 6)
            ]

        if not self.me:
            self.me = [
                Card(
                    card_coordinates_value=bbox_from_region(regions.get(f"player_card_{i}_number")),
                    card_coordinates_suit=bbox_from_region(regions.get(f"player_card_{i}_symbol")),
                    template_set=_template_set_for_card(
                        regions,
                        f"player_card_{i}_number",
                        f"player_card_{i}_symbol",
                    ),
                )
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
            
if __name__ == "__main__":
    # Petit stub de test local
    cards_state = CardsState()
    print(cards_state.me[0])

__all__ = ["CardsState"]
def _template_set_from_region(region: Optional[Region]) -> Optional[str]:
    if region is None:
        return None
    value = region.meta.get("template_set")
    if value is None:
        return None
    value_str = str(value).strip()
    return value_str or None


def _template_set_for_card(
    regions: Dict[str, Region],
    number_key: str,
    symbol_key: str,
) -> Optional[str]:
    tpl = _template_set_from_region(regions.get(number_key))
    if tpl:
        return tpl
    return _template_set_from_region(regions.get(symbol_key))

