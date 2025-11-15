"""Service d'orchestration autour de l'état de la table de jeu."""


from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from objet.state import ButtonsState, CaptureState, CardsState
from objet.scanner.scan import ScanTable
from objet.utils.calibration import load_coordinates, bbox_from_region
DEFAULT_COORD_PATH = Path("config/PMU/coordinates.json")

@dataclass
class Fond:
    coordinates_value: Optional[tuple[int, int, int, int]] = None
    amount: float = 0.0
    
    def reset(self) -> None:
        self.amount = 0.0
    

@dataclass
class Player:
    coordinates_value: Optional[tuple[int, int, int, int]] = None
    active_start : bool = True
    continue_round : bool = True
    fond = Fond(coordinates_value=coordinates_value)
    
    def reset(self) -> None:
        self.amount = 0.0
    

@dataclass
class Table:
    """Réunit Les éléments à scanner et service de scan."""
    
    coord_path: Path | str = DEFAULT_COORD_PATH
    cards: CardsState = field(default_factory=CardsState)
    buttons: ButtonsState = field(default_factory=ButtonsState)
    captures: CaptureState = field(default_factory=CaptureState)
    scan: ScanTable = field(default_factory=ScanTable)
    pot: Fond = field(default_factory=Fond)
    new_party_flag: bool = False
    
    
    def __post_init__(self) -> None:
        regions, _, _ = load_coordinates(self.coord_path)
        self.pot.coordinates_value = bbox_from_region(regions.get("pot"))
        
        
    def launch_scan(self) -> bool:
       
        if not self.scan.test_scan():
            return False
        
        # --- Main héros (2 cartes) ---
        for card in self.cards.me:
            value, suit, confidence_value, confidence_suit = self.scan.scan_carte(
                position_value=card.card_coordinates_value,
                position_suit=card.card_coordinates_suit,
                template_set=card.template_set,
            )
            if value is not None or suit is not None:
                if value != card.value and suit != card.suit:
                    self.New_Party()
            card.apply_observation(
                value=value,
                suit=suit,
                value_score=confidence_value,
                suit_score=confidence_suit,
            )
        
        for card in self.cards.board:
            if card.formatted is None:
                value, suit, confidence_value, confidence_suit = self.scan.scan_carte(
                    position_value=card.card_coordinates_value,
                    position_suit=card.card_coordinates_suit,
                    template_set=card.template_set,
                )
                if value is None and suit is None:
                    continue
                card.apply_observation(
                    value=value,
                    suit=suit,
                    value_score=confidence_value,
                    suit_score=confidence_suit,
                )

                
        # self.scan.scan_pot()


        return True

    def New_Party(self)-> None:
        """Réinitialise l'état de la Table. et fait remonter un événement."""
        self.cards.reset()
        self.new_party_flag = True
        
    
        
if __name__ == "__main__":
    # Petit stub de test local
    table = Table()
    table.launch_scan()
    print("Cartes joueur:", table.cards.me_cards())
    

__all__ = ["Table"]
