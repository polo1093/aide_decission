"""Service d'orchestration autour de l'état de la table de jeu."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from objet.state import ButtonsState, CaptureState, CardsState
from objet.scanner.scan import ScanTable


@dataclass
class Table:
    """Réunit Les éléments à scanner et service de scan."""

    cards: CardsState = field(default_factory=CardsState)
    buttons: ButtonsState = field(default_factory=ButtonsState)
    captures: CaptureState = field(default_factory=CaptureState)
    scan: ScanTable = field(default_factory=ScanTable)
    new_party_flag: bool = False

    def launch_scan(self) -> bool:
       
        if not self.scan.test_scan():
            return False
        
        # --- Main héros (2 cartes) ---
        for idx, card in enumerate(self.cards.me, start=1):
            value, suit, confidence_value, confidence_suit = self.scan.scan_carte(
                position=card.card_coordinates
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
        
        # --- Board (5 cartes) ---
        # On suppose que `cards.board` est indexable 0..4
        for idx, card in enumerate(self.cards.board, start=1):
            if card.formatted is None:
                value, suit, confidence_value, confidence_suit = self.scan.scan_carte(
                    position=card.card_coordinates
                )
                if value is None and suit is None:
                    continue
                card.apply_observation(
                    value=value,
                    suit=suit,
                    value_score=confidence_value,
                    suit_score=confidence_suit,
                )


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
