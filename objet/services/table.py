"""Service d'orchestration autour de l'état de la table de jeu."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional


from objet.state import ButtonsState, CardsState, CaptureState, extract_scan_value
from objet.scanner.scan import ScanTable


@dataclass
class Table:
    """Réunit cartes, boutons et informations de capture."""

    cards: CardsState = field(default_factory=CardsState)
    # buttons: ButtonsState = field(default_factory=ButtonsState)
    # players: list[Any] = field(default_factory=list)
    scan : ScanTable    = field(default_factory=ScanTable)


    def launch_scan(self):
        if self.scan.test_scan():

        
        
            for i in range(1, 6):
                value, suit, confidence_value, confidence_suit = self.scan.scan_carte(position=self.cards.card.board[i].card_coordinates)
                self.cards.card.board[i].apply_observation(
                    value=value,
                    suit=suit,
                    value_score=confidence_value,
                    suit_score=confidence_suit,
                )
              
            for i in range(1, 3):
                value, suit, confidence_value, confidence_suit = self.scan.scan_carte(position=self.cards.card.me[i].card_coordinates)
                self.cards.card.me[i].apply_observation(
                    value=value,
                    suit=suit,
                    value_score=confidence_value,
                    suit_score=confidence_suit,
                )
        # self.buttons.update_from_scan(scan_table)

   


__all__ = ["Table"]
