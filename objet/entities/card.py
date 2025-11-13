"""Entité unique représentant une carte scannée et sa normalisation."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

from pokereval.card import Card as PokerCard

LOGGER = logging.getLogger(__name__)

SUIT_ALIASES = {
    "hearts": "\u2665",
    "diamonds": "\u2666",
    "spades": "\u2660",
    "clubs": "\u2663",
    "heart": "\u2665",
    "diamond": "\u2666",
    "spade": "\u2660",
    "club": "\u2663",
}


@dataclass
class Card:
    """Observation d'une carte et conversion vers l'objet PokerCard."""
    card_coordinates: Optional[tuple[int, int, int, int]] = [0,0,0,0]
    value: Optional[str] = None
    suit: Optional[str] = None
    value_score: Optional[float] = None
    suit_score: Optional[float] = None
    poker_card: Optional[PokerCard] = None

    def formatted(self) -> Optional[str]:
        """Chaîne human-readable de la forme '10♥' ou 'A♠', ou None si incomplet."""
        if not self.value or not self.suit:
            return None
        suit_sym = SUIT_ALIASES.get(self.suit, self.suit)
        return f"{self.value}{suit_sym}"

    
    def scan(self) -> Optional[PokerCard]:
        
        
        
        
        return self.value, self.suit
    
    
    
    def apply_observation(
        self,
        value: Optional[str],
        suit: Optional[str],
        value_score: Optional[float] = None,
        suit_score: Optional[float] = None,
        source: str = "scan",
    ) -> None:
        """Applique une nouvelle observation et met à jour l'objet PokerCard."""
        self.value = value
        self.suit = suit
        self.value_score = value_score
        self.suit_score = suit_score
        formatted = self.formatted()
        self.poker_card = self._convert_string_to_pokercard(formatted) if formatted else None

    @staticmethod
    def _convert_string_to_pokercard(string_carte: Optional[str]) -> Optional[PokerCard]:
        """Convertit une chaîne '10♥' / 'A♠' en PokerCard (ou None si invalide)."""
        suit_dict = {"\u2666": 1, "\u2665": 2, "\u2660": 3, "\u2663": 4}
        value_dict = {
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "10": 10,
            "J": 11,
            "Q": 12,
            "K": 13,
            "A": 14,
        }

        if string_carte in (None, "", "_"):
            return None

        string_carte = string_carte.strip()
        if not string_carte:
            return None

        # correction éventuelle si le scanner a renvoyé '0' au lieu de '10' en première position
        if string_carte[0] == "0" and len(string_carte) >= 2:
            LOGGER.debug(
                "Debug : La carte spécifiée '%s' est modifiée en '10%s' pour correction.",
                string_carte,
            )
            string_carte = "10" + string_carte[1:]

        if len(string_carte) >= 2:
            value_part = string_carte[:-1]
            suit_part = string_carte[-1]
            if value_part in value_dict and suit_part in suit_dict:
                value = value_dict[value_part]
                suit = suit_dict[suit_part]
                return PokerCard(value, suit)
            else:
                LOGGER.debug("Debug : La carte spécifiée '%s' n'est pas reconnue.", string_carte)
                return None
        else:
            LOGGER.debug("Debug : La carte spécifiée '%s' est trop courte.", string_carte)
            return None

    
__all__ = ["Card"]
