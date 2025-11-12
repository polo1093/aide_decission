"""Entités liées aux cartes et utilitaires de conversion."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

from pokereval.card import Card

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
class CardObservation:
    """Observation d'une carte issue d'un scan ou d'une capture."""

    value: Optional[str]
    suit: Optional[str]
    value_score: Optional[float] = None
    suit_score: Optional[float] = None
    source: str = "scan"

    def formatted(self) -> Optional[str]:
        if not self.value or not self.suit:
            return None
        suit = SUIT_ALIASES.get(self.suit, self.suit)
        return f"{self.value}{suit}"


@dataclass
class CardSlot:
    """Carte normalisée stockée dans l'état courant."""

    observation: Optional[CardObservation] = None
    card: Optional[Card] = None

    def apply(self, observation: CardObservation) -> None:
        self.observation = observation
        formatted = observation.formatted()
        self.card = convert_card(formatted) if formatted else None


def convert_card(string_carte: Optional[str]) -> Optional[Card]:
    """Convertit une chaîne représentant une carte de poker en objet :class:`Card`."""

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

    if string_carte[0] == "0":
        message = (
            f"Debug : La carte spécifiée '{string_carte}' est modifiée en '10{string_carte[1:]}' pour correction."
        )
        LOGGER.debug(message)
        string_carte = "10" + string_carte[1:]

    if len(string_carte) >= 2:
        value_part = string_carte[:-1]
        suit_part = string_carte[-1]
        if value_part in value_dict and suit_part in suit_dict:
            value = value_dict[value_part]
            suit = suit_dict[suit_part]
        else:
            LOGGER.debug("Debug : La carte spécifiée '%s' n'est pas reconnue.", string_carte)
            return None
    else:
        LOGGER.debug("Debug : La carte spécifiée '%s' est trop courte.", string_carte)
        return None

    return Card(value, suit)


    class card_scan:
        def __init__(self):
        
        
        
        utilise  les fonctions dans scripts pour retourner les valeurs des cartes
        
        




__all__ = [
    "CardObservation",
    "CardSlot",
    "convert_card",
]


