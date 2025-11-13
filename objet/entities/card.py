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
    card_coordinates: Optional[tuple[int, int, int, int]] = None
    value: Optional[str] = None
    suit: Optional[str] = None
    value_score: Optional[float] = None
    suit_score: Optional[float] = None
    poker_card: Optional[PokerCard] = None
    formatted: Optional[str] = None


    def scan(self) -> tuple[Optional[str], Optional[str]]:
        """
        Retourne la valeur brute scannée (value, suit).

        Utile pour debugger le flux OCR avant conversion en PokerCard.
        """
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
        LOGGER.debug(
            "apply_observation(source=%s, value=%s, suit=%s, value_score=%s, suit_score=%s)",
            source,
            value,
            suit,
            value_score,
            suit_score,
        )
        self.value = value
        self.suit = suit
        self.value_score = value_score
        self.suit_score = suit_score
        formatted = self.formatted()
        if self.value and self.suit: 
            suit_sym = SUIT_ALIASES.get(self.suit, self.suit)
            formatted = f"{self.value}{suit_sym}"
            self.poker_card = self._convert_string_to_pokercard(formatted) 

    @staticmethod
    def _convert_string_to_pokercard(string_carte: Optional[str]) -> Optional[PokerCard]:
        """
        Convertit une chaîne '10♥' / 'A♠' en PokerCard (ou None si invalide).

        Mapping suits pokereval:
            1 -> spades (s)
            2 -> hearts (h)
            3 -> diamonds (d)
            4 -> clubs (c)
        """
        suit_dict = {
            "\u2660": 1,  # ♠
            "\u2665": 2,  # ♥
            "\u2666": 3,  # ♦
            "\u2663": 4,  # ♣
        }
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
            original = string_carte
            corrected = "10" + string_carte[1:]
            LOGGER.debug(
                "Debug : La carte spécifiée '%s' est modifiée en '%s' pour correction.",
                original,
                corrected,
            )
            string_carte = corrected

        if len(string_carte) >= 2:
            value_part = string_carte[:-1]
            suit_part = string_carte[-1]
            value = value_dict.get(value_part)
            suit = suit_dict.get(suit_part)
            if value is not None and suit is not None:
                return PokerCard(value, suit)
            LOGGER.debug("Debug : La carte spécifiée '%s' n'est pas reconnue.", string_carte)
            return None

        LOGGER.debug("Debug : La carte spécifiée '%s' est trop courte.", string_carte)
        return None


__all__ = ["Card"]


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")

    print("=== Tests manuels de Card ===")

    tests = [
        ("A", "hearts"),
        ("10", "spades"),
        ("J", "diamonds"),
        (None, "clubs"),   # valeur manquante
        ("Q", None),       # couleur manquante
    ]

    for idx, (val, suit) in enumerate(tests, start=1):
        c = Card()
        c.apply_observation(value=val, suit=suit, source="manual-test")
        print(f"Test {idx} : value={val!r}, suit={suit!r}")
        print(f"  formatted   = {c.formatted()!r}")
        print(f"  poker_card  = {c.poker_card!r}")
        print(f"  raw scan    = {c.scan()!r}")
        print("-" * 40)

    print("Vous pouvez également passer une carte en argument, ex :")
    print("  python card.py 'A♥'")

    if len(sys.argv) > 1:
        raw = sys.argv[1]
        print(f"\n=== Conversion directe depuis l'argument CLI : {raw!r} ===")
        pc = Card._convert_string_to_pokercard(raw)
        print("PokerCard =>", pc)
