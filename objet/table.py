"""Structures d'état liées à la table de jeu."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from pokereval.card import Card

from objet import bouton

LOGGER = logging.getLogger(__name__)

SUIT_ALIASES = {
    "hearts": "♥",
    "diamonds": "♦",
    "spades": "♠",
    "clubs": "♣",
    "heart": "♥",
    "diamond": "♦",
    "spade": "♠",
    "club": "♣",
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


@dataclass
class CardsState:
    """Regroupe les cartes du board et du joueur."""

    board: List[CardSlot] = field(default_factory=lambda: [CardSlot() for _ in range(5)])
    player: List[CardSlot] = field(default_factory=lambda: [CardSlot() for _ in range(2)])
    observations: Dict[str, CardObservation] = field(default_factory=dict)

    def apply_observation(self, base_key: str, observation: CardObservation) -> None:
        self.observations[base_key] = observation
        slot = self._slot_for_base_key(base_key)
        if slot:
            slot.apply(observation)

    def _slot_for_base_key(self, base_key: str) -> Optional[CardSlot]:
        if base_key.startswith("player_card_"):
            try:
                idx = int(base_key.split("_")[-1]) - 1
            except (ValueError, IndexError):
                return None
            if 0 <= idx < len(self.player):
                return self.player[idx]
        if base_key.startswith("board_card_"):
            try:
                idx = int(base_key.split("_")[-1]) - 1
            except (ValueError, IndexError):
                return None
            if 0 <= idx < len(self.board):
                return self.board[idx]
        return None

    def player_cards(self) -> List[Card]:
        return [slot.card for slot in self.player if slot.card is not None]

    def board_cards(self) -> List[Card]:
        return [slot.card for slot in self.board if slot.card is not None]

    def as_strings(self) -> Dict[str, List[str]]:
        def _format(slots: Iterable[CardSlot]) -> List[str]:
            out: List[str] = []
            for slot in slots:
                if slot.observation is None:
                    out.append("?")
                else:
                    out.append(slot.observation.formatted() or "?")
            return out

        return {
            "player": _format(self.player),
            "board": _format(self.board),
        }


@dataclass
class ButtonsState:
    """Maintient les trois boutons d'action."""

    buttons: Dict[str, bouton.Bouton] = field(
        default_factory=lambda: {f"button_{i}": bouton.Bouton() for i in range(1, 4)}
    )

    def update_from_scan(self, scan_table: Mapping[str, Any]) -> None:
        for name, btn in self.buttons.items():
            raw_value = extract_scan_value(scan_table, name)
            btn.string_to_bouton(raw_value)

    def best_button(self) -> Optional[str]:
        best_name: Optional[str] = None
        best_gain: float = float("-inf")
        for name, btn in self.buttons.items():
            if btn.gain is None:
                continue
            if btn.gain > best_gain:
                best_gain = btn.gain
                best_name = name
        return best_name


@dataclass
class CaptureState:
    """Paramètres liés aux captures et aux zones OCR."""

    table_capture: Dict[str, Any] = field(default_factory=dict)
    regions: "OrderedDict[str, Any]" = field(default_factory=OrderedDict)
    templates: Dict[str, Any] = field(default_factory=dict)
    reference_path: Optional[str] = None
    card_observations: Dict[str, CardObservation] = field(default_factory=dict)
    workflow: Optional[str] = None

    def update_from_coordinates(
        self,
        *,
        table_capture: Optional[Mapping[str, Any]] = None,
        regions: Optional[Mapping[str, Any]] = None,
        templates: Optional[Mapping[str, Any]] = None,
        reference_path: Optional[str] = None,
    ) -> None:
        if table_capture is not None:
            self.table_capture = dict(table_capture)
        if regions is not None:
            self.regions = OrderedDict(regions)
        if templates is not None:
            self.templates = dict(templates)
        if reference_path is not None:
            self.reference_path = reference_path

    def record_observation(self, base_key: str, observation: CardObservation) -> None:
        self.card_observations[base_key] = observation

    @property
    def size(self) -> Optional[List[int]]:
        size = self.table_capture.get("size") if isinstance(self.table_capture, dict) else None
        if isinstance(size, (list, tuple)) and len(size) == 2:
            return [int(size[0]), int(size[1])]
        return None

    @property
    def ref_offset(self) -> Optional[List[int]]:
        offset = self.table_capture.get("ref_offset") if isinstance(self.table_capture, dict) else None
        if isinstance(offset, (list, tuple)) and len(offset) == 2:
            return [int(offset[0]), int(offset[1])]
        return None


@dataclass
class Table:
    """Réunit cartes, boutons et informations de capture."""

    cards: CardsState = field(default_factory=CardsState)
    buttons: ButtonsState = field(default_factory=ButtonsState)
    captures: CaptureState = field(default_factory=CaptureState)
    players: List[Any] = field(default_factory=list)

    def apply_scan(self, scan_table: Mapping[str, Any]) -> None:
        for i in range(1, 6):
            number_key = f"board_card_{i}_number"
            symbol_key = f"board_card_{i}_symbol"
            observation = CardObservation(
                value=extract_scan_value(scan_table, number_key),
                suit=extract_scan_value(scan_table, symbol_key),
                source="scan",
            )
            self.cards.apply_observation(f"board_card_{i}", observation)
        for i in range(1, 3):
            number_key = f"player_card_{i}_number"
            symbol_key = f"player_card_{i}_symbol"
            observation = CardObservation(
                value=extract_scan_value(scan_table, number_key),
                suit=extract_scan_value(scan_table, symbol_key),
                source="scan",
            )
            self.cards.apply_observation(f"player_card_{i}", observation)
        self.buttons.update_from_scan(scan_table)

    def update_coordinates(
        self,
        *,
        table_capture: Optional[Mapping[str, Any]] = None,
        regions: Optional[Mapping[str, Any]] = None,
        templates: Optional[Mapping[str, Any]] = None,
        reference_path: Optional[str] = None,
    ) -> None:
        self.captures.update_from_coordinates(
            table_capture=table_capture,
            regions=regions,
            templates=templates,
            reference_path=reference_path,
        )

    def add_card_observation(self, base_key: str, observation: CardObservation) -> None:
        self.captures.record_observation(base_key, observation)
        self.cards.apply_observation(base_key, observation)

    def card_coordinates(self) -> Dict[str, Any]:
        """Retourne les coordonnées connues pour les cartes."""

        card_regions = {
            key: value
            for key, value in self.captures.regions.items()
            if key.startswith("board_card_") or key.startswith("player_card_")
        }
        return {
            "table_capture": dict(self.captures.table_capture),
            "regions": card_regions,
            "reference_path": self.captures.reference_path,
            "templates": {
                key: value
                for key, value in self.captures.templates.items()
                if key.startswith("board_card_") or key.startswith("player_card_")
            },
        }

    def suggest_action(
        self,
        *,
        chance_win_x: Optional[float],
        ev_calculator: Callable[[Optional[float], Optional[float]], Optional[float]],
    ) -> Optional[str]:
        max_gain = float("-inf")
        best_button: Optional[str] = None
        for name, btn in self.buttons.buttons.items():
            if btn.value is not None and btn.name is not None:
                gain = ev_calculator(chance_win_x, btn.value)
            elif btn.name == "se coucher":
                gain = 0.0
            elif btn.name is not None:
                gain = ev_calculator(chance_win_x, 0)
            else:
                gain = None
            btn.gain = gain
            if gain is not None and gain > max_gain:
                max_gain = gain
                best_button = name
        return best_button


def extract_scan_value(scan_table: Mapping[str, Any], key: str) -> Optional[str]:
    raw = scan_table.get(key)
    if isinstance(raw, Mapping):
        return raw.get("value")
    return raw


def convert_card(string_carte: Optional[str]) -> Optional[Card]:
    """Convertit une chaîne représentant une carte de poker en objet :class:`Card`."""

    suit_dict = {"♦": 1, "♥": 2, "♠": 3, "♣": 4}
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


__all__ = [
    "CardObservation",
    "CardsState",
    "ButtonsState",
    "CaptureState",
    "Table",
    "convert_card",
    "extract_scan_value",
]
