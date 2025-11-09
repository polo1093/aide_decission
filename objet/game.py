"""Gestion centralisée de l'état du jeu.

Le module expose :class:`Game` qui encapsule désormais l'ensemble des
structures manipulées par les différents workflows (cartes, boutons,
métriques et paramètres de capture). Les scripts s'appuient sur les méthodes
`from_scan` et `from_capture` pour remplir l'état sans manipuler directement
les dictionnaires intermédiaires.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional

from pokereval.card import Card
from pokereval.hand_evaluator import HandEvaluator

import tool
from objet import bouton
from scripts.state_requirements import SCRIPT_STATE_USAGE, StatePortion

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
            raw_value = _extract_value(scan_table, name)
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
class MetricsState:
    """Regroupe les métriques numériques."""

    pot: Optional[float] = None
    fond: Optional[float] = None
    chance_win_0: Optional[float] = None
    chance_win_x: Optional[float] = None
    player_money: Dict[str, Optional[float]] = field(
        default_factory=lambda: {f"J{i}": None for i in range(1, 6)}
    )
    players_count: int = 0

    def update_from_scan(self, scan_table: Mapping[str, Any]) -> None:
        self.pot = tool.convert_to_float(_extract_value(scan_table, "pot"))
        self.fond = tool.convert_to_float(_extract_value(scan_table, "fond"))
        for key in list(self.player_money.keys()):
            raw_key = f"player_money_{key}"
            self.player_money[key] = tool.convert_to_float(_extract_value(scan_table, raw_key))
        self.players_count = sum(1 for money in self.player_money.values() if money not in (None, 0))


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
class Game:
    """Stocke l'état courant de la table et calcule les décisions."""

    workflow: Optional[str] = None
    raw_scan: Dict[str, Any] = field(default_factory=dict)
    cards: CardsState = field(default_factory=CardsState)
    buttons: ButtonsState = field(default_factory=ButtonsState)
    metrics: MetricsState = field(default_factory=MetricsState)
    captures: CaptureState = field(default_factory=CaptureState)
    resultat_calcul: Dict[str, Any] = field(default_factory=dict)

    # ---- Fabrication -------------------------------------------------
    @classmethod
    def for_script(cls, script_name: str) -> "Game":
        game = cls(workflow=script_name)
        usage = SCRIPT_STATE_USAGE.get(script_name)
        if usage and StatePortion.CAPTURES in usage.portions:
            game.captures.workflow = script_name
        return game

    @classmethod
    def from_scan(cls, scan_table: Mapping[str, Any]) -> "Game":
        game = cls()
        game.update_from_scan(scan_table)
        return game

    @classmethod
    def from_capture(
        cls,
        *,
        table_capture: Optional[Mapping[str, Any]] = None,
        regions: Optional[Mapping[str, Any]] = None,
        templates: Optional[Mapping[str, Any]] = None,
        reference_path: Optional[str] = None,
        card_observations: Optional[Mapping[str, CardObservation]] = None,
        workflow: Optional[str] = None,
    ) -> "Game":
        game = cls(workflow=workflow)
        game.update_from_capture(
            table_capture=table_capture,
            regions=regions,
            templates=templates,
            reference_path=reference_path,
            card_observations=card_observations,
        )
        return game

    # ---- Mutateurs ---------------------------------------------------
    def update_from_scan(self, scan_table: Mapping[str, Any]) -> None:
        self.raw_scan = dict(scan_table)
        for i in range(1, 6):
            number_key = f"board_card_{i}_number"
            symbol_key = f"board_card_{i}_symbol"
            observation = CardObservation(
                value=_extract_value(scan_table, number_key),
                suit=_extract_value(scan_table, symbol_key),
                source="scan",
            )
            self.cards.apply_observation(f"board_card_{i}", observation)
        for i in range(1, 3):
            number_key = f"player_card_{i}_number"
            symbol_key = f"player_card_{i}_symbol"
            observation = CardObservation(
                value=_extract_value(scan_table, number_key),
                suit=_extract_value(scan_table, symbol_key),
                source="scan",
            )
            self.cards.apply_observation(f"player_card_{i}", observation)
        self.metrics.update_from_scan(scan_table)
        self.buttons.update_from_scan(scan_table)

    def update_from_capture(
        self,
        *,
        table_capture: Optional[Mapping[str, Any]] = None,
        regions: Optional[Mapping[str, Any]] = None,
        templates: Optional[Mapping[str, Any]] = None,
        reference_path: Optional[str] = None,
        card_observations: Optional[Mapping[str, CardObservation]] = None,
    ) -> None:
        self.captures.update_from_coordinates(
            table_capture=table_capture,
            regions=regions,
            templates=templates,
            reference_path=reference_path,
        )
        if card_observations:
            for base_key, observation in card_observations.items():
                self.add_card_observation(base_key, observation)

    def add_card_observation(self, base_key: str, observation: CardObservation) -> None:
        self.captures.record_observation(base_key, observation)
        self.cards.apply_observation(base_key, observation)

    # ---- Décision ----------------------------------------------------
    def decision(self) -> Optional[str]:
        if len(self.cards.player_cards()) != 2:
            return None
        try:
            self._calcul_chance_win()
        except ValueError as exc:  # état incomplet : on journalise et on abandonne
            LOGGER.warning("Impossible de calculer la décision: %s", exc)
            return None
        max_gain = float("-inf")
        best_button = None
        for name, btn in self.buttons.buttons.items():
            gain: Optional[float]
            if btn.value is not None and btn.name is not None:
                gain = self._calcule_ev(self.metrics.chance_win_x, btn.value)
            elif btn.name == "se coucher":
                gain = 0.0
            elif btn.name is not None:
                gain = self._calcule_ev(self.metrics.chance_win_x, 0)
            else:
                gain = None
            btn.gain = gain
            if gain is not None and gain > max_gain:
                max_gain = gain
                best_button = name
        return best_button

    # ---- Calculs internes --------------------------------------------
    def _calcul_chance_win(self) -> None:
        me_cards = self.cards.player_cards()
        board_cards = self.cards.board_cards()
        if len(me_cards) != 2:
            raise ValueError("Les cartes du joueur ne sont pas complètes ou invalides.")
        if len(board_cards) not in (0, 3, 4, 5):
            raise ValueError("Le nombre de cartes sur le board est incorrect.")
        self.metrics.chance_win_0 = HandEvaluator.evaluate_hand(me_cards, board_cards)
        players = max(1, int(self.metrics.players_count or 1))
        self.metrics.chance_win_x = (self.metrics.chance_win_0 or 0) ** players

    def _calcule_ev(self, chance_win: Optional[float], mise: Optional[float]) -> Optional[float]:
        if chance_win is None or mise is None or self.metrics.pot is None:
            return None
        players = max(1, int(self.metrics.players_count or 1))
        return chance_win * (self.metrics.pot + (mise * (players + 1))) - (1 - chance_win) * mise

    # ---- Diagnostics -------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow": self.workflow,
            "cards": self.cards.as_strings(),
            "buttons": {
                name: {
                    "name": btn.name,
                    "value": btn.value,
                    "gain": btn.gain,
                }
                for name, btn in self.buttons.buttons.items()
            },
            "metrics": {
                "pot": self.metrics.pot,
                "fond": self.metrics.fond,
                "chance_win_0": self.metrics.chance_win_0,
                "chance_win_x": self.metrics.chance_win_x,
                "player_money": self.metrics.player_money,
                "players_count": self.metrics.players_count,
            },
            "capture": {
                "table_capture": self.captures.table_capture,
                "regions": self.captures.regions,
                "templates": self.captures.templates,
                "reference_path": self.captures.reference_path,
            },
        }

    # Ancien nom conservé pour compatibilité éventuelle
    scan_to_data_table = update_from_scan


def _extract_value(scan_table: Mapping[str, Any], key: str) -> Optional[str]:
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
        logging.debug(message)
        string_carte = "10" + string_carte[1:]

    if len(string_carte) >= 2:
        value_part = string_carte[:-1]
        suit_part = string_carte[-1]
        if value_part in value_dict and suit_part in suit_dict:
            value = value_dict[value_part]
            suit = suit_dict[suit_part]
        else:
            logging.debug("Debug : La carte spécifiée '%s' n'est pas reconnue.", string_carte)
            return None
    else:
        logging.debug("Debug : La carte spécifiée '%s' est trop courte.", string_carte)
        return None

    return Card(value, suit)


__all__ = [
    "CardObservation",
    "CardsState",
    "ButtonsState",
    "MetricsState",
    "CaptureState",
    "Game",
    "convert_card",
]
