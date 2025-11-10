"""Gestion de l'état des cartes de la table."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from pokereval.card import Card

from objet.entities.card import CardObservation, CardSlot


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


__all__ = ["CardsState"]
