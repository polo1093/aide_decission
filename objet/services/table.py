"""Service d'orchestration autour de l'état de la table de jeu."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

from objet.entities.card import CardObservation
from objet.state import ButtonsState, CardsState, CaptureState, extract_scan_value


@dataclass
class Table:
    """Réunit cartes, boutons et informations de capture."""

    cards: CardsState = field(default_factory=CardsState)
    buttons: ButtonsState = field(default_factory=ButtonsState)
    captures: CaptureState = field(default_factory=CaptureState)
    players: list[Any] = field(default_factory=list)

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


__all__ = ["Table"]
