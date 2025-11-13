"""Gestion centralisǸe de l'Ǹtat du jeu."""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from pokereval.hand_evaluator import HandEvaluator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from objet.entities.card import Card
from objet.services.table import Table
from objet.state import ButtonsState, CardsState, CaptureState, MetricsState

from scripts.state_requirements import SCRIPT_STATE_USAGE, StatePortion

LOGGER = logging.getLogger(__name__)


@dataclass
class Game:
    """Stocke l'Ǹtat courant de la table et calcule les dǸcisions."""

    workflow: Optional[str] = None
    raw_scan: Dict[str, Any] = field(default_factory=dict)
    table: Table = field(default_factory=Table)
    metrics: MetricsState = field(default_factory=MetricsState)
    resultat_calcul: Dict[str, Any] = field(default_factory=dict)

    @property
    def cards(self) -> CardsState:
        return self.table.cards

    @property
    def buttons(self) -> ButtonsState:
        return self.table.buttons

    @property
    def captures(self) -> CaptureState:
        return self.table.captures

    @classmethod
    def for_script(cls, script_name: str) -> "Game":
        game = cls(workflow=script_name)
        usage = SCRIPT_STATE_USAGE.get(script_name)
        if usage and StatePortion.CAPTURES in usage.portions:
            game.table.captures.workflow = script_name
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
        card_observations: Optional[Mapping[str, Card]] = None,
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

    def scan_to_data_table(self) -> bool:
        if not self.table.launch_scan():
            return False
        return True

    def update_from_scan(self, scan_table: Mapping[str, Any]) -> None:
        self.raw_scan = dict(scan_table)
        if hasattr(self.table, "apply_scan"):
            self.table.apply_scan(scan_table)
        self.metrics.update_from_scan(scan_table)

    def update_from_capture(
        self,
        *,
        table_capture: Optional[Mapping[str, Any]] = None,
        regions: Optional[Mapping[str, Any]] = None,
        templates: Optional[Mapping[str, Any]] = None,
        reference_path: Optional[str] = None,
        card_observations: Optional[Mapping[str, Card]] = None,
    ) -> None:
        self.table.captures.update_from_coordinates(
            table_capture=table_capture,
            regions=regions,
            templates=templates,
            reference_path=reference_path,
        )
        if card_observations:
            for base_key, observation in card_observations.items():
                self.table.captures.record_observation(base_key, observation)

    def add_card_observation(self, base_key: str, observation: Card) -> None:
        self.table.captures.record_observation(base_key, observation)

    # ---- DǸcision ----------------------------------------------------
    def decision(self) -> Optional[str]:
        if len(self.table.cards.player_cards()) != 2:
            return None
        try:
            self._calcul_chance_win()
        except ValueError as exc:  # Ǹtat incomplet : on journalise et on abandonne
            LOGGER.warning("Impossible de calculer la dǸcision: %s", exc)
            return None
        return self.table.suggest_action(
            chance_win_x=self.metrics.chance_win_x,
            ev_calculator=self._calcule_ev,
        )

    # ---- Calculs internes --------------------------------------------
    def _calcul_chance_win(self) -> None:
        me_cards = self.table.cards.player_cards()
        board_cards = self.table.cards.board_cards()
        if len(me_cards) != 2:
            raise ValueError("Les cartes du joueur ne sont pas compl��tes ou invalides.")
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
            "cards": self.table.cards.as_strings(),
            "buttons": {
                name: {
                    "name": btn.name,
                    "value": btn.value,
                    "gain": btn.gain,
                }
                for name, btn in self.table.buttons.buttons.items()
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
                "table_capture": self.table.captures.table_capture,
                "regions": self.table.captures.regions,
                "templates": self.table.captures.templates,
                "reference_path": self.table.captures.reference_path,
            },
        }

    # Ancien nom conservǸ pour compatibilitǸ Ǹventuelle
    scan_to_data_table = update_from_scan


__all__ = ["Game"]
