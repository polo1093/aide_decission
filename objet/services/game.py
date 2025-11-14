"""Gestion centralisée de l'état du jeu."""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from pokereval.hand_evaluator import HandEvaluator

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tool
from objet.entities.card import Card
from objet.services.table import Table
from objet.scanner.cards_recognition import CardObservation
from objet.state import ButtonsState, CardsState, CaptureState, MetricsState

from objet.services.script_state import SCRIPT_STATE_USAGE, StatePortion

LOGGER = logging.getLogger(__name__)


@dataclass
class Game:
    """Stocke l'état courant de la table et calcule les décisions."""

    
    table: Table = field(default_factory=Table)
    metrics: MetricsState = field(default_factory=MetricsState)
    resultat_calcul: Dict[str, Any] = field(default_factory=dict)
    workflow: Optional[str] = None

    def __post_init__(self) -> None:
        """Garantit que les états dépendants existent."""
        if not isinstance(self.table.cards, CardsState):
            self.table.cards = CardsState()
        if not hasattr(self.table, "buttons") or not isinstance(self.table.buttons, ButtonsState):
            self.table.buttons = ButtonsState()
        if not hasattr(self.table, "captures") or not isinstance(self.table.captures, CaptureState):
            self.table.captures = CaptureState()

    # ---- Fabriques ---------------------------------------------------
    @classmethod
    def for_script(cls, script_name: str) -> "Game":
        """Construit un état de jeu léger pour un script donné."""

        game = cls()
        name = Path(script_name).name
        usage = SCRIPT_STATE_USAGE.get(name)
        if not usage:
            return game

        portions = usage.portions
        if StatePortion.CARDS not in portions:
            game.table.cards = CardsState()
        if StatePortion.BUTTONS not in portions:
            game.table.buttons = ButtonsState()
        if StatePortion.CAPTURES not in portions:
            game.table.captures = CaptureState()
        return game

    # ---- Accès pratiques --------------------------------------------
    @property
    def cards(self) -> CardsState:
        return self.table.cards

    @cards.setter
    def cards(self, state: CardsState) -> None:
        self.table.cards = state

   
    def scan_to_data_table(self) -> bool:
       
        if not self.table.launch_scan():
            return False
       
       
        return True
    

    def update_from_scan(self) -> None:
        """Met à jour l'état du jeu à partir du dernier scan."""
        # TODO: compléter lorsque les métriques / boutons seront branchés
        return None

    def update_from_capture(
        self,
        *,
        table_capture: Optional[Mapping[str, Any]] = None,
        regions: Optional[Mapping[str, Any]] = None,
        templates: Optional[Mapping[str, Any]] = None,
        reference_path: Optional[str] = None,
    ) -> None:
        """Injecte des paramètres de capture dans l'état courant."""

        self.table.captures.update_from_coordinates(
            table_capture=table_capture,
            regions=regions,
            templates=templates,
            reference_path=reference_path,
        )

    def add_card_observation(self, base_key: str, observation: CardObservation) -> None:
        """Enregistre une observation de carte pour inspection ultérieure."""

        card = Card()
        card.apply_observation(
            observation.value,
            observation.suit,
            observation.value_score,
            observation.suit_score,
        )
        self.table.captures.record_observation(base_key, card)
   
   
   



    # ---- Décision ----------------------------------------------------
    def decision(self) -> Optional[str]:
        if len(self.table.cards.player_cards()) != 2:
            return None
        try:
            self._calcul_chance_win()
        except ValueError as exc:  # état incomplet : on journalise et on abandonne
            LOGGER.warning("Impossible de calculer la décision: %s", exc)
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



__all__ = [
    "Game",
    "CardObservation",
    "CardsState",
    "ButtonsState",
    "MetricsState",
    "CaptureState",
    "convert_card",
]
