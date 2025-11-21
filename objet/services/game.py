"""Gestion centralisée de l'état du jeu."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tool
from objet.entities.card import Card, CardsState
from objet.entities.player import Players
from objet.entities.buttons import Buttons

from objet.services.table import Table
from objet.scanner.cards_recognition import CardObservation
from objet.utils.capture import CaptureState

from pokereval.hand_evaluator import HandEvaluator

from objet.services.script_state import SCRIPT_STATE_USAGE, StatePortion

LOGGER = logging.getLogger(__name__)



@dataclass
class Etat:
    """Stocke l'état courant de la table et calcule les décisions."""
    cards : CardsState = field(default_factory=CardsState)
    players : Players = field(default_factory=Players)
    cards_change : int = 0
    chance_win_0 : float = None
    montant_a_jouer : float = None
    
    def __post_init__(self) -> None:
        """Garantit que les états dépendants existent."""
        self.cards = CardsState()
        self.players = Players()

    
    
    
    def _cal_win_chances(self) -> float:
        """Calcule les chances de gain en fonction des cartes connues."""


        me_cards = self.cards.me_cards()
        if len(me_cards) != 2:
            raise ValueError("Les cartes du joueur ne sont pas completes ou invalides.")

        hero_cards = [
            self._require_poker_card(card, f"main heros {idx + 1}")
            for idx, card in enumerate(me_cards)
        ]

        board_cards = [card for card in cards_state.board_cards() if card.formatted]
        board_length = len(board_cards)
        if board_length not in (0, 3, 4, 5):
            raise ValueError("Le nombre de cartes sur le board est incorrect.")

        board_poker_cards = [
            self._require_poker_card(card, f"board card {idx + 1}")
            for idx, card in enumerate(board_cards)
        ]
        try:
            chance_win_0 = HandEvaluator.evaluate_hand(hero_cards, board_poker_cards)
        except Exception:
            logger.exception(
                "Erreur HandEvaluator.evaluate_hand : hero=%r board=%r",
                hero_cards,
                board_poker_cards,
            )
            return 0
        self.chance_win_0 = chance_win_0
        return chance_win_0
 

  
  



    def _calcul_montant_a_jouer(self)->float:
        self.montant_a_jouer = self.chance_win*(self.pot)/ (1-(self.chance_win*(self.nbr_palyer_start +1 )))
    
    def update_players(self,players : Players)->None:
        self.players = players # TODO a ameliorer 
        self.players.cal_nbr_player_start()
        self.players.cal_nbr_player_active()
    
    
    def update_cards_state(self, cards_state: CardsState) -> None:
        """Met à jour l'état des cartes."""
        nbr_scan = 3 *2 
        for i,card in enumerate(cards_state.board):
            if self.cards.board[i].formatted is None:
                self.cards.board[i] = card
            if card.formatted is None :
                continue   
            if card.formatted != self.cards.board[i].formatted:
                self.cards_change +=2
                if self.cards_change >= nbr_scan:
                        self.cards.board[i] = card
                        self.cards_change = 0
        
            
        for i,card in enumerate(cards_state.me):
            if self.cards.me[i].formatted is None:
                self.cards.me[i] = card
            if card.formatted is None :
                continue   
            if card.formatted != self.cards.me[i].formatted:
                self.cards_change +=2
                if self.cards_change >= nbr_scan:
                        self.cards.me[i] = card
                        self.cards_change = 0
        self.cards_change -=1
        
    def update(self,*,cards_state: CardsState, players : Players) -> None:
        self.update_cards_state(cards_state)
        self.update_players(players)
        self._cal_win_chances()
        self._calcul_montant_a_jouer()

@dataclass
class Game:
    """Stocke l'état courant de la table et calcule les décisions."""

    etat: Etat = field(default_factory=Etat)
    table: Table = field(default_factory=Table)
    metrics: Optional[MetricsState] = None
    resultat_calcul: Dict[str, Any] = field(default_factory=dict)
    workflow: Optional[str] = None
    _last_pot_amount: Optional[float] = field(default=None, init=False, repr=False)
    _new_party_flag: bool = field(default=False, init=False, repr=False)
    _pending_new_party_cleanup: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Garantit que les états dépendants existent."""
        self.table.cards = CardsState()
        self.table.buttons = Buttons()
        self.table.captures = CaptureState()
        



    
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
    

    def update_from_scan(self) -> Optional[bool]:
        """Met à jour l'état du jeu à partir du dernier scan.

        Returns:
            Optional[bool]: True si une nouvelle partie a été détectée, False sinon, None si le pot n'a pas été lu.
        """

        party_state = self._detect_new_party()
        self.etat.update(cards_state = self.table.cards, players = self.table.players)
        self.metrics.update(self.etat)
        self._calcul_chance_win()
        return party_state


   



    # ---- Décision ----------------------------------------------------
    def decision(self) -> Optional[str]:
        if self.metrics is None:
            raise ValueError("Les métriques de décision sont indisponibles.")

        
        return self.table.suggest_action(
            chance_win_x=self.metrics.chance_win_x,
            ev_calculator=self._calcule_ev,
        )

    # ---- Calculs internes --------------------------------------------
    def _detect_new_party(self) -> Optional[bool]:
        current_pot = getattr(self.table.pot, "amount", None)
        if current_pot is None:
            self._last_pot_amount = None
            self._new_party_flag = False
            return None

        if self._last_pot_amount is None:
            self._last_pot_amount = current_pot
            self._new_party_flag = False
            return False

        if current_pot < self._last_pot_amount:
            self._new_party_flag = True
            self._pending_new_party_cleanup = True
            self._last_pot_amount = current_pot
            return True

        self._new_party_flag = False
        self._pending_new_party_cleanup = False
        self._last_pot_amount = current_pot
        return False

    def ack_new_party(self) -> None:
        if not self._pending_new_party_cleanup:
            return
        self.table.New_Party()
        self.etat.cards.reset()
        self.etat.players.reset()
        self.metrics = None
        self._pending_new_party_cleanup = False
        self._new_party_flag = False

    @property
    def new_party_detected(self) -> bool:
        return self._new_party_flag

    





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
            game.table.buttons = Buttons()
        if StatePortion.CAPTURES not in portions:
            game.table.captures = CaptureState()
        return game
   

__all__ = [
    "Game",
    "CardObservation",
    "CardsState",
    "Buttons",
    "MetricsState",
    "CaptureState",
]
