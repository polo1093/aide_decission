"""Decision helper to recommend a poker action for the hero."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from objet.entities.card import Card, CardsState
from objet.services.game import Game

ActionType = Literal["WAIT", "FOLD", "CALL", "CHECK", "RAISE"]


@dataclass(frozen=True)
class DecisionResult:
    """Immutable result describing the recommended poker action."""

    action: ActionType
    reason: str
    raise_amount: Optional[float] = None


class Decision:
    """Simple, fail-fast decision engine for the hero."""

    FOLD_THRESHOLD: float = 0.20
    AGGRESSIVE_THRESHOLD: float = 0.55
    RAISE_POT_FRACTION: float = 0.50

    def decide(self, game: Game) -> DecisionResult:
        """Return the recommended action for the hero based on the current state."""

        known_cards = self._extract_known_hero_cards(game.cards)
        if len(known_cards) < 2:
            return DecisionResult(action="WAIT", reason="hero_cards_not_detected_yet")

        chance_win = self._resolve_chance(game)
        pot_amount = self._resolve_pot(game)

        if chance_win < self.FOLD_THRESHOLD:
            return DecisionResult(action="FOLD", reason="chance_win_below_fold_threshold")

        if chance_win < self.AGGRESSIVE_THRESHOLD:
            return DecisionResult(action="CALL", reason="chance_win_between_thresholds")

        raise_amount = self._compute_raise_amount(pot_amount)
        return DecisionResult(
            action="RAISE",
            reason="chance_win_above_aggressive_threshold",
            raise_amount=raise_amount,
        )

    def _extract_known_hero_cards(self, cards_state: CardsState) -> Sequence[Card]:
        hero_cards = cards_state.me_cards()
        if len(hero_cards) != 2:
            raise ValueError("Decision: hero must have exactly two cards slots.")
        return [card for card in hero_cards if getattr(card, "formatted", None)]

    def _resolve_chance(self, game: Game) -> float:
        chance_win = game.etat.chance_win
        if chance_win is None:
            # Pas de valeur par défaut ici : on laisse l'exception remonter pour
            # exposer immédiatement le problème, conformément à la philosophie
            # du projet.
            raise ValueError("Decision: aucune estimation de chance de gain disponible.")
        if not 0.0 <= chance_win <= 1.0:
            raise ValueError("Decision: chance_win doit être entre 0 et 1.")
        if game.etat.players.nbr_player_active <= 0:
            raise ValueError("Decision: aucun joueur actif détecté.")
        if game.etat.players.nbr_player_start <= 0:
            raise ValueError("Decision: aucune main en cours.")
        return chance_win

    def _resolve_pot(self, game: Game) -> float:
        pot_amount = game.etat.pot
        if pot_amount is None:
            raise ValueError("Decision: pot manquant pour calculer le raise.")
        if pot_amount < 0:
            raise ValueError("Decision: pot négatif impossible.")
        return pot_amount

    def _compute_raise_amount(self, pot_amount: float) -> float:
        return pot_amount * self.RAISE_POT_FRACTION


__all__ = ["ActionType", "DecisionResult", "Decision"]
