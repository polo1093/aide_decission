"""Decision helper to recommend a poker action for the hero."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from objet.entities.card import Card, CardsState
from objet.services.game import Game
from objet.utils.metrics import MetricsState

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

        metrics = self._resolve_metrics(game)
        self._validate_metrics(metrics)

        chance_win = metrics.chance_win_x
        if chance_win is None:
            return DecisionResult(action="WAIT", reason="missing_chance_win_estimate")
        self._validate_chance(chance_win)

        if chance_win < self.FOLD_THRESHOLD:
            return DecisionResult(action="FOLD", reason="chance_win_below_fold_threshold")

        if chance_win < self.AGGRESSIVE_THRESHOLD:
            return DecisionResult(action="CALL", reason="chance_win_between_thresholds")

        raise_amount = self._compute_raise_amount(metrics)
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

    def _resolve_metrics(self, game: Game) -> MetricsState:
        metrics = game.metrics
        if metrics is None:
            metrics = MetricsState.from_game(game)
            game.metrics = metrics
        return metrics

    def _validate_metrics(self, metrics: MetricsState) -> None:
        if metrics.pot < 0:
            raise ValueError("Decision: pot must be non-negative.")
        if metrics.players_active <= 0:
            raise ValueError("Decision: there must be at least one active player.")
        if metrics.players_at_start <= 0:
            raise ValueError("Decision: the hand must have at least one starting player.")

    def _validate_chance(self, chance_win: float) -> None:
        if not 0.0 <= chance_win <= 1.0:
            raise ValueError("Decision: chance_win_x must be between 0 and 1.")

    def _compute_raise_amount(self, metrics: MetricsState) -> float:
        pot = metrics.pot
        if pot < 0:
            raise ValueError("Decision: cannot compute raise amount with negative pot.")
        return pot * self.RAISE_POT_FRACTION


__all__ = ["ActionType", "DecisionResult", "Decision"]
