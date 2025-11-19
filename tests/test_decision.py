"""Tests for the Decision service."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import pytest

from objet.entities.card import Card, CardsState
from objet.services.decision import Decision
from objet.utils.metrics import MetricsState


class DummyPlayer:
    def __init__(self, fond_amount: float) -> None:
        self.fond = SimpleNamespace(amount=fond_amount)
        self.fond_start_Party = fond_amount
        self.active_at_start = True
        self.etat = "play"

    def is_activate(self) -> bool:
        return True


class DummyPlayers:
    def __init__(self) -> None:
        self._players = [DummyPlayer(100.0), DummyPlayer(80.0)]

    def __len__(self) -> int:
        return len(self._players)

    def __getitem__(self, index: int) -> DummyPlayer:
        return self._players[index]

    def __iter__(self):
        return iter(self._players)


class DummyTable:
    def __init__(self, cards_state: CardsState, pot_amount: Optional[float] = 50.0) -> None:
        self.cards = cards_state
        self.players = DummyPlayers()
        self.pot = None if pot_amount is None else SimpleNamespace(amount=pot_amount)


class DummyGame:
    def __init__(self, cards_state: CardsState, *, pot_amount: Optional[float] = 50.0) -> None:
        self.cards = cards_state
        self.table = DummyTable(cards_state, pot_amount)
        self.metrics: Optional[MetricsState] = None


def _cards_state(first: Optional[str], second: Optional[str]) -> CardsState:
    board = [Card() for _ in range(5)]
    me = [Card(formatted=first), Card(formatted=second)]
    return CardsState(board=board, me=me)


def _game_with_cards(first: Optional[str], second: Optional[str]) -> DummyGame:
    return DummyGame(_cards_state(first, second))


def _metrics(chance: Optional[float], *, pot: float = 50.0) -> MetricsState:
    return MetricsState(
        pot=pot,
        players_active=2,
        players_at_start=2,
        hero_stack=100.0,
        hero_stack_start_hand=100.0,
        hero_stack_delta=0.0,
        hero_invested=5.0,
        board_cards=0,
        street="PREFLOP",
        player_stacks={"J1": 100.0, "J2": 50.0},
        chance_win_0=chance,
        chance_win_x=chance,
    )


def test_wait_when_hero_cards_missing() -> None:
    game = _game_with_cards("A♠", None)
    decision = Decision()

    result = decision.decide(game)

    assert result.action == "WAIT"
    assert result.reason == "hero_cards_not_detected_yet"


def test_fold_when_chance_below_threshold() -> None:
    game = _game_with_cards("A♠", "K♠")
    game.metrics = _metrics(0.10)
    decision = Decision()

    result = decision.decide(game)

    assert result.action == "FOLD"
    assert result.reason == "chance_win_below_fold_threshold"


def test_raise_when_chance_high() -> None:
    game = _game_with_cards("A♠", "K♠")
    game.metrics = _metrics(0.80, pot=120.0)
    decision = Decision()

    result = decision.decide(game)

    assert result.action == "RAISE"
    assert result.raise_amount == pytest.approx(60.0)
    assert result.reason == "chance_win_above_aggressive_threshold"


def test_value_error_when_pot_missing() -> None:
    game = _game_with_cards("A♠", "K♠")
    game.metrics = None
    game.table.pot = None
    decision = Decision()

    with pytest.raises(ValueError):
        decision.decide(game)
