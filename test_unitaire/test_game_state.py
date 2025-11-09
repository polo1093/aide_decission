import sys
from pathlib import Path
import types

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if "numpy" not in sys.modules:
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.random = types.SimpleNamespace(uniform=lambda a, b: (float(a) + float(b)) / 2)
    numpy_stub.isscalar = lambda obj: isinstance(obj, (int, float))
    numpy_stub.bool_ = bool
    sys.modules.setdefault("numpy", numpy_stub)

if "pyautogui" not in sys.modules:
    pyautogui_stub = types.SimpleNamespace(
        locateOnScreen=lambda *_, **__: None,
        center=lambda loc: loc,
        leftClick=lambda *_, **__: None,
    )
    sys.modules.setdefault("pyautogui", pyautogui_stub)

try:
    import pokereval.card  # type: ignore
    import pokereval.hand_evaluator  # type: ignore
except Exception:
    pokereval_module = types.ModuleType("pokereval")
    card_module = types.ModuleType("pokereval.card")
    evaluator_module = types.ModuleType("pokereval.hand_evaluator")

    class DummyCard:
        def __init__(self, value, suit):
            self.value = value
            self.suit = suit

        def __repr__(self):
            return f"Card({self.value},{self.suit})"

    class DummyEvaluator:
        @staticmethod
        def evaluate_hand(me_cards, board_cards):
            return 0.5

    card_module.Card = DummyCard
    evaluator_module.HandEvaluator = DummyEvaluator
    pokereval_module.card = card_module
    pokereval_module.hand_evaluator = evaluator_module

    sys.modules.setdefault("pokereval", pokereval_module)
    sys.modules.setdefault("pokereval.card", card_module)
    sys.modules.setdefault("pokereval.hand_evaluator", evaluator_module)

from objet.game import CardObservation, Game
from scripts.state_requirements import SCRIPT_STATE_USAGE, StatePortion, describe_scripts


@pytest.fixture
def sample_scan_table():
    return {
        "player_card_1_number": {"value": "A"},
        "player_card_1_symbol": {"value": "♠"},
        "player_card_2_number": {"value": "10"},
        "player_card_2_symbol": {"value": "♥"},
        "board_card_1_number": {"value": "K"},
        "board_card_1_symbol": {"value": "♦"},
        "board_card_2_number": {"value": "Q"},
        "board_card_2_symbol": {"value": "♣"},
        "board_card_3_number": {"value": "J"},
        "board_card_3_symbol": {"value": "♠"},
        "pot": {"value": "1,20"},
        "fond": {"value": "10,00"},
        "button_1": {"value": "Miser 0,50 €"},
        "button_2": {"value": "Suivre"},
        "button_3": {"value": "Se coucher"},
        "player_money_J1": {"value": "2,00"},
        "player_money_J2": {"value": "0,00"},
        "player_money_J3": {"value": "1,50"},
        "player_money_J4": {"value": ""},
        "player_money_J5": {"value": None},
    }


def test_game_from_scan_updates_state(sample_scan_table):
    game = Game.from_scan(sample_scan_table)

    cards_summary = game.cards.as_strings()
    assert cards_summary["player"][0] == "A♠"
    assert cards_summary["player"][1] == "10♥"
    assert cards_summary["board"][:3] == ["K♦", "Q♣", "J♠"]

    assert pytest.approx(game.metrics.pot) == 1.20
    assert pytest.approx(game.metrics.fond) == 10.00
    assert game.metrics.players_count == 2

    assert game.buttons.buttons["button_1"].name.lower().startswith("m")
    assert game.buttons.buttons["button_3"].name == "Se coucher"


def test_capture_observation_uses_aliases():
    obs = {
        "player_card_1": CardObservation(value="K", suit="hearts", value_score=0.9, suit_score=0.95, source="capture"),
        "board_card_1": CardObservation(value="9", suit="spades", value_score=0.85, suit_score=0.9, source="capture"),
    }
    game = Game.from_capture(card_observations=obs)
    summary = game.cards.as_strings()
    assert summary["player"][0] == "K♥"
    assert summary["board"][0] == "9♠"
    assert game.captures.card_observations["player_card_1"].source == "capture"


def test_scan_then_capture_overwrites_board(sample_scan_table):
    game = Game.from_scan(sample_scan_table)
    game.add_card_observation(
        "board_card_1",
        CardObservation(value="7", suit="clubs", value_score=0.7, suit_score=0.8, source="capture"),
    )
    assert game.cards.as_strings()["board"][0] == "7♣"


def test_script_usage_catalogue():
    capture_usage = SCRIPT_STATE_USAGE["capture_cards.py"]
    assert StatePortion.CARDS in capture_usage.portions
    assert StatePortion.CAPTURES in capture_usage.portions

    description = describe_scripts()
    assert "Crop_Video_Frames.py" in description
    assert "cards" in description["capture_cards.py"]["portions"]
