import json
import sys
from pathlib import Path

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
for candidate in (ROOT_DIR, SCRIPTS_DIR):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from scripts import _utils


def test_resolve_templates_keeps_alias_and_layout():
    templates = {
        "action_button": {"size": [140, 70], "type": "texte", "layout": {"lock_same_y": True}},
        "secondary_button": {"alias_of": "action_button"},
        "player_money": {"size": [95, 40]},
    }

    resolved = _utils.resolve_templates(templates)

    assert resolved["action_button"]["size"] == [140, 70]
    assert resolved["secondary_button"]["size"] == [140, 70]
    assert resolved["secondary_button"]["type"] == "texte"
    assert resolved["secondary_button"]["layout"] == {"lock_same_y": True}
    assert resolved["player_money"]["type"] == ""


def test_coerce_int_handles_strings_and_invalid_values():
    assert _utils.coerce_int("12") == 12
    assert _utils.coerce_int("3.6") == 4
    assert _utils.coerce_int(None, default=5) == 5


def test_load_coordinates(tmp_path: Path):
    payload = {
        "table_capture": {"enabled": True},
        "templates": {
            "player_card_number": {"size": [20, 30]},
            "player_card_symbol": {"alias_of": "player_card_number", "type": "symbol"},
        },
        "regions": {
            "player_card_1_number": {"group": "player_card_number", "top_left": [10, 15]},
            "player_card_1_symbol": {"group": "player_card_symbol", "top_left": [40, 15]},
        },
    }
    coord_path = tmp_path / "coordinates.json"
    coord_path.write_text(json.dumps(payload), encoding="utf-8")

    regions, templates, table_capture = _utils.load_coordinates(coord_path)

    assert table_capture["enabled"] is True
    assert templates["player_card_symbol"]["type"] == "symbol"
    assert regions["player_card_1_number"].top_left == (10, 15)
    # alias_of → taille héritée
    assert regions["player_card_1_symbol"].size == (20, 30)


def test_extract_region_images_supports_dataclass_and_dict(tmp_path: Path):
    img = Image.new("RGB", (80, 60), color="white")
    regions = {
        "player_card_1_number": _utils.Region(
            key="player_card_1_number",
            group="player_card_number",
            top_left=(5, 10),
            size=(10, 12),
        ),
        "player_card_1_symbol": {
            "group": "player_card_symbol",
            "top_left": [20, 10],
            "size": [10, 12],
        },
    }

    patches = _utils.extract_region_images(
        img,
        regions,
        pad=0,
        groups_numbers=("player_card_number",),
        groups_suits=("player_card_symbol",),
    )

    assert "player_card_1" in patches
    number_patch, suit_patch = patches["player_card_1"]
    assert number_patch.size == (10, 12)
    assert suit_patch.size == (10, 12)
