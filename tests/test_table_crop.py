import math
import sys
import types
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

ImageModule = pytest.importorskip("PIL.Image")
pytest.importorskip("PIL.ImageChops")


def _ensure_optional_dependencies():
    """Provide lightweight fallbacks for optional runtime dependencies."""

    try:
        import numpy  # noqa: F401
    except Exception:
        class SimpleArray:
            def __init__(self, data):
                self._data = data

            def __getitem__(self, key):
                if isinstance(key, tuple):
                    y, x = key
                    return self._data[y][x]
                return self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

            def tolist(self):
                return self._data

            def __array__(self):
                return self._data

        def _coerce(obj):
            if isinstance(obj, SimpleArray):
                return obj._data
            if hasattr(obj, "size") and hasattr(obj, "getdata"):
                width, height = obj.size
                source = obj.convert("RGB")
                data = list(source.getdata())
                rows = []
                idx = 0
                for _ in range(height):
                    row = []
                    for _ in range(width):
                        pixel = data[idx]
                        idx += 1
                        row.append(list(pixel))
                    rows.append(row)
                return rows
            if isinstance(obj, list):
                return [list(row) if isinstance(row, (list, tuple)) else row for row in obj]
            if isinstance(obj, tuple):
                return list(obj)
            return [obj]

        def np_array(obj):
            return SimpleArray(_coerce(obj))

        def np_clip(value, low, high):
            return max(low, min(high, value))

        def np_norm(vec):
            return math.sqrt(sum((float(v) ** 2 for v in vec)))

        numpy_stub = types.ModuleType("numpy")
        numpy_stub.array = np_array
        numpy_stub.asarray = np_array
        numpy_stub.asanyarray = np_array
        numpy_stub.clip = np_clip
        numpy_stub.uint8 = int
        numpy_stub.float32 = float
        numpy_stub.ndarray = SimpleArray
        numpy_stub.linalg = types.SimpleNamespace(norm=np_norm)
        sys.modules.setdefault("numpy", numpy_stub)

    try:
        import cv2  # noqa: F401
    except Exception:
        cv2_stub = types.ModuleType("cv2")
        cv2_stub.TM_CCOEFF_NORMED = 0
        cv2_stub.imread = lambda *_, **__: None
        cv2_stub.matchTemplate = lambda *_, **__: None
        cv2_stub.minMaxLoc = lambda *_: (0.0, 0.0, (0, 0), (0, 0))
        sys.modules.setdefault("cv2", cv2_stub)

    try:
        import matplotlib.pyplot  # noqa: F401
    except Exception:
        plt_stub = types.SimpleNamespace(
            imshow=lambda *_, **__: None,
            show=lambda *_, **__: None,
            figure=lambda *_, **__: None,
            clf=lambda *_, **__: None,
        )
        matplotlib_stub = types.ModuleType("matplotlib")
        matplotlib_stub.pyplot = plt_stub
        sys.modules.setdefault("matplotlib", matplotlib_stub)
        sys.modules.setdefault("matplotlib.pyplot", plt_stub)

    try:
        import pyautogui  # noqa: F401
    except Exception:
        pyautogui_stub = types.SimpleNamespace(
            click=lambda *_, **__: None,
            moveTo=lambda *_, **__: None,
            press=lambda *_, **__: None,
        )
        sys.modules.setdefault("pyautogui", pyautogui_stub)

    try:
        import folder_tool.ocr  # noqa: F401
    except Exception:
        folder_tool_pkg = types.ModuleType("folder_tool")
        ocr_module = types.ModuleType("folder_tool.ocr")

        class DummyOCR:
            def ocr_table(self, table, image, debug=False):
                return table

        ocr_module.OCR = DummyOCR
        folder_tool_pkg.ocr = ocr_module
        sys.modules.setdefault("folder_tool", folder_tool_pkg)
        sys.modules.setdefault("folder_tool.ocr", ocr_module)

    try:
        import pokereval.card  # noqa: F401
    except Exception:
        pokereval_module = types.ModuleType("pokereval")
        card_module = types.ModuleType("pokereval.card")

        class DummyCard:
            pass

        card_module.Card = DummyCard
        pokereval_module.card = card_module
        sys.modules.setdefault("pokereval", pokereval_module)
        sys.modules.setdefault("pokereval.card", card_module)


_ensure_optional_dependencies()

from objet.scan import ScanTable


def _load_image(path: Path):
    return ImageModule.open(path)


def _locate_asset(filename: str, candidate_dirs):
    for directory in candidate_dirs:
        base_dir = ROOT_DIR if not directory else ROOT_DIR / Path(directory)
        path = base_dir / filename
        if path.exists():
            return path
    pytest.skip(f"Missing required asset: {filename}")


def test_apply_table_crop_matches_expected():
    screenshot_path = _locate_asset(
        "test_crop.jpg",
        ["ScreenTestUnitaire", "screen/test_unitaire", "screen"]
    )
    expected_path = _locate_asset(
        "test_crop_result.png",
        ["ScreenTestUnitaire", "screen/test_unitaire", "screen"]
    )
    _locate_asset(
        "me.png",
        [
            "ScreenTestUnitaire",
            "screen/launch",
            "ScreenLaunch",
            "screen"
        ]
    )

    screenshot = _load_image(screenshot_path)
    expected = _load_image(expected_path)

    scan = ScanTable.__new__(ScanTable)
    scan.capture_settings = {"enabled": True, "relative_bounds": [-20, -10, 90, 60]}
    scan.table = {}
    scan.reference_point = (60, 80)
    scan.screen_old = screenshot.copy()
    scan.screen_array = None

    scan.apply_table_crop()

    result = scan.screen_old.convert("RGB")
    expected_rgb = expected.convert("RGB")

    assert result.size == expected_rgb.size
    assert result.tobytes() == expected_rgb.tobytes()
    assert scan.reference_point == (20, 10)