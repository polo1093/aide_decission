import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

from PIL import Image
import numpy as np
import cv2


def _load_image(path: Path) -> Image.Image:
    return Image.open(path)


def _infer_from_images(
    screenshot_path: Path,
    crop_image_path: Path,
    reference_path: Path,
    *,
    tolerance: int = 1,
) -> Tuple[Dict[str, object], Tuple[int, int]]:
    screenshot = _load_image(screenshot_path)
    expected = _load_image(crop_image_path)
    reference = _load_image(reference_path)
    # Lazy import to avoid requiring heavy deps when not needed
    from objet.scan import ScanTable  # type: ignore
    return ScanTable.infer_capture_settings_from_images(
        screenshot, expected, reference, tolerance=tolerance
    )


def _match_reference_point(
    screenshot_path: Path, reference_path: Path
) -> Tuple[int, int]:
    screenshot_rgb = _load_image(screenshot_path).convert("RGB")
    reference_rgb = _load_image(reference_path).convert("RGB")
    screenshot_gray = cv2.cvtColor(np.array(screenshot_rgb), cv2.COLOR_RGB2GRAY)
    reference_gray = cv2.cvtColor(np.array(reference_rgb), cv2.COLOR_RGB2GRAY)
    result = cv2.matchTemplate(screenshot_gray, reference_gray, cv2.TM_CCOEFF_NORMED)
    _, _, _, loc = cv2.minMaxLoc(result)
    return int(loc[0]), int(loc[1])


def _update_or_create_config(
    output_path: Path, capture_settings: Dict[str, object]
) -> None:
    data: Dict[str, object]
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"regions": {}, "table_capture": {}}

    data.setdefault("regions", data.get("regions", {}))
    data["table_capture"] = {
        "enabled": bool(capture_settings.get("enabled", True)),
        "relative_bounds": list(capture_settings["relative_bounds"]),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_box(s: str) -> Tuple[int, int, int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--crop-box requires 4 comma-separated integers: x1,y1,x2,y2")
    try:
        x1, y1, x2, y2 = map(int, parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError("--crop-box values must be integers") from e
    return x1, y1, x2, y2


def parse_point(s: str) -> Tuple[int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--reference-point requires 2 comma-separated integers: x,y")
    try:
        x, y = map(int, parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError("--reference-point values must be integers") from e
    return x, y


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Configure table crop settings and write a game JSON configuration."
    )
    parser.add_argument("--game", default="coordinates", help="Game name (written to config/<game>.json by default)")
    parser.add_argument("--screenshot", required=True, help="Path to the full screenshot image")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--crop-image", help="Path to the expected cropped image of the table")
    group.add_argument("--crop-box", type=parse_box, help="Absolute crop box x1,y1,x2,y2 in the screenshot")
    parser.add_argument(
        "--reference-image",
        default=str(Path("screen/launch/me.png")),
        help="Path to the reference template used to determine the reference point",
    )
    parser.add_argument(
        "--reference-point",
        type=parse_point,
        help="Explicit reference point x,y (overrides --reference-image matching)",
    )
    parser.add_argument("--output", help="Output JSON path (defaults to config/<game>.json)")
    parser.add_argument("--tolerance", type=int, default=1, help="Tolerance in pixel value for verification")
    parser.add_argument("--disable", action="store_true", help="Write capture_settings.enabled = False")

    args = parser.parse_args(argv)

    screenshot_path = Path(args.screenshot)
    reference_path = Path(args.reference_image)
    output_path = Path(args.output) if args.output else Path("config") / f"{args.game}.json"

    if args.crop_image:
        capture_settings, _ = _infer_from_images(
            screenshot_path, Path(args.crop_image), reference_path, tolerance=args.tolerance
        )
    else:
        # Compute relative bounds from absolute crop box and a reference point
        x1, y1, x2, y2 = args.crop_box
        if args.reference_point:
            rx, ry = args.reference_point
        else:
            rx, ry = _match_reference_point(screenshot_path, reference_path)
        relative_bounds = [x1 - rx, y1 - ry, x2 - rx, y2 - ry]
        capture_settings = {"enabled": True, "relative_bounds": relative_bounds}

    if args.disable:
        capture_settings["enabled"] = False

    _update_or_create_config(output_path, capture_settings)

    print("Wrote:", output_path)
    print("capture_settings:", json.dumps(capture_settings))
    print("Hint: set GAME_CONFIG_FILE=\"" + str(output_path.resolve()) + "\" to use this file at runtime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
