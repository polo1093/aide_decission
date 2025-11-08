# ==============================
# crop_core.py — Coeur mémoire (size + ref_offset)
# ==============================
from __future__ import annotations
from typing import Tuple, Dict, Optional
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
import json

# ---------- Matching ----------

def find_ref_point(screenshot_img: Image.Image, reference_img: Image.Image) -> Tuple[int, int]:
    """Renvoie (x,y) du coin haut-gauche de `reference_img` dans `screenshot_img`."""
    scr_gray = cv2.cvtColor(np.array(screenshot_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    ref_gray = cv2.cvtColor(np.array(reference_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    result = cv2.matchTemplate(scr_gray, ref_gray, cv2.TM_CCOEFF_NORMED)
    _, _, _, loc = cv2.minMaxLoc(result)
    return int(loc[0]), int(loc[1])


def find_crop_top_left_by_matching(screenshot_img: Image.Image, crop_img: Image.Image) -> Tuple[int, int]:
    """Retrouve (x,y) du crop attendu dans le screenshot par corrélation."""
    scr_gray = cv2.cvtColor(np.array(screenshot_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    crop_gray = cv2.cvtColor(np.array(crop_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    result = cv2.matchTemplate(scr_gray, crop_gray, cv2.TM_CCOEFF_NORMED)
    _, _, _, loc = cv2.minMaxLoc(result)
    return int(loc[0]), int(loc[1])

# ---------- Géométrie ----------

def _clamp_box(box: Tuple[int,int,int,int], size: Tuple[int,int]) -> Tuple[int,int,int,int]:
    x1,y1,x2,y2 = box
    W,H = size
    x1 = max(0, min(x1, W))
    y1 = max(0, min(y1, H))
    x2 = max(0, min(x2, W))
    y2 = max(0, min(y2, H))
    if x2 < x1:
        x1,x2 = x2,x1
    if y2 < y1:
        y1,y2 = y2,y1
    return x1,y1,x2,y2


def _clamp_origin(x0: int, y0: int, size: Tuple[int,int], canvas: Tuple[int,int]) -> Tuple[int,int]:
    W,H = canvas
    w,h = size
    x0 = max(0, min(x0, max(0, W - w)))
    y0 = max(0, min(y0, max(0, H - h)))
    return x0, y0

# ---------- Crop runtime (mémoire) ----------

def crop_from_size_and_offset(
    screenshot_img: Image.Image,
    size: Tuple[int,int],
    ref_offset: Tuple[int,int],  # offset du point REF vers le coin haut-gauche de la fenêtre
    *,
    reference_img: Optional[Image.Image] = None,
    reference_point: Optional[Tuple[int,int]] = None,
) -> Tuple[Image.Image, Tuple[int,int]]:
    """Retourne (crop, (x0,y0)).

    - `size` = (W,H) de la fenêtre à extraire.
    - `ref_offset` = (ox,oy) = position RELATIVE du gabarit `me` *dans* la fenêtre (distance depuis le coin haut-gauche de la fenêtre jusqu'au coin haut-gauche de `me`).
    - `reference_point` ou `reference_img` sert à retrouver la position absolue de `me` dans le screenshot.

    Coin du crop = REF_ABS - ref_offset. Clamp si nécessaire.
    """
    W, H = int(size[0]), int(size[1])
    ox, oy = int(ref_offset[0]), int(ref_offset[1])
    if W <= 0 or H <= 0:
        raise ValueError("Invalid size; width/height must be > 0")

    if reference_point is None:
        if reference_img is None:
            raise ValueError("Provide reference_point or reference_img")
        reference_point = find_ref_point(screenshot_img, reference_img)

    rx, ry = int(reference_point[0]), int(reference_point[1])
    x0_raw, y0_raw = rx - ox, ry - oy
    x0, y0 = _clamp_origin(x0_raw, y0_raw, (W, H), screenshot_img.size)
    x1, y1 = x0 + W, y0 + H
    crop = screenshot_img.crop((x0, y0, x1, y1))
    return crop, (x0, y0)

# ---------- Comparaison / Vérif ----------

def _compare_images(img_a: Image.Image, img_b: Image.Image, pix_tol: int = 0) -> Tuple[bool, Dict[str, float]]:
    a = np.array(img_a.convert("RGB"))
    b = np.array(img_b.convert("RGB"))
    if a.shape != b.shape:
        return False, {"reason": "size_mismatch", "a_w": float(a.shape[1]), "a_h": float(a.shape[0]), "b_w": float(b.shape[1]), "b_h": float(b.shape[0])}
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    max_diff = int(diff.max())
    mean_diff = float(diff.mean())
    return (max_diff <= int(pix_tol)), {"max_diff": float(max_diff), "mean_diff": mean_diff}


def verify_geom(
    screenshot_img: Image.Image,
    expected_img: Image.Image,
    size: Tuple[int,int],
    ref_offset: Tuple[int,int],
    *,
    reference_img: Optional[Image.Image] = None,
    reference_point: Optional[Tuple[int,int]] = None,
    geom_tol: int = 1,
    pix_tol: int = 0,
) -> Tuple[bool, Dict[str, float]]:
    """Vérifie l'ALIGNEMENT géométrique:
       - prédit (px,py) via (size, ref_offset, ref_point)
       - mesure (mx,my) en matchant expected_img dans screenshot
       OK si |px-mx|<=geom_tol et |py-my|<=geom_tol.
       Ajoute stats pixel (max_diff/mean_diff) à titre informatif.
    """
    # prédit via runtime
    crop_pred, (px, py) = crop_from_size_and_offset(
        screenshot_img, size, ref_offset, reference_img=reference_img, reference_point=reference_point
    )
    # mesure via matching
    mx, my = find_crop_top_left_by_matching(screenshot_img, expected_img)

    dx, dy = int(px - mx), int(py - my)
    ok_geom = (abs(dx) <= int(geom_tol)) and (abs(dy) <= int(geom_tol))

    ok_pix, pix_stats = _compare_images(crop_pred, expected_img, pix_tol)
    stats = {"pred_top_left": (px, py), "match_top_left": (mx, my), "dx": float(dx), "dy": float(dy), **pix_stats}
    return ok_geom, stats

# ---------- Inference offset ----------

def infer_size_and_offset(
    screenshot_img: Image.Image,
    expected_img: Image.Image,
    reference_img: Image.Image,
) -> Tuple[Tuple[int,int], Tuple[int,int], Tuple[int,int], Tuple[int,int]]:
    """Calcule:
      - size = expected_img.size
      - crop_top_left_abs (cx,cy) en matchant `expected_img` dans le screenshot
      - ref_point_abs (rx,ry)
      - ref_offset = (rx-cx, ry-cy)
    Retourne: (size, ref_offset, (cx,cy), (rx,ry))
    """
    size = expected_img.size
    cx, cy = find_crop_top_left_by_matching(screenshot_img, expected_img)
    rx, ry = find_ref_point(screenshot_img, reference_img)
    ref_offset = (rx - cx, ry - cy)
    return size, ref_offset, (cx, cy), (rx, ry)

# ---------- JSON helpers ----------

def save_capture_json(path: Path, size: Tuple[int,int], ref_offset: Tuple[int,int]) -> None:
    data = {}
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data["table_capture"] = {"size": [int(size[0]), int(size[1])], "ref_offset": [int(ref_offset[0]), int(ref_offset[1])]}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ==========================================
# configure_table_crop.py — CLI de validation (geom d'abord)
# ==========================================
import argparse
from typing import Optional, List

from PIL import Image

# from crop_core import (
#   find_ref_point, find_crop_top_left_by_matching, crop_from_size_and_offset,
#   verify_geom, infer_size_and_offset, save_capture_json
# )


def _load_image(path: Path) -> Image.Image:
    return Image.open(path)


def _find_first(game_dir: Path, base: str, exts: Optional[List[str]] = None) -> Optional[Path]:
    exts = exts or [".png", ".jpg", ".jpeg"]
    for ext in exts:
        p = game_dir / f"{base}{ext}"
        if p.exists():
            return p
    return None


def _auto_paths_for_game(game: str, game_dir_opt: Optional[str]) -> dict:
    game_dir = Path(game_dir_opt) if game_dir_opt else Path("config") / (game or "PMU")
    screenshot = _find_first(game_dir, "test_crop", [".jpg", ".png", ".jpeg"])  # plein écran
    expected = _find_first(game_dir, "test_crop_result", [".png", ".jpg", ".jpeg"])  # fenêtre attendue
    reference = _find_first(game_dir, "me", [".png", ".jpg", ".jpeg"])  # gabarit ref
    output = game_dir / "coordinates.json"
    return {"game_dir": game_dir, "screenshot": screenshot, "expected": expected, "reference": reference, "output": output}


def parse_size(s: str) -> Tuple[int,int]:
    s = s.lower().replace("x", ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--size requires WxH or W,H")
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError("--size values must be integers") from e
    return w, h


def parse_offset(s: str) -> Tuple[int,int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--ref-offset requires two integers: ox,oy")
    try:
        ox, oy = int(parts[0]), int(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError("--ref-offset values must be integers") from e
    return ox, oy


def _with_debug_suffix(p: Path) -> Path:
    return p.with_name(p.stem + "_debug" + p.suffix)


def _save_any(path: Path, img: Image.Image) -> None:
    # Corrige "cannot write mode RGBA as JPEG"
    if path.suffix.lower() in (".jpg", ".jpeg") and img.mode == "RGBA":
        img = img.convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Infère et valide (size, ref_offset); écrit coordinates.json (geom match)")
    parser.add_argument("--game", default="PMU", help="Game folder under config/ (default: PMU)")
    parser.add_argument("--game-dir", help="Explicit game dir (overrides --game)")
    parser.add_argument("--size", type=parse_size, help="Override size WxH; default = expected image size")
    parser.add_argument("--ref-offset", type=parse_offset, help="Override ref offset ox,oy; default = inferred from images")
    parser.add_argument("--runs", type=int, default=10, help="Repeat verification N times (default: 10)")
    parser.add_argument("--pix-tol", type=int, default=0, help="Pixel tolerance (info only)")
    parser.add_argument("--geom-tol", type=int, default=1, help="Geometric tolerance in pixels (default: 1)")
    parser.add_argument("--write-crop", help="Optional path to save one computed crop for inspection")

    args = parser.parse_args(argv)

    auto = _auto_paths_for_game(args.game, args.game_dir)
    screenshot_path = auto["screenshot"]
    expected_path = auto["expected"]
    reference_path = auto["reference"]
    output_path = auto["output"]

    if screenshot_path is None or not screenshot_path.exists():
        raise SystemExit("ERROR: screenshot test not found (test_crop.jpg|.png)")
    if expected_path is None or not expected_path.exists():
        raise SystemExit("ERROR: expected crop not found (test_crop_result.*)")
    if reference_path is None or not reference_path.exists():
        raise SystemExit("ERROR: reference template not found (me.*)")

    scr = _load_image(screenshot_path).convert("RGBA")
    exp = _load_image(expected_path).convert("RGBA")
    ref = _load_image(reference_path).convert("RGBA")

    # 1) Taille + offset (inférence par défaut)
    if args.size and args.ref_offset:
        size = args.size
        ref_offset = args.ref_offset
        print("[override] size:", size, "ref_offset:", ref_offset)
    else:
        from __main__ import infer_size_and_offset  # if same file; adjust import if split
        size_inf, ref_off_inf, crop_pos, ref_pos = infer_size_and_offset(scr, exp, ref)
        size = args.size if args.size else size_inf
        ref_offset = args.ref_offset if args.ref_offset else ref_off_inf
        print("[infer] crop_top_left:", crop_pos, "ref_point:", ref_pos, "→ ref_offset:", ref_off_inf, "size:", size_inf)

    # 2) Écrit JSON (taille + ref_offset)
    from __main__ import save_capture_json  # adjust import if split
    save_capture_json(output_path, size, ref_offset)
    print("Wrote:", output_path)
    print("table_capture.size:", list(size), "table_capture.ref_offset:", list(ref_offset))

    # 3) Vérification répétée (géométrie en priorité)
    from __main__ import verify_geom, crop_from_size_and_offset
    ok_count = 0
    last_stats = {}
    for i in range(int(args.runs)):
        ok, stats = verify_geom(
            scr, exp, size, ref_offset,
            reference_img=ref,
            reference_point=None,
            geom_tol=int(args.geom_tol),
            pix_tol=int(args.pix_tol),
        )
        last_stats = stats
        print(f"run {i+1:02d}: ", "OK" if ok else "FAIL", stats)
        if ok:
            ok_count += 1

    # 4) Sauvegarde debug systématique
    crop, origin = crop_from_size_and_offset(scr, size, ref_offset, reference_img=ref)
    debug_path = _with_debug_suffix(expected_path)
    _save_any(debug_path, crop)
    print("Wrote computed crop (debug):", debug_path, "origin:", origin)

    if args.write_crop:
        _save_any(Path(args.write_crop), crop)
        print("Wrote computed crop (custom):", args.write_crop)

    print(f"Summary: {ok_count}/{args.runs} OK (geom_tol={args.geom_tol}, pix_tol={args.pix_tol})")
    return 0 if ok_count == int(args.runs) else 2


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))