# crop_video_frames.py — extrait une image crop de la table toutes les 1s depuis une vidéo de test
# Dossier vidéo par défaut : config/PMU/debug/cards_video/cards_video.*
# Sortie par défaut :       config/PMU/debug/crops/
# Dépendances: opencv-python, pillow, numpy
#   pip install opencv-python pillow numpy


import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# -----------------------------
# Helpers: config & path
# -----------------------------

def _load_capture_params(game_dir: Path) -> Tuple[Tuple[int,int], Tuple[int,int], Path]:
    """Lit coordinates.json et me.* dans config/<game>.
    Retourne (size(W,H), ref_offset(ox,oy), ref_path).
    """
    coords = game_dir / "coordinates.json"
    if not coords.exists():
        raise SystemExit(f"ERROR: {coords} not found. Pass --game-dir or run configure_table_crop first.")
    with coords.open("r", encoding="utf-8") as f:
        data = json.load(f)
    tc = data.get("table_capture", {})
    size = tuple(tc.get("size", [0, 0]))
    ref_offset = tuple(tc.get("ref_offset", [0, 0]))
    if not size or size == (0, 0):
        raise SystemExit("ERROR: table_capture.size missing or zero in coordinates.json")
    # ref image
    ref_path: Optional[Path] = None
    for ext in (".png", ".jpg", ".jpeg"):
        p = game_dir / f"me{ext}"
        if p.exists():
            ref_path = p
            break
    if not ref_path:
        raise SystemExit(f"ERROR: me.png/.jpg not found in {game_dir}")
    return (int(size[0]), int(size[1])), (int(ref_offset[0]), int(ref_offset[1])), ref_path


def _auto_video(game_dir: Path) -> Optional[Path]:
    base = game_dir / "debug" / "cards_video"
    if base.is_file():
        return base
    if base.is_dir():
        # cherche cards_video.* à l'intérieur OU le 1er fichier vidéo
        for ext in (".avi", ".mp4", ".mkv", ".mov"):
            p = base / f"cards_video{ext}"
            if p.exists():
                return p
        for f in sorted(base.glob("*")):
            if f.suffix.lower() in {".avi",".mp4",".mkv",".mov"}:
                return f
    # fallback: à la racine debug/
    for ext in (".avi", ".mp4", ".mkv", ".mov"):
        p = game_dir / "debug" / f"cards_video{ext}"
        if p.exists():
            return p
    return None


def _default_game_dir() -> Path:
    """Essaie de déduire config/PMU depuis l'emplacement du script ou le CWD.
    Évite le double préfixe quand on exécute depuis la racine du projet.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "config" / "PMU",   # .../aide_decission/config/PMU
        Path.cwd() / "config" / "PMU",
        Path.cwd().parent / "config" / "PMU",
    ]
    for c in candidates:
        if (c / "coordinates.json").exists():
            return c
    return candidates[0]

# -----------------------------
# Matching & crop (mémoire)
# -----------------------------

def _find_ref_point(screenshot_rgba: Image.Image, reference_rgba: Image.Image) -> Tuple[int,int]:
    scr_gray = cv2.cvtColor(np.array(screenshot_rgba.convert("RGB")), cv2.COLOR_RGB2GRAY)
    ref_gray = cv2.cvtColor(np.array(reference_rgba.convert("RGB")), cv2.COLOR_RGB2GRAY)
    res = cv2.matchTemplate(scr_gray, ref_gray, cv2.TM_CCOEFF_NORMED)
    _minVal, _maxVal, _minLoc, maxLoc = cv2.minMaxLoc(res)
    return int(maxLoc[0]), int(maxLoc[1])


def _clamp_box(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> Tuple[int,int,int,int]:
    x1 = max(0, min(x1, W)); y1 = max(0, min(y1, H))
    x2 = max(0, min(x2, W)); y2 = max(0, min(y2, H))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def crop_from_size_and_offset(frame_rgba: Image.Image, size: Tuple[int,int], ref_offset: Tuple[int,int], *, reference_img: Image.Image) -> Tuple[Image.Image, Tuple[int,int]]:
    W, H = int(size[0]), int(size[1])
    rx, ry = int(ref_offset[0]), int(ref_offset[1])
    ref_pt = _find_ref_point(frame_rgba, reference_img)
    x0, y0 = int(ref_pt[0] - rx), int(ref_pt[1] - ry)
    x1, y1 = x0 + W, y0 + H
    x0, y0, x1, y1 = _clamp_box(x0, y0, x1, y1, frame_rgba.width, frame_rgba.height)
    crop = frame_rgba.crop((x0, y0, x1, y1))
    return crop, (x0, y0)

# -----------------------------
# Vidéo → crops chaque 1s
# -----------------------------

def _iter_time_step(cap: cv2.VideoCapture, seconds_step: float):
    """Itère en lisant séquentiellement et en ne gardant qu'un frame toutes les N secondes.
    Certaines vidéos ne seek pas bien → lecture linéaire + modulo.
    """
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-6:
        fps = 30.0
    step_frames = max(1, int(round(fps * float(seconds_step))))
    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx % step_frames == 0:
            yield fidx, frame
        fidx += 1

# -----------------------------
# Main
# -----------------------------

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Crop table every N seconds from a test video")
    parser.add_argument("--game-dir", default=str(_default_game_dir()), help="Path to game dir (default: auto-detected config/PMU)")
    parser.add_argument("--video", help="Explicit video path; default: game_dir/debug/cards_video/cards_video.*")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between crops (default: 1.0)")
    parser.add_argument("--out", help="Output dir (default: game_dir/debug/crops)")
    args = parser.parse_args(argv)

    game_dir = Path(args.game_dir)
    out_dir = Path(args.out) if args.out else (game_dir / "debug" / "crops")
    out_dir.mkdir(parents=True, exist_ok=True)

    size, ref_offset, ref_path = _load_capture_params(game_dir)
    ref_img = Image.open(ref_path).convert("RGBA")

    video_path = Path(args.video) if args.video else _auto_video(game_dir)
    if not video_path or not video_path.exists():
        raise SystemExit(f"ERROR: no video found. Put a file inside {game_dir/'debug'/'cards_video'} or pass --video")

    print(f"Using game_dir: {game_dir}")
    print(f"Using video:    {video_path}")
    print(f"Crop size:      {size}  | ref_offset: {ref_offset}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open video: {video_path}")

    count = 0
    for fidx, frame_bgr in _iter_time_step(cap, seconds_step=float(args.interval)):
        # BGR -> PIL RGBA
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_img = Image.fromarray(frame_rgb).convert("RGBA")
        crop, origin = crop_from_size_and_offset(frame_img, size, ref_offset, reference_img=ref_img)
        # nom de fichier
        fname = f"crop_f{fidx:06d}.png"
        out_path = out_dir / fname
        crop.save(out_path)
        count += 1
        print(f"saved {out_path.name} origin={origin} size={crop.size}")

    cap.release()
    print(f"Done. {count} crops written to {out_dir}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
