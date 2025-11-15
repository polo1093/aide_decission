# crop_video_frames.py — extrait une image crop de la table toutes les 1s depuis une vidéo de test
# Dossier vidéo par défaut : config/PMU/debug/cards_video/cards_video.*
# Sortie par défaut :       config/PMU/debug/crops/
# Dépendances: opencv-python, pillow, numpy
#   pip install opencv-python pillow numpy

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Tuple
import random
import logging

import cv2
import numpy as np
from PIL import Image

# Ensure project root is on sys.path when running directly
import sys
try:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
except Exception:
    pass

from objet.services.game import Game
from objet.utils.calibration import load_coordinates

from crop_core import crop_and_save
# -----------------------------
# Helpers: config & path
# -----------------------------

def _load_capture_bounds(game_dir: Path) -> Tuple[int, int, int, int]:
    """Lit coordinates.json et retourne les bornes absolues de la table."""

    coords = game_dir / "coordinates.json"
    if not coords.exists():
        raise SystemExit(f"ERROR: {coords} not found. Pass --game-dir or run configure_table_crop first.")

    _regions, _templates, table_capture = load_coordinates(coords)
    bounds = table_capture.get("bounds") if isinstance(table_capture, dict) else None
    if not bounds or len(bounds) != 4:
        raise SystemExit("ERROR: table_capture.bounds missing or invalid in coordinates.json")
    x1, y1, x2, y2 = map(int, bounds)
    if x2 <= x1 or y2 <= y1:
        raise SystemExit("ERROR: table_capture.bounds has zero area")
    return x1, y1, x2, y2


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

<<<<<<< HEAD
=======
# -----------------------------
# Matching & crop (mémoire)
# -----------------------------

def crop_from_bounds(frame_rgba: Image.Image, bounds: Tuple[int, int, int, int]) -> Tuple[Image.Image, Tuple[int, int]]:
    x1, y1, x2, y2 = bounds
    x1 = max(0, min(int(x1), frame_rgba.width))
    y1 = max(0, min(int(y1), frame_rgba.height))
    x2 = max(0, min(int(x2), frame_rgba.width))
    y2 = max(0, min(int(y2), frame_rgba.height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    crop = frame_rgba.crop((x1, y1, x2, y2))
    return crop, (x1, y1)
>>>>>>> ba58e5c3c8cd276697a82ea4edfc5a2b6aacb52a

# -----------------------------
# Vidéo → crops chaque 1s (nom aléatoire)
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
    parser.add_argument("--interval", type=float, default=3.0, help="Seconds between crops (default: 3.0)")
    parser.add_argument("--out", help="Output dir (default: game_dir/debug/crops)")
    parser.add_argument("--log-level",default="INFO",help="Logging level (DEBUG, INFO, ...). Default: INFO",)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )
    
    logger = logging.getLogger("crop_video_frames")
    game_dir = Path(args.game_dir)
    out_dir = Path(args.out) if args.out else (game_dir / "debug" / "crops")
    out_dir.mkdir(parents=True, exist_ok=True)

<<<<<<< HEAD
    size, ref_offset, ref_path = _load_capture_params(game_dir)
    ref_img = Image.open(ref_path).convert("RGBA")
    game = None
    if Game is not None:
        game = Game.for_script(Path(__file__).name)
        game.update_from_capture(
            table_capture={"size": list(size), "ref_offset": list(ref_offset)},
            reference_path=str(ref_path),
        )
    else:
        logger.warning("Game service unavailable: %s", _IMPORT_ERROR)
    video_path = Path(args.video) if args.video else _auto_video(game_dir)
    if not video_path or not video_path.exists():
        raise SystemExit(f"ERROR: no video found. Put a file inside {game_dir/'debug'/'cards_video'} or pass --video")
    logger.info("Using game_dir: %s", game_dir)
    logger.info("Using video:    %s", video_path)
    logger.info("Crop size:      %s | ref_offset: %s", size, ref_offset)
=======
    bounds = _load_capture_bounds(game_dir)
    game = Game.for_script(Path(__file__).name)
    game.update_from_capture(
        table_capture={"bounds": list(bounds)},
    )

    video_path = Path(args.video) if args.video else _auto_video(game_dir)
    if not video_path or not video_path.exists():
        raise SystemExit(f"ERROR: no video found. Put a file inside {game_dir/'debug'/'cards_video'} or pass --video")

    print(f"Using game_dir: {game_dir}")
    print(f"Using video:    {video_path}")
    print(f"Crop bounds:    {bounds}")
>>>>>>> ba58e5c3c8cd276697a82ea4edfc5a2b6aacb52a

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open video: {video_path}")

    count = 0
    for fidx, frame_bgr in _iter_time_step(cap, seconds_step=float(args.interval)):
        # BGR -> PIL RGBA
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_img = Image.fromarray(frame_rgb).convert("RGBA")
<<<<<<< HEAD
        if game is not None and getattr(game.table.captures, "size", None):
            capture_size = tuple(game.table.captures.size)
        else:
            capture_size = tuple(size)
        if game is not None and getattr(game.table.captures, "ref_offset", None):
            capture_offset = tuple(game.table.captures.ref_offset)
        else:
            capture_offset = tuple(ref_offset)
=======
        capture_bounds = tuple(game.table.captures.bounds or bounds)
        crop, origin = crop_from_bounds(frame_img, capture_bounds)
>>>>>>> ba58e5c3c8cd276697a82ea4edfc5a2b6aacb52a
        # nom aléatoire dans [1, 10000]
        n = random.randint(1, 10000)
        fname = f"crop_{n}.png"
        out_path = out_dir / fname
        origin, debug = crop_and_save(
            frame_img,
            capture_size,
            capture_offset,
            reference_img=ref_img,
            output_path=out_path,
            logger=logger,
        )
        if origin is None:
            logger.warning("frame %s: crop failed (details=%s)", fidx, debug.to_log_dict())
            continue
        count += 1
        logger.info(
            "frame %s: saved %s origin=%s anchor_score=%.3f",
            fidx,
            out_path.name,
            origin,
            (debug.anchor_score or 0.0),)


    cap.release()
    logger.info("Extraction complete: %s crops saved", count)
    print(f"Done. {count} crops written to {out_dir}")
    if game is not None:
        print("Game capture context:", game.table.captures.table_capture)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
