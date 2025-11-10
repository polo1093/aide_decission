# ==============================
# cards_core.py — extraction + matching
# ==============================
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import cv2
import numpy as np
from PIL import Image

# Accès modules du dépôt
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from objet.game import CardObservation, Game
from _utils import extract_region_images, load_coordinates

# --- Modèle ---

def is_card_present(patch: Image.Image, *, threshold: int = 240, min_ratio: float = 0.08) -> bool:
    """Heuristique simple: proportion de pixels *très clairs* sur toute la zone.
    - threshold: niveau (0–255) à partir duquel un pixel est considéré "blanc"
    - min_ratio: ratio minimal (ex: 8%)
    """
    arr = np.array(patch.convert("RGB"), dtype=np.uint8)
    white = np.all(arr >= threshold, axis=2)
    ratio = float(white.mean())
    return ratio >= float(min_ratio)


# --- Templates ---

class TemplateIndex:
    """Charge les gabarits de chiffres/figures et de symboles depuis config/<game>/cards.
    Dossier attendu:
      cards/numbers/<VALUE>/*.png
      cards/suits/<SUIT>/*.png
    """
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.numbers: Dict[str, List[np.ndarray]] = {}
        self.suits: Dict[str, List[np.ndarray]] = {}

    @staticmethod
    def _prep(gray: np.ndarray) -> np.ndarray:
        # optionnel: normalisation légère
        return gray

    @staticmethod
    def _imread_gray(p: Path) -> Optional[np.ndarray]:
        try:
            img = Image.open(p).convert("L")
            return np.array(img)
        except Exception:
            return None

    def _load_dir(self, sub: str) -> Dict[str, List[np.ndarray]]:
        base = self.root / sub
        out: Dict[str, List[np.ndarray]] = {}
        if not base.exists():
            return out
        for label_dir in sorted(base.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            imgs: List[np.ndarray] = []
            for f in sorted(label_dir.glob("*.png")):
                g = self._imread_gray(f)
                if g is not None:
                    imgs.append(self._prep(g))
            if imgs:
                out[label] = imgs
        return out

    def load(self) -> None:
        self.numbers = self._load_dir("numbers")
        self.suits = self._load_dir("suits")

    def check_missing(self, expect_numbers: Optional[List[str]] = None, expect_suits: Optional[List[str]] = None) -> Dict[str, List[str]]:
        miss: Dict[str, List[str]] = {"numbers": [], "suits": []}
        if expect_numbers:
            for v in expect_numbers:
                if v not in self.numbers:
                    miss["numbers"].append(v)
        if expect_suits:
            for s in expect_suits:
                if s not in self.suits:
                    miss["suits"].append(s)
        return miss


# --- Matching ---

def _to_gray(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)


def match_best(gray_img: np.ndarray, templates: List[np.ndarray], method: int = cv2.TM_CCOEFF_NORMED) -> float:
    best = -1.0
    for tpl in templates:
        if gray_img.shape[0] < tpl.shape[0] or gray_img.shape[1] < tpl.shape[1]:
            # si le template est plus grand que l'extrait, on saute (ou on pourrait resize)
            continue
        res = cv2.matchTemplate(gray_img, tpl, method)
        _, score, _, _ = cv2.minMaxLoc(res)
        best = max(best, float(score))
    return best


def recognize_number_and_suit(number_patch: Image.Image, suit_patch: Image.Image, idx: TemplateIndex) -> Tuple[Optional[str], Optional[str], float, float]:
    g_num = _to_gray(number_patch)
    g_suit = _to_gray(suit_patch)
    best_num, best_num_score = None, -1.0
    for label, tpls in idx.numbers.items():
        score = match_best(g_num, tpls)
        if score > best_num_score:
            best_num_score = score
            best_num = label
    best_suit, best_suit_score = None, -1.0
    for label, tpls in idx.suits.items():
        score = match_best(g_suit, tpls)
        if score > best_suit_score:
            best_suit_score = score
            best_suit = label
    return best_num, best_suit, best_num_score, best_suit_score


# ==============================
# cards_validate.py — CLI de vérification basique
# ==============================
import argparse

# from crop_core import crop_from_size_and_offset  # optionnel si on part du plein écran
# Ici on supposera qu'on travaille sur test_crop_result.* directement (table déjà croppée)


def _find_first(game_dir: Path, base: str, exts=(".png", ".jpg", ".jpeg")) -> Optional[Path]:
    for ext in exts:
        p = game_dir / f"{base}{ext}"
        if p.exists():
            return p
    return None


def _auto_paths_for_game(game: str, game_dir_opt: Optional[str]) -> dict:
    game_dir = Path(game_dir_opt) if game_dir_opt else Path("config") / (game or "PMU")
    table = _find_first(game_dir, "test_crop_result")
    coords = game_dir / "coordinates.json"
    cards_root = game_dir / "cards"
    return {"game_dir": game_dir, "table": table, "coords": coords, "cards_root": cards_root}


def _save_png(p: Path, img: Image.Image) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    img.save(p)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Vérifie l'extraction + matching des cartes pour un jeu")
    parser.add_argument("--game", default="PMU")
    parser.add_argument("--game-dir")
    parser.add_argument("--dump", action="store_true", help="Sauver extraits dans debug/")
    parser.add_argument("--num-th", type=float, default=0.6, help="Seuil score pour numbers (0..1)")
    parser.add_argument("--suit-th", type=float, default=0.6, help="Seuil score pour suits (0..1)")
    parser.add_argument("--pad", type=int, default=4)
    args = parser.parse_args(argv)

    auto = _auto_paths_for_game(args.game, args.game_dir)
    table_path = auto["table"]
    coords_path = auto["coords"]
    cards_root = auto["cards_root"]

    if not table_path or not table_path.exists():
        raise SystemExit("ERROR: test_crop_result not found")
    if not coords_path.exists():
        raise SystemExit("ERROR: coordinates.json not found")
    if not cards_root.exists():
        raise SystemExit("ERROR: cards folder not found (expected numbers/ and suits/)")

    # 1) Charger index
    idx = TemplateIndex(cards_root)
    idx.load()
    missing = idx.check_missing(
        expect_numbers=["A","K","Q","J","10","9","8","7","6","5","4","3","2"],
        expect_suits=["hearts","diamonds","clubs","spades"],
    )
    if missing["numbers"] or missing["suits"]:
        print("Missing templates:")
        if missing["numbers"]:
            print("  numbers:", ", ".join(missing["numbers"]))
        if missing["suits"]:
            print("  suits:", ", ".join(missing["suits"]))
        return 2

    # 2) Charger table et coordonnées
    table_img = Image.open(table_path).convert("RGBA")
    game = Game.for_script(Path(__file__).name)
    regions, resolved, table_capture = load_coordinates(coords_path)
    game.update_from_capture(
        table_capture=table_capture,
        regions={k: {"group": r.group, "top_left": r.top_left, "size": r.size} for k, r in regions.items()},
        templates=resolved,
        reference_path=str(table_path) if table_path else None,
    )

    # 3) Extraire patches cartes
    pairs = extract_region_images(table_img, regions, pad=int(args.pad))
    if not pairs:
        print("No card regions found (check coordinates.json groups)")
        return 2

    # 4) Reconnaissance
    ok = True
    debug_dir = auto["game_dir"] / "debug" / "cards"
    for base_key, (patch_num, patch_suit) in pairs.items():
        # filtre présence
        if not is_card_present(patch_num):  # si la zone nombre semble vide, on ignore la carte
            print(f"{base_key}: probably empty (skip)")
            continue
        val, suit, s_val, s_suit = recognize_number_and_suit(patch_num, patch_suit, idx)
        game.add_card_observation(
            base_key,
            CardObservation(value=val, suit=suit, value_score=s_val, suit_score=s_suit, source="capture"),
        )
        hit_val = (val is not None and s_val >= float(args.num_th))
        hit_suit = (suit is not None and s_suit >= float(args.suit_th))
        status = "OK" if (hit_val and hit_suit) else "LOW"
        print(f"{base_key}: {status}  value={val} ({s_val:.3f})  suit={suit} ({s_suit:.3f})")
        if args.dump:
            _save_png(debug_dir / f"{base_key}_number.png", patch_num)
            _save_png(debug_dir / f"{base_key}_symbol.png", patch_suit)
        if not (hit_val and hit_suit):
            ok = False

    if game.cards.as_strings():
        summary = game.cards.as_strings()
        print("Résumé Game → joueur:", ", ".join(summary["player"]))
        print("Résumé Game → board:", ", ".join(summary["board"]))

    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))


# ==========================================
# controller.py — Orchestrateur runtime (capture → crop → extract → match → état)
# ==========================================
# from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Iterable
from pathlib import Path

import json
from PIL import Image

# dépendances internes
# from crop_core import crop_from_size_and_offset, find_ref_point
# from cards_core import load_coordinates, TemplateIndex, extract_region_images, is_card_present, recognize_number_and_suit

@dataclass
class CardObs:
    value: Optional[str]
    suit: Optional[str]
    value_score: float
    suit_score: float

@dataclass
class CardState:
    value: Optional[str] = None
    suit: Optional[str] = None
    value_score: float = 0.0
    suit_score: float = 0.0
    stable: int = 0      # nb frames consécutifs où l'observation est identique et au-dessus des seuils
    last_seen: int = -1

class TableState:
    def __init__(self) -> None:
        self.cards: Dict[str, CardState] = {}  # base_key -> state

    def update(self, base_key: str, obs: CardObs, frame_idx: int, *, num_th: float, suit_th: float, require_k: int = 2) -> bool:
        """Met à jour l'état d'une carte; retourne True si *nouvelle* valeur stabilisée (changement)."""
        st = self.cards.get(base_key, CardState())
        confident = (obs.value is not None and obs.value_score >= num_th) and (obs.suit is not None and obs.suit_score >= suit_th)
        if confident:
            same = (st.value == obs.value) and (st.suit == obs.suit)
            st.stable = (st.stable + 1) if same else 1
            st.value, st.suit = obs.value, obs.suit
            st.value_score, st.suit_score = obs.value_score, obs.suit_score
            st.last_seen = frame_idx
            changed = (st.stable == require_k) and (not same)  # première fois où on atteint K
        else:
            st.stable = 0
            st.last_seen = frame_idx
            changed = False
        self.cards[base_key] = st
        return changed

    def snapshot(self) -> Dict[str, Dict[str, object]]:
        return {k: {"value": v.value, "suit": v.suit, "value_score": v.value_score, "suit_score": v.suit_score, "stable": v.stable} for k, v in self.cards.items()}

class TableController:
    def __init__(self, game_dir: Path, game_state: Optional[Game] = None) -> None:
        self.game_dir = Path(game_dir)
        self.coords_path = self.game_dir / "coordinates.json"
        self.ref_path = self._first_of("me", (".png", ".jpg", ".jpeg"))
        self.game = game_state or Game.for_script(Path(__file__).name)
        self.regions, self.templates, table_capture = load_coordinates(self.coords_path)
        self.game.update_from_capture(
            table_capture=table_capture,
            regions={k: {"group": r.group, "top_left": r.top_left, "size": r.size} for k, r in self.regions.items()},
            templates=self.templates,
            reference_path=str(self.ref_path) if self.ref_path else None,
        )
        self.size, self.ref_offset = self._load_capture_params()
        # runtime caches
        self.ref_img = Image.open(self.ref_path).convert("RGBA") if self.ref_path else None
        self.idx = TemplateIndex(self.game_dir / "cards")
        self.idx.load()
        self.state = TableState()

    def _first_of(self, stem: str, exts) -> Optional[Path]:
        for ext in exts:
            p = self.game_dir / f"{stem}{ext}"
            if p.exists():
                return p
        return None

    def _load_capture_params(self) -> Tuple[Tuple[int,int], Tuple[int,int]]:
        tc = self.game.table.captures.table_capture
        size = tc.get("size", [0, 0]) if isinstance(tc, dict) else [0, 0]
        ref_offset = tc.get("ref_offset", [0, 0]) if isinstance(tc, dict) else [0, 0]
        return (int(size[0]), int(size[1])), (int(ref_offset[0]), int(ref_offset[1]))

    def process_frame(self, frame_rgba: Image.Image, frame_idx: int, *, num_th: float = 0.6, suit_th: float = 0.6, require_k: int = 2) -> Dict[str, Dict[str, object]]:
        """Retourne un snapshot d'état après traitement du frame."""
        # 1) crop table via size + ref_offset
        crop, _ = crop_from_size_and_offset(frame_rgba, self.size, self.ref_offset, reference_img=self.ref_img)
        # 2) extractions number/symbol
        pairs = extract_region_images(crop, self.regions, pad=4)
        # 3) matching + mise à jour d'état
        for base_key, (patch_num, patch_suit) in pairs.items():
            if not is_card_present(patch_num):
                continue
            val, suit, s_val, s_suit = recognize_number_and_suit(patch_num, patch_suit, self.idx)
            obs = CardObs(val, suit, s_val, s_suit)
            self.state.update(base_key, obs, frame_idx, num_th=num_th, suit_th=suit_th, require_k=require_k)
            self.game.add_card_observation(
                base_key,
                CardObservation(value=val, suit=suit, value_score=s_val, suit_score=s_suit, source="capture"),
            )
        return self.state.snapshot()


# ==========================================
# capture_source.py — vidéo → frames PIL
# ==========================================
# from __future__ import annotations
from typing import Iterator
import cv2
import numpy as np
from PIL import Image

class VideoFrameSource:
    def __init__(self, path: str, *, bgr_to_rgb: bool = True) -> None:
        self.cap = cv2.VideoCapture(path)
        self.bgr_to_rgb = bgr_to_rgb
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

    def __iter__(self) -> Iterator[Image.Image]:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            if self.bgr_to_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield Image.fromarray(frame).convert("RGBA")
        self.cap.release()


# ==========================================
# labeler_cli.py — collecte des inconnus & labellisation simple
# ==========================================
# from __future__ import annotations
from typing import Optional
from pathlib import Path
from PIL import Image

class SampleSink:
    """Sauvegarde les extraits non reconnus vers config/<game>/unlabeled/{numbers|suits}/."""
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def save_number(self, key: str, img: Image.Image, frame_idx: int) -> Path:
        p = self.root / "unlabeled" / "numbers" / f"{key}_{frame_idx}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p)
        return p

    def save_suit(self, key: str, img: Image.Image, frame_idx: int) -> Path:
        p = self.root / "unlabeled" / "suits" / f"{key}_{frame_idx}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p)
        return p

class InteractiveLabeler:
    def __init__(self, cards_root: Path) -> None:
        self.cards_root = Path(cards_root)

    def add_number(self, img_path: Path, label: str) -> Path:
        dst = self.cards_root / "numbers" / label / img_path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.open(img_path).save(dst)
        return dst

    def add_suit(self, img_path: Path, label: str) -> Path:
        dst = self.cards_root / "suits" / label / img_path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.open(img_path).save(dst)
        return dst


# ==========================================
# run_video_validate.py — CLI: vidéo → détection en ligne + stockage inconnus
# ==========================================
# from __future__ import annotations
import argparse
from typing import Optional

# from controller import TableController
# from capture_source import VideoFrameSource
# from cards_core import extract_region_images, is_card_present
# from labeler_cli import SampleSink

from pathlib import Path


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Valide détection cartes sur une vidéo (crop+match+mémoire)")
    parser.add_argument("--game", default="PMU")
    parser.add_argument("--game-dir")
    parser.add_argument("--video", required=True)
    parser.add_argument("--stride", type=int, default=3, help="Ne traiter qu'un frame sur N (perf)")
    parser.add_argument("--num-th", type=float, default=0.65)
    parser.add_argument("--suit-th", type=float, default=0.65)
    parser.add_argument("--require-k", type=int, default=2, help="Nb de frames pour stabiliser")
    args = parser.parse_args(argv)

    game_dir = Path(args.game_dir) if args.game_dir else Path("config") / (args.game or "PMU")
    ctrl = TableController(game_dir)
    sink = SampleSink(game_dir)

    for i, frame in enumerate(VideoFrameSource(args.video)):
        if i % max(1, int(args.stride)) != 0:
            continue
        snap = ctrl.process_frame(frame, i, num_th=float(args.num_th), suit_th=float(args.suit_th), require_k=int(args.require_k))
        # stocker les inconnus (option basique: si zone présente mais non stable)
        crop, _ = crop_from_size_and_offset(frame, ctrl.size, ctrl.ref_offset, reference_img=ctrl.ref_img)
        pairs = extract_region_images(crop, ctrl.regions, pad=4)
        for base_key, (patch_num, patch_suit) in pairs.items():
            if not is_card_present(patch_num):
                continue
            st = ctrl.state.cards.get(base_key)
            if not st or st.stable == 0:
                # pas encore reconnu → on garde un échantillon pour labellisation ultérieure
                sink.save_number(base_key, patch_num, i)
                sink.save_suit(base_key, patch_suit, i)
        # Affiche un résumé court
        pretty = ", ".join([f"{k}:{v['value'] or '?'}-{v['suit'] or '?'}(s{v['stable']})" for k, v in sorted(snap.items())])
        print(f"frame {i:05d}: {pretty}")

    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))


# --- Helpers: default video path + dedupe hash (to paste into run_video_validate.py) ---
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image

def _auto_video_for_game(game_dir: Path) -> Optional[Path]:
    """Retourne config/<game>/cards_video.{avi,mp4,mkv,mov} si présent; sinon None."""
    for ext in (".avi", ".mp4", ".mkv", ".mov"):
        p = game_dir / f"cards_video{ext}"
        if p.exists():
            return p
    return None


def _ahash(img: Image.Image, hash_size: int = 8) -> str:
    """Average-hash (8x8 par défaut) pour éviter de sauvegarder des doublons d'extraits."""
    g = img.convert("L").resize((hash_size, hash_size), Image.BILINEAR)
    arr = np.array(g, dtype=np.float32)
    mean = float(arr.mean())
    # bitstring stable
    return "".join("1" if v > mean else "0" for v in arr.flatten())

# Exemple d'usage dans main():
#   video_path = Path(args.video) if args.video else _auto_video_for_game(game_dir)
#   seen_hash_numbers, seen_hash_suits = set(), set()
#   ...
#   hn, hs = _ahash(patch_num), _ahash(patch_suit)
#   if hn not in seen_hash_numbers: sink.save_number(base_key, patch_num, i); seen_hash_numbers.add(hn)
#   if hs not in seen_hash_suits:   sink.save_suit(base_key, patch_suit, i);   seen_hash_suits.add(hs)
