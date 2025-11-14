# Bundle — aide_decission
_Généré le 2025-11-14 13:17:37_

## Fichiers

### folder_tool/__init__.py
```python

```
### folder_tool/timer.py
```python
import time


class Timer():
    """Timer used to check whether a waiting period has passed.

    Args:
        time_wait (float): Duration to wait in seconds.

    Returns:
        bool: True if the timer has expired.
    """
    def __init__(self,time_wait):
        self.start_time = time.perf_counter()
        self.time_wait = time_wait
    
    def is_expire(self):
        return time.perf_counter()-self.start_time >= self.time_wait    
    
    def is_running(self):
        return time.perf_counter()-self.start_time < self.time_wait
    
    def refresh(self,time_wait=0):
        if time_wait > 0: 
            self.time_wait = time_wait
        self.start_time = time.perf_counter()




```
### launch.py
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI simple pour exécuter périodiquement Controller.main()
=======================================================

- Boucle Tkinter non bloquante (`after`), Start/Stop/Snapshot.
- Intervalle de rafraîchissement paramétrable.
- On instancie directement Controller, sans import dynamique.
- On affiche directement la chaîne renvoyée par Controller.main().

Raccourcis clavier
------------------
- F5   : Start
- F6   : Snapshot (un seul appel à main())
- Échap: Stop
"""

from __future__ import annotations
import argparse
import time
from typing import Optional

import tkinter as tk
from tkinter import ttk

from objet.services.controller import Controller   # <--- IMPORTANT : import direct


class App(tk.Tk):
    def __init__(self, controller: Controller, scan_interval_ms: int = 250):
        super().__init__()

        self.title("Live Table – Controller UI")
        self.geometry("880x640")
        self.minsize(780, 520)

        # Backend
        self.controller = controller

        # Orchestration
        self.scanning = False
        self.scan_interval_ms = scan_interval_ms
        self._last_tick_t: Optional[float] = None

        # Perf
        self.last_call_ms: Optional[float] = None
        self.fps: Optional[float] = None

        # UI
        self._build()
        self._layout()
        self._bind_keys()

        self._set_text("Prêt. Appuie sur F5 ou sur ▶ Start.")

    # ---------- Construction UI ----------

    def _build(self):
        # Barre supérieure
        self.frm_top = ttk.Frame(self)
        self.btn_start = ttk.Button(self.frm_top, text="▶ Start", command=self.start_scan)
        self.btn_stop = ttk.Button(self.frm_top, text="⏸ Stop", command=self.stop_scan)
        self.btn_snap = ttk.Button(self.frm_top, text="📸 Snapshot", command=self.snapshot_once)
        self.lbl_interval = ttk.Label(self.frm_top, text="Interval (ms):")
        self.var_interval = tk.StringVar(value=str(self.scan_interval_ms))
        self.ent_interval = ttk.Entry(self.frm_top, width=6, textvariable=self.var_interval)
        self.lbl_status = ttk.Label(self.frm_top, text="Ready.", width=30, anchor="w")

        # Zone centrale texte + scroll
        self.frm_center = ttk.Frame(self)
        self.txt = tk.Text(
            self.frm_center,
            height=26,
            wrap="none",
            font=("Consolas", 12),
            state="disabled",
        )
        self.scroll = ttk.Scrollbar(
            self.frm_center,
            orient="vertical",
            command=self.txt.yview
        )
        self.txt.configure(yscrollcommand=self.scroll.set)

        # Barre inférieure (perf)
        self.frm_bottom = ttk.Frame(self)
        self.var_perf = tk.StringVar(value="scan: — ms | fps: —")
        self.lbl_perf = ttk.Label(self.frm_bottom, textvariable=self.var_perf)

    def _layout(self):
        # Top
        self.frm_top.pack(side="top", fill="x", padx=10, pady=8)
        self.btn_start.pack(side="left", padx=(0, 6))
        self.btn_stop.pack(side="left", padx=(0, 12))
        self.btn_snap.pack(side="left", padx=(0, 18))
        self.lbl_interval.pack(side="left")
        self.ent_interval.pack(side="left", padx=(6, 18))
        self.lbl_status.pack(side="left", padx=6)

        # Centre
        self.frm_center.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 6))
        self.txt.pack(side="left", fill="both", expand=True)
        self.scroll.pack(side="right", fill="y")

        # Bottom
        self.frm_bottom.pack(side="bottom", fill="x", padx=10, pady=(0, 10))
        self.lbl_perf.pack(side="left")

    def _bind_keys(self):
        # IMPORTANT : ne pas utiliser `_bind` (réservé par tkinter)
        self.bind("<Escape>", lambda e: self.stop_scan())
        self.bind("<F5>", lambda e: self.start_scan())
        self.bind("<F6>", lambda e: self.snapshot_once())
        self.ent_interval.bind("<Return>", lambda e: self._update_interval())

    # ---------- Helpers UI ----------

    def _set_text(self, text: str):
        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", text)
        self.txt.configure(state="disabled")

    def _update_interval(self):
        try:
            v = int(self.var_interval.get())
            self.scan_interval_ms = max(25, min(2000, v))
        except Exception:
            self.scan_interval_ms = 250
            self.var_interval.set(str(self.scan_interval_ms))

    # ---------- Orchestration ----------

    def start_scan(self):
        self._update_interval()
        self.scanning = True
        self.lbl_status.configure(text="Scanning…")
        self._last_tick_t = time.time()
        self.after(self.scan_interval_ms, self._tick)

    def stop_scan(self):
        self.scanning = False
        self.lbl_status.configure(text="Stopped.")

    def snapshot_once(self):
        """
        Un seul appel à Controller.main(), sans boucle continue.
        """
        t0 = time.perf_counter()
        try:
            out = self.controller.main()
        except Exception as e:
            self.last_call_ms = None
            self.fps = None
            self._set_text(f"Erreur Controller.main(): {e}")
            self.var_perf.set("scan: — ms | fps: —")
            self.lbl_status.configure(text="Erreur controller.")
            return

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.last_call_ms = dt_ms
        self.fps = None

        text = out if isinstance(out, str) else repr(out)
        self._set_text(text)

        self.var_perf.set(f"scan: {dt_ms:.1f} ms | fps: —")
        self.lbl_status.configure(text="Snapshot done.")

    def _tick(self):
        """
        Boucle périodique : appelle Controller.main() toutes les X ms.
        """
        if not self.scanning:
            return

        t0 = time.perf_counter()
        try:
            out = self.controller.main()
        except Exception as e:
            self.last_call_ms = None
            self.fps = None
            self._set_text(f"Erreur Controller.main(): {e}")
            self.var_perf.set("scan: — ms | fps: —")
            self.lbl_status.configure(text="Erreur controller.")
        else:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.last_call_ms = dt_ms

            # FPS estimé sur la boucle
            now = time.time()
            if self._last_tick_t is not None:
                dt_s = max(now - self._last_tick_t, 1e-6)
                self.fps = 1.0 / dt_s
            else:
                self.fps = None
            self._last_tick_t = now

            text = out if isinstance(out, str) else repr(out)
            self._set_text(text)

            fps_txt = f"{self.fps:.1f}" if self.fps else "—"
            self.var_perf.set(f"scan: {dt_ms:.1f} ms | fps: {fps_txt}")
            self.lbl_status.configure(text="OK.")

        if self.scanning:
            self.after(self.scan_interval_ms, self._tick)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="UI simple autour de Controller.main()")
    ap.add_argument(
        "--interval",
        type=int,
        default=250,
        help="Intervalle entre deux appels à main() en ms (25..2000)",
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    controller = Controller()
    app = App(controller=controller, scan_interval_ms=args.interval)
    app.mainloop()


if __name__ == "__main__":
    main()

```
### objet/__init__.py
```python
"""Paquetage structurǸ en entitǸs, Ǹtats et services."""

__all__ = ["entities", "services", "state"]

```
### objet/entities/__init__.py
```python
"""Entités de base manipulées par les services du projet."""
from .bouton import Action, Bouton
from .card import Card
from .player import Player

__all__ = [
    "Action",
    "Bouton",
    "CardObservation",
    "CardSlot",
    "convert_card",
    "Player",
]

```
### objet/entities/bouton.py
```python
import logging
from dataclasses import dataclass, field
from typing import ClassVar, Optional
import unicodedata
import re
import tool


@dataclass
class Action:
    """Représente une action détectée sur un bouton."""

    POSSIBLE_ACTIONS_BOUTON: ClassVar[list[str]] = [
        "parole",
        "suivre",
        "relancer a",
        "se coucher",
        "miser",
    ]
    liste_actions: ClassVar[list[str]] = POSSIBLE_ACTIONS_BOUTON + [
        "pas en jeu",
        "relance à fois 4",
        "close",
    ]  # À améliorer

    name: Optional[str] = None
    value: Optional[float] = None

    @classmethod
    def _normalize_string(cls, s):
        """
        Convertit une chaîne en minuscules et enlève les accents.
        
        Args:
            s (str): La chaîne à normaliser.
        
        Returns:
            str: La chaîne normalisée.
        """
        s = s.lower()
        # Enlever les accents
        s = ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
        return s

    
    # def get_possible_actions(cls):
    #     """
    #     Retourne la liste complète des actions possibles.
        
    #     Returns:
    #         list: Liste des actions possibles.
    #     """
    #     return cls.liste_actions

    @classmethod
    def create_action_from_string(cls, action_str):
        """
        Crée une instance d'Action à partir d'une chaîne de caractères.
        
        Args:
            action_str (str): La chaîne de caractères représentant l'action (e.g., "Mise 0,02 €").
        
        Returns:
            Action or None: Une instance d'Action si valide, sinon None.
        """
        if not action_str:
            return None

        normalized_str = cls._normalize_string(action_str)

        # Trouver l'action correspondante
        action_found = None
        for action in cls.POSSIBLE_ACTIONS_BOUTON:
            if action[:3] in normalized_str:
                action_found = action
                break

        if not action_found:
            logging.warning(f"Action non reconnue dans la chaîne : '{action_str}'")
            return None

        # Extraire la valeur numérique avec la virgule, si présente
        value = None
        match = re.search(r'(\d+,\d+)', normalized_str)
        if match:
            value_str = match.group(1).replace(',', '.')
            try:
                value = tool.convert_to_float(value_str)
            except ValueError:
                logging.warning(f"Impossible de convertir la valeur '{value_str}' en float.")
                value = None

        return cls(name=action_found.capitalize(), value=value)



@dataclass
class Bouton:
    """Représente un bouton d'action détecté à l'écran."""

    POSSIBLE_ACTIONS: ClassVar[list[str]] = Action.POSSIBLE_ACTIONS_BOUTON

    name: Optional[str] = None
    value: Optional[float] = None
    gain: Optional[float] = None

    def string_to_bouton(self, button_string):
        """
        Args:
            button_string (str): La chaîne de caractères extraite par l'OCR (e.g., "Mise 0,02 €").
        
        Returns:
            True or None.
        """
        # Utiliser la méthode de la classe Action pour créer une Action
        action_instance = Action.create_action_from_string(button_string)
        self.gain=None
        if action_instance:
            self.name = action_instance.name
            self.value = action_instance.value
            return True
        else:
            self.name = None
            self.value = None
            return None


```
### objet/entities/card.py
```python
"""Entité unique représentant une carte scannée et sa normalisation."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

from pokereval.card import Card as PokerCard

LOGGER = logging.getLogger(__name__)

SUIT_ALIASES = {
    "hearts": "\u2665",
    "diamonds": "\u2666",
    "spades": "\u2660",
    "clubs": "\u2663",
    "heart": "\u2665",
    "diamond": "\u2666",
    "spade": "\u2660",
    "club": "\u2663",
}


@dataclass
class Card:
    """Observation d'une carte et conversion vers l'objet PokerCard."""
    card_coordinates_value: Optional[tuple[int, int, int, int]] = None
    card_coordinates_suit: Optional[tuple[int, int, int, int]] = None
    value: Optional[str] = None
    suit: Optional[str] = None
    value_score: Optional[float] = None
    suit_score: Optional[float] = None
    poker_card: Optional[PokerCard] = None
    formatted: Optional[str] = None


    def scan(self) -> tuple[Optional[str], Optional[str]]:
        """
        Retourne la valeur brute scannée (value, suit).

        Utile pour debugger le flux OCR avant conversion en PokerCard.
        """
        return self.value, self.suit

    def apply_observation(
        self,
        value: Optional[str],
        suit: Optional[str],
        value_score: Optional[float] = None,
        suit_score: Optional[float] = None,
    ) -> None:
        """Applique une nouvelle observation et met à jour ."""
        LOGGER.debug(
            "apply_observation( value=%s, suit=%s, value_score=%s, suit_score=%s)",
            value,
            suit,
            value_score,
            suit_score,
        )
        self.value = value
        self.suit = suit
        self.value_score = value_score
        self.suit_score = suit_score
        if self.value and self.suit: 
            suit_sym = SUIT_ALIASES.get(self.suit, self.suit)
            formatted = f"{self.value}{suit_sym}"
            self.formatted = formatted
            self.poker_card = self._convert_string_to_pokercard(formatted) 
            
    def reset(self) -> None:
        """Réinitialise l'état de la carte."""
        self.value = None
        self.suit = None
        self.value_score = None
        self.suit_score = None
        self.poker_card = None
        self.formatted = None

    @staticmethod
    def _convert_string_to_pokercard(string_carte: Optional[str]) -> Optional[PokerCard]:
        """
        Convertit une chaîne '10♥' / 'A♠' en PokerCard (ou None si invalide).

        Mapping suits pokereval:
            1 -> spades (s)
            2 -> hearts (h)
            3 -> diamonds (d)
            4 -> clubs (c)
        """
        suit_dict = {
            "\u2660": 1,  # ♠
            "\u2665": 2,  # ♥
            "\u2666": 3,  # ♦
            "\u2663": 4,  # ♣
        }
        value_dict = {
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "10": 10,
            "J": 11,
            "Q": 12,
            "K": 13,
            "A": 14,
        }

        if string_carte in (None, "", "_"):
            return None

        string_carte = string_carte.strip()
        if not string_carte:
            return None

        # correction éventuelle si le scanner a renvoyé '0' au lieu de '10' en première position
        if string_carte[0] == "0" and len(string_carte) >= 2:
            original = string_carte
            corrected = "10" + string_carte[1:]
            LOGGER.debug(
                "Debug : La carte spécifiée '%s' est modifiée en '%s' pour correction.",
                original,
                corrected,
            )
            string_carte = corrected

        if len(string_carte) >= 2:
            value_part = string_carte[:-1]
            suit_part = string_carte[-1]
            value = value_dict.get(value_part)
            suit = suit_dict.get(suit_part)
            if value is not None and suit is not None:
                return PokerCard(value, suit)
            LOGGER.debug("Debug : La carte spécifiée '%s' n'est pas reconnue.", string_carte)
            return None

        LOGGER.debug("Debug : La carte spécifiée '%s' est trop courte.", string_carte)
        return None


__all__ = ["Card"]


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")

    print("=== Tests manuels de Card ===")

    tests = [
        ("A", "hearts"),
        ("10", "spades"),
        ("J", "diamonds"),
        (None, "clubs"),   # valeur manquante
        ("Q", None),       # couleur manquante
    ]

    for idx, (val, suit) in enumerate(tests, start=1):
        c = Card()
        c.apply_observation(value=val, suit=suit)
        print(f"Test {idx} : value={val!r}, suit={suit!r}")
        print(f"  formatted   = {c.formatted()!r}")
        print(f"  poker_card  = {c.poker_card!r}")
        print(f"  raw scan    = {c.scan()!r}")
        print("-" * 40)

    print("Vous pouvez également passer une carte en argument, ex :")
    print("  python card.py 'A♥'")

    if len(sys.argv) > 1:
        raw = sys.argv[1]
        print(f"\n=== Conversion directe depuis l'argument CLI : {raw!r} ===")
        pc = Card._convert_string_to_pokercard(raw)
        print("PokerCard =>", pc)

```
### objet/entities/player.py
```python
"""Entités décrivant les joueurs présents à la table."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Player:
    active_days_at_start: bool = False
    fond: Optional[float] = 0
    fond_start: Optional[float] = 0
    active_player: bool = False
    money_relance: Optional[float] = 0
    money_paid: Optional[float] = 0

    def refresh(self, fond: Optional[float]) -> None:
        self.fond = fond
        if self.fond:
            self.active_days_at_start = True


__all__ = ["Player"]

```
### objet/scanner/__init__.py
```python
# -*- coding: utf-8 -*-
from .scan import ScanTable

```
### objet/scanner/cards_recognition.py
```python
"""Card template helpers shared between scanner code and CLI tools."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

__all__ = [
    "CardObservation",
    "TemplateIndex",
    "is_card_present",
    "match_best",
    "recognize_number_and_suit",
]


@dataclass
class CardObservation:
    """Observation brute d'une carte (issue de la capture)."""

    value: Optional[str]
    suit: Optional[str]
    value_score: float
    suit_score: float
    source: str = "capture"


class TemplateIndex:
    """Charge les gabarits de chiffres/figures et de symboles depuis config/<game>/cards."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.numbers: Dict[str, List[np.ndarray]] = {}
        self.suits: Dict[str, List[np.ndarray]] = {}

    @staticmethod
    def _prep(gray: np.ndarray) -> np.ndarray:
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

    def check_missing(
        self,
        expect_numbers: Optional[Iterable[str]] = None,
        expect_suits: Optional[Iterable[str]] = None,
    ) -> Dict[str, List[str]]:
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


def _to_gray(img):
    """Normalise un patch en niveau de gris (ndarray 2D).

    Accepte :
    - un numpy.ndarray (BGR ou déjà en gris),
    - une image PIL,
    - au pire, tout objet convertible en ndarray.
    """
    # 1) Cas OpenCV / numpy
    if isinstance(img, np.ndarray):
        # Déjà en niveaux de gris
        if img.ndim == 2:
            return img
        # Image couleur (en pratique BGR si ça vient de cv2 / screen_crop)
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        raise ValueError(f"Format ndarray inattendu pour _to_gray: shape={img.shape}")

    # 2) Cas PIL.Image
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # 3) Fallback : on tente de convertir en ndarray
    arr = np.array(img)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # On part du principe que c’est du BGR (cas le plus probable avec OpenCV)
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    raise ValueError(f"Type d'image non supporté pour _to_gray: {type(img)}")


def is_card_present(patch: np.ndarray | Image.Image, *, threshold: int = 240, min_ratio: float = 0.08) -> bool:
    """Heuristique simple : proportion de pixels *très clairs* sur la zone."""

    if isinstance(patch, np.ndarray):
        arr = patch
        if arr.ndim == 2:
            arr_u8 = arr.astype(np.uint8, copy=False)
            white = arr_u8 >= threshold
            ratio = float(white.mean())
            return ratio >= float(min_ratio)
        if arr.ndim == 3:
            arr_u8 = arr.astype(np.uint8, copy=False)
        else:
            raise ValueError(f"Unsupported array shape for card presence: {arr.shape}")
    elif isinstance(patch, Image.Image):
        arr_u8 = np.array(patch.convert("RGB"), dtype=np.uint8)
    else:
        raise TypeError(f"Unsupported patch type for card presence: {type(patch)!r}")

    if arr_u8.ndim == 2:
        white = arr_u8 >= threshold
    else:
        white = np.all(arr_u8 >= threshold, axis=2)

    ratio = float(white.mean())
    return ratio >= float(min_ratio)


def match_best(gray_img: np.ndarray, templates: List[np.ndarray], method: int = cv2.TM_CCOEFF_NORMED) -> float:
    best = -1.0
    for tpl in templates:
        if gray_img.shape[0] < tpl.shape[0] or gray_img.shape[1] < tpl.shape[1]:
            continue
        res = cv2.matchTemplate(gray_img, tpl, method)
        _, score, _, _ = cv2.minMaxLoc(res)
        best = max(best, float(score))
    return best


def recognize_number_and_suit(
    number_patch: Image.Image,
    suit_patch: Image.Image,
    idx: TemplateIndex,
) -> Tuple[Optional[str], Optional[str], float, float]:
    """Retourne (value, suit, score_value, score_suit)."""

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

```
### objet/scanner/scan.py
```python

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import ImageGrab, Image
import logging

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from objet.utils.calibration import load_coordinates
from objet.utils.pyauto import locate_in_image
from objet.scanner.cards_recognition import TemplateIndex, is_card_present, recognize_number_and_suit

DEFAULT_COORD_PATH = Path("config/PMU/coordinates.json")
DEFAULT_CARDS_ROOT = Path("config/PMU/Cards")


class ScanTable:
    """Scan de la table PMU basé sur capture écran + pyautogui.

    - Localisation de la fenêtre via un template d'ancre (me.png) avec locate_in_image().
    - Utilisation de (size, ref_offset) issus de coordinates.json pour reconstruire le crop.
    - screen_array / screen_crop en BGR (convention OpenCV).
    """

    def __init__(self) -> None:
        # --- Config / calibration ---
        self.coord_path = DEFAULT_COORD_PATH
        _, _, table_capture = load_coordinates(self.coord_path)

        size_list = table_capture.get("size")
        ref_list = table_capture.get("ref_offset")

        if not size_list or not ref_list:
            raise ValueError(f"Invalid table_capture in {self.coord_path}: {table_capture}")

        self.size_crop: Tuple[int, int] = (int(size_list[0]), int(size_list[1]))
        self.offset_ref: Tuple[int, int] = (int(ref_list[0]), int(ref_list[1]))

        # Gabarit de référence (ancre) utilisé par pyautogui/locate
        self.reference_pil: Image.Image = Image.open("config/PMU/me.png").convert("RGB")

        # --- État runtime ---
        self.screen_array: Optional[np.ndarray] = None     # plein écran, BGR
        self.screen_crop: Optional[np.ndarray] = None      # crop table, BGR
        self.table_origin: Optional[Tuple[int, int]] = None
        self.scan_string: str = "init"
        self.cards_root = DEFAULT_CARDS_ROOT
        self.template_index = TemplateIndex(self.cards_root)
        # Première capture
        self.screen_refresh()


    
    def test_scan(self) -> bool:
        self.screen_refresh()
        return self.find_table()
    
    def screen_refresh(self) -> None:
        """Capture plein écran dans self.screen_array (numpy BGR)."""
        grab = ImageGrab.grab()               # PIL RGB
        rgb = np.array(grab)                  # numpy RGB
        self.screen_array = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # numpy BGR

    # ------------------------------------------------------------------
    # Localisation de la table via pyautogui
    # ------------------------------------------------------------------
    def find_table(self, *, grayscale: bool = True, confidence: float = 0.9) -> bool:
        """Localise la table via l'ancre + (size, ref_offset).

        Remplit :
          - self.screen_crop : crop couleur BGR de la table
          - self.table_origin : (x0, y0) top-left sur l'écran
          - self.scan_string : 'ok' ou "don't find".
        """
        if self.screen_array is None:
            self.scan_string = "no_screen"
            return False

        # 1) Localiser l'ancre dans le plein écran via pyautogui
        box = locate_in_image(
            haystack=self.screen_array,
            needle=self.reference_pil,
            assume_bgr=True,
            grayscale=grayscale,
            confidence=confidence,
        )

        if box is None:
            self.scan_string = "don't find"
            self.screen_crop = None
            self.table_origin = None
            return False

        anchor_left, anchor_top, anchor_w, anchor_h = box
        W, H = self.size_crop
        ox, oy = self.offset_ref

        # 2) Calcul du coin haut-gauche de la fenêtre de table
        x0 = int(anchor_left - ox)
        y0 = int(anchor_top - oy)

        # Clamp dans les bornes de l'écran
        h_scr, w_scr = self.screen_array.shape[:2]
        x0 = max(0, min(x0, w_scr - 1))
        y0 = max(0, min(y0, h_scr - 1))
        x1 = max(0, min(x0 + W, w_scr))
        y1 = max(0, min(y0 + H, h_scr))

        if x1 <= x0 or y1 <= y0:
            self.scan_string = "invalid_crop"
            self.screen_crop = None
            self.table_origin = None
            return False

        # 3) Crop BGR pour le reste du pipeline
        self.screen_crop = self.screen_array[y0:y1, x0:x1].copy()
        self.table_origin = (x0, y0)
        self.scan_string = "ok"
        return True

    # ------------------------------------------------------------------
    # Scan des cartes dans la table (identique à ta version, basé sur screen_crop)
    # ------------------------------------------------------------------
    def scan_carte(self, position_value: Tuple[int, int, int, int],position_suit ) -> Tuple[Optional[str], Optional[str], float, float]:
        """
        Retourne:
            (value, suit, confidence_value, confidence_suit)

        - value, suit : str ou None
        - confidence_* : float entre 0.0 et 1.0
        """

        h_img, w_img = self.screen_crop.shape[:2]


            

        # crops séparés pour la valeur et le symbole
        image_card_value = self._crop_box_gray(position_value)
        image_card_suit = self._crop_box_gray(position_suit)

        
        # rgb = cv2.cvtColor(image_card_value, cv2.COLOR_BGR2RGB)
        # Image.fromarray(rgb).show()
        
        if is_card_present(image_card_value):
            carte_value, carte_suit, score_value, score_suit = recognize_number_and_suit(image_card_value,image_card_suit,self.template_index) # manque un argument  sans dout pour la suit

            # Si ta fonction de reco ne retourne pas de score, on considère confidence = 1.0
            conf_val = 1.0 if carte_value is not None else 0.0
            conf_suit = 1.0 if carte_suit is not None else 0.0
            return carte_value, carte_suit, conf_val, conf_suit
        return None, None, 0.0, 0.0


 
    
    # Stubs à compléter plus tard
    def scan_pot(self, position):
        _ = self.screen_crop
        return None

    def scan_player(self, position):
        _ = self.screen_crop
        return None, None

    def scan_money_player(self, position):
        _ = self.screen_crop
        return None

    def scan_bouton(self, position):
        _ = self.screen_crop
        return None, None


    def _crop_box_gray(self,box, pad=3):
        """Retourne un crop (numpy BGR) pour box=(x,y,w,h) avec padding et clamp."""
        x, y, w, h = box
        x0 = max(0, int(x - pad))
        y0 = max(0, int(y - pad))
        x1 = min(self.screen_crop.shape[1], int(x + w + pad))
        y1 = min(self.screen_crop.shape[0], int(y + h + pad))
        img = self.screen_crop[y0:y1, x0:x1].copy()
        return img

if __name__ == "__main__":
    scan = ScanTable()
    print(scan.test_scan())
    

    import cv2
    import numpy as np
    from PIL import Image

    img = scan.screen_crop  # BGR

    if isinstance(img, np.ndarray):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).show()
    elif isinstance(img, Image.Image):
        img.show()
    else:
        print("Type d'image inattendu:", type(img))
        
```
### objet/services/__init__.py
```python
"""Services d'orchestration et composants applicatifs."""
from .cliqueur import Cliqueur
from .controller import Controller
# from .game import Game
# from .party import Party
from .table import Table

__all__ = [
    "Cliqueur",
    "Controller",
    "Table",
]

```
### objet/services/cliqueur.py
```python
import tool
import pyautogui
import sys
import time
from dataclasses import dataclass
from typing import Tuple
import pytweening
import random

@dataclass
class Cliqueur:
    wait_default: float = 0.05
    wait_press_default: float = 0.15

    def __post_init__(self):
        pass

    def click(self, coords: Tuple[int, int], button: str = 'left', wait: float = None, wait_press: float = None):
        """
        Effectue un clic de souris aux coordonnées spécifiées.

        Args:
            coords (Tuple[int, int]): Les coordonnées (x, y) pour le clic.
            button (str, optional): Le bouton de la souris à utiliser ('left', 'right'). Defaults to 'left'.
            wait (float, optional): Temps d'attente avant le clic. Defaults to wait_default.
            wait_press (float, optional): Durée du clic. Defaults to wait_press_default.
        """
        if wait is None:
            wait = self.wait_default
        if wait_press is None:
            wait_press = self.wait_press_default

        # Déplace la souris vers les coordonnées spécifiées avec une durée et un tweening aléatoires
        pyautogui.moveTo(
            coords[0],
            coords[1],
            duration=random.uniform(0.2, 0.6),
            tween=pytweening.easeInOutBounce
        )
        time.sleep(wait)
        pyautogui.mouseDown(button=button)
        time.sleep(wait_press + random.uniform(0.2, 0.6))
        pyautogui.mouseUp(button=button)
        time.sleep(wait)

    def click_button(self, button_rect: Tuple[int, int, int, int], shrink_factor: float = 0.6):
        """
        Clique sur un bouton en utilisant un point aléatoire dans un rectangle réduit.

        Args:
            button_rect (Tuple[int, int, int, int]): Coordonnées du rectangle du bouton (left, top, right, bottom).
        """
        # Décompose les coordonnées
        left, top, right, bottom = button_rect
        width = right - left
        height = bottom - top

        # Réduit le rectangle de 40% pour obtenir un rectangle centré plus petit
        new_width = width * shrink_factor
        new_height = height * shrink_factor
        left += (width - new_width) / 2
        top += (height - new_height) / 2

        # Sélectionne un point aléatoire dans le nouveau rectangle
        x = random.uniform(left, left + new_width)
        y = random.uniform(top, top + new_height)
        coords = (x, y)

        # Effectue le clic aux coordonnées choisies
        self.click(coords, 'left')

```
### objet/services/controller.py
```python
# launch_controller.py à la racine du projet

import cv2
import numpy as np
import PIL
from PIL import ImageGrab, Image
# from objet.services.cliqueur import Cliqueur
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class Controller():
    def __init__(self):
        self.count = 0
        self.running = False
        self.cpt = 0
        self.game_stat = {}
        #
        from objet.services.game import Game
        self.game = Game()
        # self.click = Cliqueur()
        
    def main(self):
        
             # machine à état de la partie et save 
            # Todo
        
            
            
        if self.game.scan_to_data_table():
        
            self.game.update_from_scan()
            

        
        
        
        
            return self.game_stat_to_string()
        self.cpt += 1
        return "don t find"+f"     Scan n°{self.cpt}"
        

    
    def game_stat_to_string(self):
        """
        Formate les informations du jeu pour l'utilisateur.

        Returns:
            str: Une chaîne de caractères contenant les informations formatées.
        """
        # Récupération des informations de base
        # metrics = self.game.metrics
        # nbr_player = metrics.players_count
        # pot = metrics.pot
        # fond = metrics.fond
        # chance_win_0 = metrics.chance_win_0
        # chance_win_x = metrics.chance_win_x

        # Fonction pour arrondir à 4 chiffres significatifs
        def round_sig(x, sig=4):
            if isinstance(x, (int, float)):
                return float(f"{x:.{sig}g}")
            else:
                return x

        # Arrondi des valeurs numériques
        # pot = round_sig(pot)
        # fond = round_sig(fond)
        # chance_win_0 = round_sig(chance_win_0)
        # chance_win_x = round_sig(chance_win_x)

        # Informations sur les cartes du joueur
        me_cards = [card.formatted for card in self.game.cards.me_cards()]
        me_cards_str = ', '.join(me_cards)

        # Informations sur le board
        board_cards = [card.formatted for card in self.game.cards.board_cards()]
        board_cards_str = ', '.join(board_cards)

        # Informations sur les boutons
        # buttons_info = []
        # # Ajout d'une ligne d'en-tête avec des largeurs de colonnes fixes
        # buttons_info.append(f"{'Bouton':<10} {'Action':<15} {'Valeur':<10} {'Gain':<10}")
        # buttons_info.append('-' * 50)  # Ligne de séparation

        # for i in range(1, 4):
        #     button = self.game.table.buttons.buttons.get(f'button_{i}')
        #     if button:
        #         name = button.name if button.name is not None else ''
        #         value = round_sig(button.value) if button.value is not None else ''
        #         gain = round_sig(button.gain) if button.gain is not None else ''
        #         buttons_info.append(f"{f'Button {i}':<10} {name:<15} {str(value):<10} {str(gain):<10}")
        #     else:
        #         buttons_info.append(f"{f'Button {i}':<10} {'':<15} {'':<10} {'':<10}")

        # buttons_str = '\n'.join(buttons_info)

        # Informations sur l'argent des joueurs
        # player_money = metrics.player_money
        # player_money_info = []
        # for player, money in player_money.items():
        #     money_str = str(round_sig(money)) if money is not None else 'Absent'
        #     player_money_info.append(f"{player}: {money_str}")

        # player_money_str = '\n'.join(player_money_info)

        return (
        #     f"Nombre de joueurs: {nbr_player}   Pot: {pot} €   Fond: {fond} €\n"
             f"Mes cartes: {me_cards_str}\n"
             f"Cartes sur le board: {board_cards_str}\n"
        #     f"Chance de gagner (1 joueur): {chance_win_0}\n"
        #     f"Chance de gagner ({nbr_player} joueurs): {chance_win_x}\n\n"
        #     f"Informations sur les boutons:\n{buttons_str}\n\n"
        #     f"Argent des joueurs:\n{player_money_str}"
         )

        return "Statistiques du jeu désactivées pour les tests de scan."



if __name__ == "__main__":
    controller = Controller()
    result = controller.main()
    print(result)
    
  # Sécurisation : on vérifie que table/scan/screen_array existent
    scan = getattr(controller.game.table, "scan", None)
    img = getattr(scan, "screen_array", None) if scan is not None else None
  

    import cv2
    import numpy as np
    from PIL import Image

    img = controller.game.table.scan.screen_array  # BGR

    if isinstance(img, np.ndarray):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).show()
    elif isinstance(img, Image.Image):
        img.show()
    else:
        print("Type d'image inattendu:", type(img))


    


    img = controller.game.table.scan.screen_crop  # BGR
    if img is None:
        print("Aucun crop de table disponible.")
    else:
        if isinstance(img, np.ndarray):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).show()
        elif isinstance(img, Image.Image):
            img.show()
        else:
            print("Type d'image inattendu:", type(img))


```
### objet/services/game.py
```python
"""Gestion centralisée de l'état du jeu."""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, Mapping, Optional

from pokereval.hand_evaluator import HandEvaluator

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tool
from objet.entities.card import Card
from objet.services.table import Table
from objet.scanner.cards_recognition import CardObservation
from objet.state import ButtonsState, CardsState, CaptureState, MetricsState

from objet.services.script_state import SCRIPT_STATE_USAGE, StatePortion

LOGGER = logging.getLogger(__name__)


@dataclass
class Game:
    """Stocke l'état courant de la table et calcule les décisions."""

    
    table: Table = field(default_factory=Table)
    metrics: MetricsState = field(default_factory=MetricsState)
    resultat_calcul: Dict[str, Any] = field(default_factory=dict)

   
    def scan_to_data_table(self) -> bool:
       
        if not self.table.launch_scan():
            return False
       
       
        return True
    

    def update_from_scan(self) -> None:
        pass
   
   
   



    # ---- Décision ----------------------------------------------------
    def decision(self) -> Optional[str]:
        if len(self.table.cards.player_cards()) != 2:
            return None
        try:
            self._calcul_chance_win()
        except ValueError as exc:  # état incomplet : on journalise et on abandonne
            LOGGER.warning("Impossible de calculer la décision: %s", exc)
            return None
        return self.table.suggest_action(
            chance_win_x=self.metrics.chance_win_x,
            ev_calculator=self._calcule_ev,
        )

    # ---- Calculs internes --------------------------------------------
    def _calcul_chance_win(self) -> None:
        me_cards = self.table.cards.player_cards()
        board_cards = self.table.cards.board_cards()
        if len(me_cards) != 2:
            raise ValueError("Les cartes du joueur ne sont pas complètes ou invalides.")
        if len(board_cards) not in (0, 3, 4, 5):
            raise ValueError("Le nombre de cartes sur le board est incorrect.")
        self.metrics.chance_win_0 = HandEvaluator.evaluate_hand(me_cards, board_cards)
        players = max(1, int(self.metrics.players_count or 1))
        self.metrics.chance_win_x = (self.metrics.chance_win_0 or 0) ** players

    def _calcule_ev(self, chance_win: Optional[float], mise: Optional[float]) -> Optional[float]:
        if chance_win is None or mise is None or self.metrics.pot is None:
            return None
        players = max(1, int(self.metrics.players_count or 1))
        return chance_win * (self.metrics.pot + (mise * (players + 1))) - (1 - chance_win) * mise

    # ---- Diagnostics -------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow": self.workflow,
            "cards": self.table.cards.as_strings(),
            "buttons": {
                name: {
                    "name": btn.name,
                    "value": btn.value,
                    "gain": btn.gain,
                }
                for name, btn in self.table.buttons.buttons.items()
            },
            "metrics": {
                "pot": self.metrics.pot,
                "fond": self.metrics.fond,
                "chance_win_0": self.metrics.chance_win_0,
                "chance_win_x": self.metrics.chance_win_x,
                "player_money": self.metrics.player_money,
                "players_count": self.metrics.players_count,
            },
            "capture": {
                "table_capture": self.table.captures.table_capture,
                "regions": self.table.captures.regions,
                "templates": self.table.captures.templates,
                "reference_path": self.table.captures.reference_path,
            },
        }

    # Ancien nom conservé pour compatibilité éventuelle
    scan_to_data_table = update_from_scan


__all__ = [
    "Game",
    "CardObservation",
    "CardsState",
    "ButtonsState",
    "MetricsState",
    "CaptureState",
    "convert_card",
]

```
### objet/services/script_state.py
```python
"""Descriptions des portions d'état consommées par les différents scripts."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet

__all__ = [
    "StatePortion",
    "ScriptStateUsage",
    "SCRIPT_STATE_USAGE",
    "describe_scripts",
]


class StatePortion(str, Enum):
    """Portions logiques de l'état de jeu consommées par les scripts."""

    CARDS = "cards"
    BUTTONS = "buttons"
    METRICS = "metrics"
    CAPTURES = "captures"


@dataclass(frozen=True)
class ScriptStateUsage:
    """Description des portions d'état nécessaires à un script."""

    name: str
    portions: FrozenSet[StatePortion]
    description: str


SCRIPT_STATE_USAGE: Dict[str, ScriptStateUsage] = {
    "capture_cards.py": ScriptStateUsage(
        name="capture_cards.py",
        portions=frozenset({StatePortion.CARDS, StatePortion.CAPTURES}),
        description="Extraction et reconnaissance des cartes depuis une capture.",
    ),
    "Crop_Video_Frames.py": ScriptStateUsage(
        name="Crop_Video_Frames.py",
        portions=frozenset({StatePortion.CAPTURES}),
        description="Découpe périodique des captures vidéo à partir des paramètres de table.",
    ),
    "crop_core.py": ScriptStateUsage(
        name="crop_core.py",
        portions=frozenset({StatePortion.CAPTURES}),
        description="Fonctions communes de capture/crop et outils de validation géométrique.",
    ),
    "position_zones.py": ScriptStateUsage(
        name="position_zones.py",
        portions=frozenset(
            {
                StatePortion.CAPTURES,
                StatePortion.CARDS,
                StatePortion.BUTTONS,
                StatePortion.METRICS,
            }
        ),
        description="Éditeur Tk classique des zones OCR (cartes, boutons, métriques).",
    ),
    "position_zones_ctk.py": ScriptStateUsage(
        name="position_zones_ctk.py",
        portions=frozenset(
            {
                StatePortion.CAPTURES,
                StatePortion.CARDS,
                StatePortion.BUTTONS,
                StatePortion.METRICS,
            }
        ),
        description="Éditeur CustomTkinter des zones OCR (cartes, boutons, métriques).",
    ),
    "zone_project.py": ScriptStateUsage(
        name="zone_project.py",
        portions=frozenset(
            {
                StatePortion.CAPTURES,
                StatePortion.CARDS,
                StatePortion.BUTTONS,
                StatePortion.METRICS,
            }
        ),
        description="Modèle et opérations associées aux projets de zones OCR.",
    ),
    "copy_python_sources.py": ScriptStateUsage(
        name="copy_python_sources.py",
        portions=frozenset(),
        description="Outil utilitaire sans dépendance sur l'état de jeu.",
    ),
}


def describe_scripts() -> Dict[str, Dict[str, str]]:
    """Retourne un dictionnaire sérialisable listant les usages déclarés."""

    return {
        name: {
            "portions": sorted(usage.portions),
            "description": usage.description,
        }
        for name, usage in SCRIPT_STATE_USAGE.items()
    }

```
### objet/services/table.py
```python
"""Service d'orchestration autour de l'état de la table de jeu."""


from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from objet.state import ButtonsState, CaptureState, CardsState
from objet.scanner.scan import ScanTable
from objet.utils.calibration import Region, load_coordinates , bbox_from_region
DEFAULT_COORD_PATH = Path("config/PMU/coordinates.json")

@dataclass
class Fond:
    coordinates_value: Optional[tuple[int, int, int, int]] = None
    amount: float = 0.0
    
    def reset(self) -> None:
        self.amount = 0.0
    

@dataclass
class Player:
    coordinates_value: Optional[tuple[int, int, int, int]] = None
    active_start : bool = True
    continue_round : bool = True
    fond = Fond(coordinates_value=coordinates_value)
    
    def reset(self) -> None:
        self.amount = 0.0
    

@dataclass
class Table:
    """Réunit Les éléments à scanner et service de scan."""
    
    coord_path: Path | str = DEFAULT_COORD_PATH
    cards: CardsState = field(default_factory=CardsState)
    # buttons: ButtonsState = field(default_factory=ButtonsState)
    # captures: CaptureState = field(default_factory=CaptureState)
    scan = ScanTable()
    pot = Fond()
    new_party_flag: bool = False
    
    
    def __post_init__(self) -> None:
        regions, templates_resolved, _ = load_coordinates(self.coord_path)
        self.pot.coordinates_value = bbox_from_region(regions.get("pot"))
        
        
    def launch_scan(self) -> bool:
       
        if not self.scan.test_scan():
            return False
        
        # --- Main héros (2 cartes) ---
        for idx, card in enumerate(self.cards.me, start=1):
            value, suit, confidence_value, confidence_suit = self.scan.scan_carte(
                position_value=card.card_coordinates_value,
                position_suit=card.card_coordinates_suit
            )
            if value is not None or suit is not None:
                if value != card.value and suit != card.suit:
                    self.New_Party()
            card.apply_observation(
                value=value,
                suit=suit,
                value_score=confidence_value,
                suit_score=confidence_suit,
            )
        
        for idx, card in enumerate(self.cards.board, start=1):
            if card.formatted is None:
                value, suit, confidence_value, confidence_suit = self.scan.scan_carte(
                    position_value=card.card_coordinates_value,
                    position_suit=card.card_coordinates_suit,
                )
                if value is None and suit is None:
                    continue
                card.apply_observation(
                    value=value,
                    suit=suit,
                    value_score=confidence_value,
                    suit_score=confidence_suit,
                )

                
        # self.scan.scan_pot()


        return True

    def New_Party(self)-> None:
        """Réinitialise l'état de la Table. et fait remonter un événement."""
        self.cards.reset()
        self.new_party_flag = True
        
    
        
if __name__ == "__main__":
    # Petit stub de test local
    table = Table()
    table.launch_scan()
    print("Cartes joueur:", table.cards.me_cards())
    

__all__ = ["Table"]

```
### objet/state/__init__.py
```python
"""Package regroupant les différents états du domaine."""
from .buttons import ButtonsState
from .capture import CaptureState
from .cards import CardsState
from .metrics import MetricsState
from .utils import extract_scan_value

__all__ = [
    "ButtonsState",
    "CaptureState",
    "CardsState",
    "MetricsState",
    "extract_scan_value",
]

```
### objet/state/buttons.py
```python
"""Gestion de l'état des boutons d'action."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from objet.entities.bouton import Bouton
from objet.state.utils import extract_scan_value


@dataclass
class ButtonsState:
    """Maintient les trois boutons d'action."""

    buttons: Dict[str, Bouton] = field(
        default_factory=lambda: {f"button_{i}": Bouton() for i in range(1, 4)}
    )

    def update_from_scan(self, scan_table: Mapping[str, Any]) -> None:
        for name, btn in self.buttons.items():
            raw_value = extract_scan_value(scan_table, name)
            btn.string_to_bouton(raw_value)

    def best_button(self) -> Optional[str]:
        best_name: Optional[str] = None
        best_gain: float = float("-inf")
        for name, btn in self.buttons.items():
            if btn.gain is None:
                continue
            if btn.gain > best_gain:
                best_gain = btn.gain
                best_name = name
        return best_name


__all__ = ["ButtonsState"]

```
### objet/state/capture.py
```python
"""Gestion de l'état des captures et paramètres OCR."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from objet.entities.card import Card


@dataclass
class CaptureState:
    """Paramètres liés aux captures et aux zones OCR."""

    table_capture: Dict[str, Any] = field(default_factory=dict)
    regions: "OrderedDict[str, Any]" = field(default_factory=OrderedDict)
    templates: Dict[str, Any] = field(default_factory=dict)
    reference_path: Optional[str] = None
    card_observations: Dict[str, Card] = field(default_factory=dict)
    workflow: Optional[str] = None

    def update_from_coordinates(
        self,
        *,
        table_capture: Optional[Mapping[str, Any]] = None,
        regions: Optional[Mapping[str, Any]] = None,
        templates: Optional[Mapping[str, Any]] = None,
        reference_path: Optional[str] = None,
    ) -> None:
        if table_capture is not None:
            self.table_capture = dict(table_capture)
        if regions is not None:
            self.regions = OrderedDict(regions)
        if templates is not None:
            self.templates = dict(templates)
        if reference_path is not None:
            self.reference_path = reference_path

    def record_observation(self, base_key: str, observation: Card) -> None:
        self.card_observations[base_key] = observation

    @property
    def size(self) -> Optional[List[int]]:
        size = self.table_capture.get("size") if isinstance(self.table_capture, dict) else None
        if isinstance(size, (list, tuple)) and len(size) == 2:
            return [int(size[0]), int(size[1])]
        return None

    @property
    def ref_offset(self) -> Optional[List[int]]:
        offset = self.table_capture.get("ref_offset") if isinstance(self.table_capture, dict) else None
        if isinstance(offset, (list, tuple)) and len(offset) == 2:
            return [int(offset[0]), int(offset[1])]
        return None


__all__ = ["CaptureState"]

```
### objet/state/cards.py
```python
# objet/state/cards.py
"""Gestion de l'état des cartes de la table.

- 5 cartes de board
- 2 cartes pour le héros
- Coordonnées injectées à la déclaration à partir de coordinates.json.
"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from objet.entities.card import Card
from objet.utils.calibration import Region, load_coordinates , bbox_from_region

CardBox = Tuple[int, int, int, int]

# Même défaut que dans _utils.load_coordinates
DEFAULT_COORD_PATH = Path("config/PMU/coordinates.json")




@dataclass
class CardsState:
    """
    Regroupe les cartes du board et du joueur, avec coordonnées injectées.

    - `coord_path` permet de surcharger le fichier de coordonnées si besoin
      (par défaut : config/PMU/coordinates.json).
    - Si `board` / `me` ne sont pas fournis, ils sont construits automatiquement
      à partir de `load_coordinates(coord_path)`.
    """

    coord_path: Path | str = DEFAULT_COORD_PATH
    board: List[Card] = field(default_factory=list)
    me: List[Card] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Cas où on injecte manuellement des cartes : on ne touche à rien.
        if self.board and self.me:
            return

        regions, templates_resolved, _ = load_coordinates(self.coord_path)


        if not self.board:
            self.board = [
                Card(card_coordinates_value=bbox_from_region(regions.get(f"board_card_{i}_number")),
                      card_coordinates_suit=bbox_from_region(regions.get(f"board_card_{i}_symbol")),)
                for i in range(1, 6)
            ]

        if not self.me:
            self.me = [
                Card(card_coordinates_value=bbox_from_region(regions.get(f"player_card_{i}_number")),
                      card_coordinates_suit=bbox_from_region(regions.get(f"player_card_{i}_symbol")),)
                for i in range(1, 3)
            ]

    # --- API pratique pour le reste du code ----------------------------------

    def me_cards(self) -> List[Card]:
        """Retourne les entités Card du joueur (avec value/suit/poker_card)."""
        return self.me

    def board_cards(self) -> List[Card]:
        """Retourne les entités Card du board."""
        return self.board


        
    def reset(self) -> None:
        """Réinitialise l'état de toutes les cartes."""
        for card in self.me + self.board:
            card.reset()
            
if __name__ == "__main__":
    # Petit stub de test local
    cards_state = CardsState()
    print(cards_state.me[0])

__all__ = ["CardsState"]

```
### objet/state/metrics.py
```python
"""Gestion des métriques numériques de la table."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import tool
from objet.state.utils import extract_scan_value


@dataclass
class MetricsState:
    """Regroupe les métriques numériques."""

    pot: Optional[float] = None
    fond: Optional[float] = None
    chance_win_0: Optional[float] = None
    chance_win_x: Optional[float] = None
    player_money: Dict[str, Optional[float]] = field(
        default_factory=lambda: {f"J{i}": None for i in range(1, 6)}
    )
    players_count: int = 0

    def update_from_scan(self, scan_table: Mapping[str, Any]) -> None:
        self.pot = tool.convert_to_float(extract_scan_value(scan_table, "pot"))
        self.fond = tool.convert_to_float(extract_scan_value(scan_table, "fond"))
        for key in list(self.player_money.keys()):
            raw_key = f"player_money_{key}"
            self.player_money[key] = tool.convert_to_float(extract_scan_value(scan_table, raw_key))
        self.players_count = sum(
            1 for money in self.player_money.values() if money not in (None, 0)
        )


__all__ = ["MetricsState"]

```
### objet/state/utils.py
```python
"""Utilitaires communs pour la gestion des états."""
from __future__ import annotations

from typing import Any, Mapping, Optional


def extract_scan_value(scan_table: Mapping[str, Any], key: str) -> Optional[str]:
    raw = scan_table.get(key)
    if isinstance(raw, Mapping):
        return raw.get("value")
    return raw


__all__ = ["extract_scan_value"]

```
### objet/utils/__init__.py
```python
"""Utility modules shared across the application and scripts."""

__all__ = ["calibration", "pyauto"]

```
### objet/utils/calibration.py
```python
"""Shared calibration helpers for screen capture tools.

This module centralises the JSON loading/parsing logic shared by the
calibration utilities as well as a couple of small image helpers.  The
functions remain dependency-light so they can be used from both
application code and standalone scripts without creating circular
imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from PIL import Image

__all__ = [
    "Region",
    "coerce_int",
    "clamp_bbox",
    "clamp_top_left",
    "resolve_templates",
    "load_coordinates",
    "extract_patch",
    "collect_card_patches",
]


@dataclass(frozen=True)
class Region:
    """Simple container describing a rectangular capture zone."""

    key: str
    group: str
    top_left: Tuple[int, int]
    size: Tuple[int, int]
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return the JSON-compatible representation of the region."""

        payload = {"group": self.group, "top_left": list(self.top_left)}
        payload.update(self.meta)
        return payload


def coerce_int(value: Any, default: int = 0) -> int:
    """Convert *value* to an int, falling back to *default* on failure."""

    try:
        return int(round(float(value)))
    except Exception:
        return default


def clamp_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Tuple[int, int, int, int]:
    """Clamp a bounding box to the image boundaries."""

    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def clamp_top_left(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int]:
    """Ensure the rectangle starting at (x, y) stays inside (W, H)."""

    if W <= 0 or H <= 0:
        return x, y
    x = max(0, min(x, max(0, W - w)))
    y = max(0, min(y, max(0, H - h)))
    return x, y


def resolve_templates(templates: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Resolve *templates* aliases and expose a uniform mapping."""

    def get_size(name: str, seen: Optional[set] = None) -> Tuple[int, int]:
        if seen is None:
            seen = set()
        if name in seen:
            return 0, 0
        seen.add(name)
        tpl = templates.get(name, {})
        if "size" in tpl:
            w, h = tpl.get("size", [0, 0])
            return coerce_int(w), coerce_int(h)
        alias = tpl.get("alias_of")
        if alias:
            return get_size(str(alias), seen)
        return 0, 0

    def get_type(name: str, seen: Optional[set] = None) -> str:
        if seen is None:
            seen = set()
        if name in seen:
            return ""
        seen.add(name)
        tpl = templates.get(name, {})
        typ = tpl.get("type")
        if typ:
            return str(typ)
        alias = tpl.get("alias_of")
        if alias:
            return get_type(str(alias), seen)
        return ""

    def get_layout(name: str, seen: Optional[set] = None) -> Dict[str, Any]:
        if seen is None:
            seen = set()
        if name in seen:
            return {}
        seen.add(name)
        tpl = templates.get(name, {})
        layout = tpl.get("layout")
        if isinstance(layout, Mapping):
            return dict(layout)
        alias = tpl.get("alias_of")
        if alias:
            return get_layout(str(alias), seen)
        return {}

    resolved: Dict[str, Dict[str, Any]] = {}
    for group in templates.keys():
        size = get_size(group)
        typ = get_type(group)
        layout = get_layout(group)
        payload = {"size": [size[0], size[1]], "type": typ}
        if layout:
            payload["layout"] = layout
        resolved[group] = payload
    return resolved


def _normalise_region_entry(key: str, raw: Mapping[str, Any], templates: Mapping[str, Dict[str, Any]]) -> Region:
    group = str(raw.get("group", ""))
    top_left = raw.get("top_left", [0, 0])
    size = templates.get(group, {}).get("size", [0, 0])
    meta = {k: v for k, v in raw.items() if k not in {"group", "top_left"}}
    return Region(
        key=key,
        group=group,
        top_left=(coerce_int(top_left[0]), coerce_int(top_left[1])),
        size=(coerce_int(size[0]), coerce_int(size[1])),
        meta=dict(meta),
    )


# --- cache coordinates.json en mémoire ---------------------------------------

# Chemin par défaut : racine/config/PMU/coordinates.json
DEFAULT_COORDINATES_PATH = Path("config/PMU/coordinates.json")

_CoordinatesCacheEntry = Tuple[Dict[str, "Region"], Dict[str, Dict[str, Any]], Dict[str, Any]]
_COORDINATES_CACHE: Dict[Path, _CoordinatesCacheEntry] = {}


def load_coordinates(
    path: Path | str = DEFAULT_COORDINATES_PATH,
) -> Tuple[Dict[str, "Region"], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Load a coordinates.json file.

    Retourne ``(regions, templates_resolved, table_capture)`` où ``regions``
    mappe les clés vers des :class:`Region`.

    - *path* est optionnel, par défaut `config/PMU/coordinates.json`.
    - Les résultats sont mis en cache par chemin absolu pour éviter
      de rouvrir et reparser le JSON à chaque appel.
    """
    coord_path = Path(path)
    key = coord_path.resolve()

    # 1) cache mémoire
    cached = _COORDINATES_CACHE.get(key)
    if cached is not None:
        return cached

    # 2) chargement disque
    with coord_path.open("r", encoding="utf-8") as fh:
        payload: Dict[str, Any] = json.load(fh)

    templates = payload.get("templates", {})
    resolved = resolve_templates(templates)
    raw_regions = payload.get("regions", {})
    regions: Dict[str, Region] = {
        r_key: _normalise_region_entry(r_key, raw, resolved)
        for r_key, raw in raw_regions.items()
    }
    table_capture = payload.get("table_capture", {})

    result: _CoordinatesCacheEntry = (regions, resolved, table_capture)
    _COORDINATES_CACHE[key] = result
    return result


def extract_patch(image: Image.Image, top_left: Tuple[int, int], size: Tuple[int, int], pad: int = 4) -> Image.Image:
    """Crop ``image`` around ``top_left``/``size`` with a soft *pad*."""

    x, y = map(int, top_left)
    w, h = map(int, size)
    width, height = image.size
    x1, y1 = x - pad, y - pad
    x2, y2 = x + w + pad, y + h + pad
    x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, width, height)
    return image.crop((x1, y1, x2, y2))


def _region_group(region: Region | Mapping[str, Any]) -> str:
    return region.group if isinstance(region, Region) else str(region.get("group", ""))


def _region_geometry(region: Region | Mapping[str, Any]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    if isinstance(region, Region):
        return region.top_left, region.size
    top_left = region.get("top_left", [0, 0])
    size = region.get("size", [0, 0])
    tl_x = coerce_int(top_left[0])
    tl_y = coerce_int(top_left[1])
    width, height = 0, 0
    if isinstance(size, Iterable):
        values = list(size)
        if values:
            width = coerce_int(values[0])
        if len(values) >= 2:
            height = coerce_int(values[1])
    return (tl_x, tl_y), (width, height)


def collect_card_patches(
    table_img: Image.Image,
    regions: Mapping[str, Region | Mapping[str, Any]],
    *,
    pad: int = 4,
    groups_numbers: Tuple[str, ...] = ("player_card_number", "board_card_number"),
    groups_suits: Tuple[str, ...] = ("player_card_symbol", "board_card_symbol"),
) -> Dict[str, Tuple[Image.Image, Image.Image]]:
    """Return ``{base_key: (number_patch, suit_patch)}`` for recognised card regions.

    The implementation walks through *regions* a single time and relies on
    :func:`extract_patch` to perform the actual cropping, keeping the
    bookkeeping logic light-weight while still supporting both
    :class:`Region` objects and plain ``dict`` entries.
    """

    slots: Dict[str, Dict[str, Image.Image]] = {}

    for key, region in regions.items():
        group = _region_group(region)
        slot: Optional[str] = None
        base_key: Optional[str] = None
        if group in groups_numbers:
            slot = "number"
            base_key = key.replace("_number", "")
        elif group in groups_suits:
            slot = "symbol"
            base_key = key.replace("_symbol", "")
        else:
            continue

        if not slot or not base_key:
            continue

    out: Dict[str, Tuple[Image.Image, Image.Image]] = {}
    for base, mapping in pairs.items():
        if "number" in mapping and "symbol" in mapping:
            out[base] = (mapping["number"], mapping["symbol"])
    return out




BBox = Tuple[int, int, int, int]  # (x, y, w, h)

def bbox_from_region(region: Optional["Region"]) -> Optional[BBox]:
 
    x, y = region.top_left  
    size = region.size

    w, h = size
    return x, y, w, h

```
### objet/utils/pyauto.py
```python
"""Helpers built around :mod:`pyautogui` usable from app code and scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import pyautogui
from PIL import Image

HaystackType = Union[np.ndarray, Image.Image]
NeedleType = Union[str, Path, Image.Image]

__all__ = ["locate_in_image"]


def _to_pil_rgb(img: HaystackType, assume_bgr: bool = False) -> Image.Image:
    """Convert ``img`` into a :class:`PIL.Image.Image` in RGB mode."""

    if isinstance(img, Image.Image):
        return img.convert("RGB")

    if not isinstance(img, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(img)}")

    arr = img
    if arr.ndim == 2:
        # grayscale -> RGB
        return Image.fromarray(arr).convert("RGB")

    if arr.ndim == 3 and arr.shape[2] == 3:
        if assume_bgr:
            arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        else:
            arr_rgb = arr
        return Image.fromarray(arr_rgb)

    raise ValueError(f"Unsupported array shape: {arr.shape}")


def locate_in_image(
    haystack: HaystackType,
    needle: NeedleType,
    *,
    assume_bgr: bool = False,
    grayscale: bool = False,
    confidence: float = 0.9,
) -> Optional[Tuple[int, int, int, int]]:
    """Locate ``needle`` inside ``haystack`` using ``pyautogui.locate``."""

    haystack_pil = _to_pil_rgb(haystack, assume_bgr=assume_bgr)

    if isinstance(needle, (str, Path)):
        needle_pil = Image.open(needle).convert("RGB")
    elif isinstance(needle, Image.Image):
        needle_pil = needle.convert("RGB")
    else:
        raise TypeError(f"Unsupported needle type: {type(needle)}")

    box = pyautogui.locate(
        needle_pil,
        haystack_pil,
        grayscale=grayscale,
        confidence=confidence,
    )
    if box is None:
        return None

    return int(box.left), int(box.top), int(box.width), int(box.height)

```
### scripts/_utils.py
```python
"""Compatibility layer for calibration helpers used by legacy scripts."""
from __future__ import annotations

from objet.utils.calibration import (
    Region,
    clamp_bbox,
    clamp_top_left,
    coerce_int,
    extract_patch,
    collect_card_patches,
    load_coordinates,
    resolve_templates,
)

__all__ = [
    "Region",
    "coerce_int",
    "clamp_bbox",
    "clamp_top_left",
    "resolve_templates",
    "load_coordinates",
    "extract_patch",
    "collect_card_patches",
]

```
### scripts/capture_cards.py
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cartes — extraction, matching, contrôleur de table et outils vidéo/labeling.

Regroupe les anciennes fonctionnalités de:
- cards_core.py
- cards_validate.py
- controller.py
- capture_source.py
- labeler_cli.py
- run_video_validate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

# Accès modules du dépôt (pour exécution directe depuis scripts/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
for root in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from _utils import collect_card_patches, load_coordinates
from crop_core import crop_from_size_and_offset
from objet.scanner.cards_recognition import (
    CardObservation,
    TemplateIndex,
    is_card_present,
    recognize_number_and_suit,
)

if TYPE_CHECKING:  # hints uniquement
    from objet.services.game import Game


# ==============================
# Modèle / observations de cartes
# ==============================

# Les classes et fonctions de reconnaissance sont fournies par
# ``objet.scanner.cards_recognition`` pour éviter les imports circulaires.



# ==============================
# cards_validate — CLI de vérification basique sur une image
# ==============================


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


def main_cards_validate(argv: Optional[list] = None) -> int:
    """Vérifie l'extraction + matching des cartes pour un jeu donné.

    Utilise un screenshot déjà croppé (test_crop_result.* dans config/<game>/).
    """

    parser = argparse.ArgumentParser(description="Vérifie l'extraction + matching des cartes pour un jeu")
    parser.add_argument("--game", default="PMU")
    parser.add_argument("--game-dir")
    parser.add_argument("--dump", action="store_true", help="Sauver extraits dans debug/")
    parser.add_argument("--num-th", type=float, default=0.6, help="Seuil score pour numbers (0..1)")
    parser.add_argument("--suit-th", type=float, default=0.6, help="Seuil score pour suits (0..1)")
    parser.add_argument("--pad", type=int, default=4)
    args = parser.parse_args(argv)

    auto = _auto_paths_for_game(args.game, args.game_dir)
    table_path: Optional[Path] = auto["table"]
    coords_path: Path = auto["coords"]
    cards_root: Path = auto["cards_root"]

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
        expect_numbers=["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"],
        expect_suits=["hearts", "diamonds", "clubs", "spades"],
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

    # Import local pour limiter les risques d'import circulaire
    from objet.services.game import Game  # type: ignore[import]

    game = Game.for_script(Path(__file__).name)
    regions, resolved, table_capture = load_coordinates(coords_path)
    game.update_from_capture(
        table_capture=table_capture,
        regions={k: {"group": r.group, "top_left": r.top_left, "size": r.size} for k, r in regions.items()},
        templates=resolved,
        reference_path=str(table_path) if table_path else None,
    )

    # 3) Extraire patches cartes
    pairs = collect_card_patches(table_img, regions, pad=int(args.pad))
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

        obs = CardObservation(value=val, suit=suit, value_score=s_val, suit_score=s_suit, source="capture")
        if hasattr(game, "add_card_observation"):
            game.add_card_observation(base_key, obs)

        hit_val = val is not None and s_val >= float(args.num_th)
        hit_suit = suit is not None and s_suit >= float(args.suit_th)
        status = "OK" if (hit_val and hit_suit) else "LOW"
        print(f"{base_key}: {status}  value={val} ({s_val:.3f})  suit={suit} ({s_suit:.3f})")
        if args.dump:
            _save_png(debug_dir / f"{base_key}_number.png", patch_num)
            _save_png(debug_dir / f"{base_key}_symbol.png", patch_suit)
        if not (hit_val and hit_suit):
            ok = False

    # Résumé éventuel si Game expose des cartes formatées
    if hasattr(game, "cards") and hasattr(game.cards, "as_strings"):
        summary = game.cards.as_strings()
        player = summary.get("player") or []
        board = summary.get("board") or []
        print("Résumé Game → joueur:", ", ".join(player))
        print("Résumé Game → board:", ", ".join(board))

    return 0 if ok else 1


# ==============================
# État de table / contrôleur runtime
# ==============================


@dataclass
class CardState:
    value: Optional[str] = None
    suit: Optional[str] = None
    value_score: float = 0.0
    suit_score: float = 0.0
    stable: int = 0  # nb frames consécutifs où l'observation est identique et au-dessus des seuils
    last_seen: int = -1


class TableState:
    def __init__(self) -> None:
        self.cards: Dict[str, CardState] = {}  # base_key -> state

    def update(
        self,
        base_key: str,
        obs: CardObservation,
        frame_idx: int,
        *,
        num_th: float,
        suit_th: float,
        require_k: int = 2,
    ) -> bool:
        """Met à jour l'état d'une carte.

        Retourne True si une nouvelle valeur stabilisée (changement) est atteinte.
        """

        confident = (
            obs.value is not None
            and obs.value_score >= num_th
            and obs.suit is not None
            and obs.suit_score >= suit_th
        )

        st = self.cards.get(base_key, CardState())

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
        return {
            k: {
                "value": v.value,
                "suit": v.suit,
                "value_score": v.value_score,
                "suit_score": v.suit_score,
                "stable": v.stable,
            }
            for k, v in self.cards.items()
        }


class TableController:
    """Orchestrateur runtime (capture → crop → extract → match → état)."""

    def __init__(self, game_dir: Path, game_state: Optional["Game"] = None) -> None:
        # Import local pour éviter les imports circulaires si objet.services.game
        # importe à son tour des scripts.
        from objet.services.game import Game  # type: ignore[import]

        self.game_dir = Path(game_dir)
        self.coords_path = self.game_dir / "coordinates.json"
        self.ref_path = self._first_of("me", (".png", ".jpg", ".jpeg"))

        self.game: Game = game_state or Game.for_script(Path(__file__).name)

        self.regions, self.templates, table_capture = load_coordinates(self.coords_path)
        self.game.update_from_capture(
            table_capture=table_capture,
            regions={k: {"group": r.group, "top_left": r.top_left, "size": r.size} for k, r in self.regions.items()},
            templates=self.templates,
            reference_path=str(self.ref_path) if self.ref_path else None,
        )

        self.size, self.ref_offset = self._load_capture_params()

        # runtime caches
        self.ref_img: Optional[Image.Image] = (
            Image.open(self.ref_path).convert("RGBA") if self.ref_path else None
        )
        self.idx = TemplateIndex(self.game_dir / "cards")
        self.idx.load()
        self.state = TableState()

    def _first_of(self, stem: str, exts: Iterable[str]) -> Optional[Path]:
        for ext in exts:
            p = self.game_dir / f"{stem}{ext}"
            if p.exists():
                return p
        return None

    def _load_capture_params(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        tc = self.game.table.captures.table_capture
        size = tc.get("size", [0, 0]) if isinstance(tc, dict) else [0, 0]
        ref_offset = tc.get("ref_offset", [0, 0]) if isinstance(tc, dict) else [0, 0]
        return (int(size[0]), int(size[1])), (int(ref_offset[0]), int(ref_offset[1]))

    def process_frame(
        self,
        frame_rgba: Image.Image,
        frame_idx: int,
        *,
        num_th: float = 0.6,
        suit_th: float = 0.6,
        require_k: int = 2,
    ) -> Dict[str, Dict[str, object]]:
        """Traite un frame et retourne un snapshot d'état de table."""

        # 1) crop table via size + ref_offset
        crop, _ = crop_from_size_and_offset(
            frame_rgba, self.size, self.ref_offset, reference_img=self.ref_img
        )

        # 2) extractions number/symbol
        pairs = collect_card_patches(crop, self.regions, pad=4)

        # 3) matching + mise à jour d'état
        for base_key, (patch_num, patch_suit) in pairs.items():
            if not is_card_present(patch_num):
                continue
            val, suit, s_val, s_suit = recognize_number_and_suit(patch_num, patch_suit, self.idx)
            obs = CardObservation(val, suit, s_val, s_suit, source="capture")
            self.state.update(
                base_key,
                obs,
                frame_idx,
                num_th=float(num_th),
                suit_th=float(suit_th),
                require_k=int(require_k),
            )
            if hasattr(self.game, "add_card_observation"):
                self.game.add_card_observation(base_key, obs)

        return self.state.snapshot()


# ==============================
# capture_source — vidéo → frames PIL
# ==============================


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


# ==============================
# labeler_cli — collecte des inconnus & labellisation simple
# ==============================


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


# ==============================
# run_video_validate — CLI vidéo → détection en ligne + stockage inconnus
# ==============================


def main_video_validate(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Valide détection cartes sur une vidéo (crop+match+mémoire)"
    )
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

        snap = ctrl.process_frame(
            frame,
            i,
            num_th=float(args.num_th),
            suit_th=float(args.suit_th),
            require_k=int(args.require_k),
        )

        # stocker les inconnus (option basique: si zone présente mais non stable)
        crop, _ = crop_from_size_and_offset(frame, ctrl.size, ctrl.ref_offset, reference_img=ctrl.ref_img)
        pairs = collect_card_patches(crop, ctrl.regions, pad=4)
        for base_key, (patch_num, patch_suit) in pairs.items():
            if not is_card_present(patch_num):
                continue
            st = ctrl.state.cards.get(base_key)
            if not st or st.stable == 0:
                # pas encore reconnu → on garde un échantillon pour labellisation ultérieure
                sink.save_number(base_key, patch_num, i)
                sink.save_suit(base_key, patch_suit, i)

        # Affiche un résumé court
        pretty = ", ".join(
            [
                f"{k}:{v['value'] or '?'}-{v['suit'] or '?'}(s{v['stable']})"
                for k, v in sorted(snap.items())
            ]
        )
        print(f"frame {i:05d}: {pretty}")

    return 0


# --- Helpers: default video path + dedupe hash (utiles pour run_video_validate) ---


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


# ==============================
# Point d'entrée unifié
# ==============================

if __name__ == "__main__":
    argv = sys.argv[1:]
    # Heuristique simple : si --video est présent, on lance le mode vidéo,
    # sinon la validation sur image fixe.
    if "--video" in argv:
        raise SystemExit(main_video_validate(argv))
    else:
        raise SystemExit(main_cards_validate(argv))

```
### scripts/crop_core.py
```python
import sys

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import json
import numpy as np
from PIL import Image


# ==============================
# crop_core.py – Coeur mémoire (size + ref_offset)
# ==============================
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))



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


def _match_top_left_and_score(
    src_img: Image.Image,
    tmpl_img: Image.Image,
    method: int = cv2.TM_CCOEFF_NORMED,
) -> Tuple[Tuple[int, int], float]:
    """Retourne ((x,y), score) du meilleur match de tmpl_img dans src_img.
    Le score est normalisé: plus haut = meilleur, quel que soit le method.
    """
    src_gray = cv2.cvtColor(np.array(src_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    tpl_gray = cv2.cvtColor(np.array(tmpl_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    res = cv2.matchTemplate(src_gray, tpl_gray, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
        score = 1.0 - float(min_val)
        loc = min_loc
    else:
        score = float(max_val)
        loc = max_loc
    return (int(loc[0]), int(loc[1])), float(score)


# ---------- Paramètres internes (non exposés) ----------
_REF_METHOD = cv2.TM_CCOEFF_NORMED
_REF_THRESHOLD = 0.80          # seuil de détection plein écran
_INSIDE_THRESHOLD = 0.80       # seuil de détection dans le crop
_GEOM_TOL = 1                  # tolérance géométrique intra-crop (px)


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
    reference_img: Image.Image,  # requis
) -> Tuple[Optional[Image.Image], Optional[Tuple[int,int]]]:
    """Retourne (crop, (x0,y0)).

    Comportement tolérant : si l'ancre n'est pas détectée sur le plein écran OU
    si elle n'est pas validée *dans* le crop (score/tolérance internes), retourne (None, None).

    - `size` = (W,H) de la fenêtre à extraire.
    - `ref_offset` = (ox,oy) = position RELATIVE du gabarit `reference_img` *dans* la fenêtre
      (distance depuis le coin haut-gauche de la fenêtre jusqu'au coin haut-gauche de `reference_img`).
    - `reference_img` sert à retrouver la position absolue de l'ancre dans le screenshot.

    Coin du crop = REF_ABS - ref_offset. Clamp si nécessaire.
    """
    W, H = int(size[0]), int(size[1])
    ox, oy = int(ref_offset[0]), int(ref_offset[1])
    if W <= 0 or H <= 0:
        raise ValueError("Invalid size; width/height must be > 0")

    # 1) Détection de l'ancre sur le screenshot
    (rx, ry), score = _match_top_left_and_score(screenshot_img, reference_img, method=_REF_METHOD)
    if score < _REF_THRESHOLD:
        return None, None

    # 2) Calcul du crop (avec clamp)
    x0_raw, y0_raw = rx - ox, ry - oy
    x0, y0 = _clamp_origin(x0_raw, y0_raw, (W, H), screenshot_img.size)
    x1, y1 = x0 + W, y0 + H
    crop = screenshot_img.crop((x0, y0, x1, y1))

    # 3) Validation intra-crop (obligatoire, non paramétrable)
    ref_w, ref_h = reference_img.size
    ax_exp, ay_exp = rx - x0, ry - y0  # position attendue de l’ancre dans le crop

    # L’ancre doit tenir entièrement dans le crop
    if not (0 <= ax_exp <= W - ref_w and 0 <= ay_exp <= H - ref_h):
        return None, None

    # Matching dans le crop
    (ax_found, ay_found), inside_score = _match_top_left_and_score(crop, reference_img, method=_REF_METHOD)
    ok_score = inside_score >= _INSIDE_THRESHOLD
    ok_geom = (abs(ax_found - ax_exp) <= _GEOM_TOL) and (abs(ay_found - ay_exp) <= _GEOM_TOL)

    if not (ok_score and ok_geom):
        return None, None

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
    reference_img: Image.Image,
    geom_tol: int = 1,
    pix_tol: int = 0,
) -> Tuple[bool, Dict[str, float]]:
    """Vérifie l'ALIGNEMENT géométrique:
       - prédit (px,py) via (size, ref_offset, ancre)
       - mesure (mx,my) en matchant expected_img dans screenshot
       OK si |px-mx|<=geom_tol et |py-my|<=geom_tol.
       Ajoute stats pixel (max_diff/mean_diff) à titre informatif.
    """
    # 1) prédiction via runtime (tolérant)
    crop_pred, origin = crop_from_size_and_offset(
        screenshot_img, size, ref_offset, reference_img=reference_img
    )
    if crop_pred is None or origin is None:
        return False, {"reason": "anchor_not_found_or_invalid"}

    px, py = origin

    # 2) mesure via matching
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
    
    from objet.services.game import Game
    game = Game.for_script(Path(__file__).name)

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
        print("[infer] crop_top_left:", crop_pos, "ref_point:", ref_pos, "-> ref_offset:", ref_off_inf, "size:", size_inf)

    # 2) Écrit JSON (taille + ref_offset)
    from __main__ import save_capture_json  # adjust import if split
    save_capture_json(output_path, size, ref_offset)
    print("Wrote:", output_path)
    print("table_capture.size:", list(size), "table_capture.ref_offset:", list(ref_offset))
    game.update_from_capture(table_capture={"size": list(size), "ref_offset": list(ref_offset)})

    # 3) Vérification répétée (géométrie en priorité)
    from __main__ import verify_geom, crop_from_size_and_offset
    ok_count = 0
    last_stats = {}
    for i in range(int(args.runs)):
        ok, stats = verify_geom(
            scr, exp, size, ref_offset,
            reference_img=ref,
            geom_tol=int(args.geom_tol),
            pix_tol=int(args.pix_tol),
        )
        last_stats = stats
        print(f"run {i+1:02d}: ", "OK" if ok else "FAIL", stats)
        if ok:
            ok_count += 1

    # 4) Sauvegarde debug (si calcul possible)
    crop, origin = crop_from_size_and_offset(scr, size, ref_offset, reference_img=ref)
    debug_path = _with_debug_suffix(expected_path)
    if crop is not None and origin is not None:
        _save_any(debug_path, crop)
        print("Wrote computed crop (debug):", debug_path, "origin:", origin)
        if args.write_crop:
            _save_any(Path(args.write_crop), crop)
            print("Wrote computed crop (custom):", args.write_crop)
    else:
        print("No debug crop written: anchor not found/validated.")

    print(f"Summary: {ok_count}/{args.runs} OK (geom_tol={args.geom_tol}, pix_tol={args.pix_tol})")
    print("Game capture context:", game.table.captures.table_capture)
    return 0 if ok_count == int(args.runs) else 2


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))

```
### scripts/Crop_Video_Frames.py
```python
# crop_video_frames.py — extrait une image crop de la table toutes les 1s depuis une vidéo de test
# Dossier vidéo par défaut : config/PMU/debug/cards_video/cards_video.*
# Sortie par défaut :       config/PMU/debug/crops/
# Dépendances: opencv-python, pillow, numpy
#   pip install opencv-python pillow numpy

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple
import random

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
    args = parser.parse_args(argv)

    game_dir = Path(args.game_dir)
    out_dir = Path(args.out) if args.out else (game_dir / "debug" / "crops")
    out_dir.mkdir(parents=True, exist_ok=True)

    size, ref_offset, ref_path = _load_capture_params(game_dir)
    ref_img = Image.open(ref_path).convert("RGBA")
    game = Game.for_script(Path(__file__).name)
    game.update_from_capture(
        table_capture={"size": list(size), "ref_offset": list(ref_offset)},
        reference_path=str(ref_path),
    )

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
        capture_size = tuple(game.table.captures.size or size)
        capture_offset = tuple(game.table.captures.ref_offset or ref_offset)
        crop, origin = crop_from_size_and_offset(frame_img, capture_size, capture_offset, reference_img=ref_img)
        # nom aléatoire dans [1, 10000]
        n = random.randint(1, 10000)
        fname = f"crop_{n}.png"
        out_path = out_dir / fname
        crop.save(out_path)
        count += 1
        print(f"saved {out_path.name} origin={origin} size={crop.size}")

    cap.release()
    print(f"Done. {count} crops written to {out_dir}")
    print("Game capture context:", game.table.captures.table_capture)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))

```
### scripts/identify_card.py
```python
#!/usr/bin/env python3
"""identify_card.py — assistant de labellisation des cartes.

Parcourt les crops de tables dans ``config/<jeu>/debug/crops``, extrait les
patches *number* et *suit* selon `coordinates.json`, tente une reco par gabarits,
**auto-skip** des cartes déjà connues (log) avec un seuil strict, et **ne demande
que la partie inconnue** (valeur OU couleur) lorsque l’autre est déjà fiable.

Usage minimal:
    python scripts/identify_card.py --game PMU

Options utiles:
  --crops-dir     Dossier d’entrée (défaut: config/<game>/debug/crops)
  --threshold     Score min (0-1) pour considérer une reco comme fiable (def 0.92)
  --strict        Score min (0-1) pour auto-skip sans UI (def 0.985)
  --force-all     Forcer l’UI même si auto-skip serait possible
  --trim          Bordure rognée (px) autour des patches AVANT reco & sauvegarde (def 6)
"""

from __future__ import annotations

import argparse
import itertools
import sys
import tkinter as tk
import tkinter.messagebox as messagebox
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Accès modules du dépôt
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from objet.scanner.cards_recognition import TemplateIndex, is_card_present, recognize_number_and_suit
from _utils import collect_card_patches, load_coordinates

DEFAULT_NUMBERS: Sequence[str] = (
    "?",
    "A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2",
)
DEFAULT_SUITS: Sequence[str] = ("?", "spades", "hearts", "diamonds", "clubs")


@dataclass
class CardSample:
    source_path: Path
    base_key: str
    number_patch: Image.Image
    suit_patch: Image.Image
    number_suggestion: Optional[str]
    suit_suggestion: Optional[str]
    number_score: float
    suit_score: float
    num_known: bool  # True si number est fiable (>= threshold)
    suit_known: bool # True si suit   est fiable (>= threshold)

    def trimmed_number(self, border: int) -> Image.Image:
        return _trim_patch(self.number_patch, border)

    def trimmed_suit(self, border: int) -> Image.Image:
        return _trim_patch(self.suit_patch, border)


def _trim_patch(img: Image.Image, border: int) -> Image.Image:
    if border <= 0:
        return img
    w, h = img.size
    if w <= border * 2 or h <= border * 2:
        return img
    # rogne de "border" px sur TOUTES les bordures
    return img.crop((border, border, w - border, h - border))


def _iter_capture_files(directory: Path) -> Iterable[Path]:
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        yield from sorted(directory.glob(ext))


def _unique_sorted(values: Iterable[str], defaults: Sequence[str]) -> List[str]:
    seen = list(defaults)
    for val in values:
        if val and val not in seen:
            seen.append(val)
    return seen


def collect_card_samples(
    table_paths: Iterable[Path],
    regions: Dict[str, object],
    idx: TemplateIndex,
    *,
    trim_border: int,
    accept_threshold: float,
    strict_threshold: float,
    force_all: bool,
) -> Tuple[List[CardSample], int, int]:
    """Retourne (samples, total_cartes, reconnues_auto).

    - Calcul des scores via `recognize_number_and_suit` sur **patches rognés** (trim_border).
    - Si score >= strict_threshold pour number/suit, on considère la partie **connue** sans UI.
    - Si les 2 parties sont connues et `force_all` False → auto-skip + log.
    - Sinon on ajoute un sample et l’UI ne demandera **que la partie inconnue**.
    """
    samples: List[CardSample] = []
    total_cards = 0
    auto_ok = 0

    for img_path in table_paths:
        try:
            with Image.open(img_path) as im:
                table_img = im.convert("RGB")
        except FileNotFoundError:
            continue

        card_pairs = collect_card_patches(table_img, regions, pad=0)
        for base_key, (num_patch, suit_patch) in card_pairs.items():
            if not is_card_present(num_patch, threshold=215, min_ratio=0.04):
                continue
            total_cards += 1

            # rognage AVANT reco
            trimmed_num = _trim_patch(num_patch, trim_border)
            trimmed_suit = _trim_patch(suit_patch, trim_border)

            suggestion_num, suggestion_suit, score_num, score_suit = recognize_number_and_suit(
                trimmed_num, trimmed_suit, idx
            )

            # deux niveaux de confiance : acceptable vs strict autoskip
            num_known = bool(suggestion_num) and float(score_num) >= float(accept_threshold)
            suit_known = bool(suggestion_suit) and float(score_suit) >= float(accept_threshold)
            num_strict = bool(suggestion_num) and float(score_num) >= float(strict_threshold)
            suit_strict = bool(suggestion_suit) and float(score_suit) >= float(strict_threshold)

            # logs informatifs
            if num_strict:
                print(f"DISCOVERED number={suggestion_num} ({score_num:.3f}) in {img_path.name} {base_key} → autoskip nombre")
            if suit_strict:
                print(f"DISCOVERED suit={suggestion_suit} ({score_suit:.3f}) in {img_path.name} {base_key} → autoskip couleur")

            if num_strict and suit_strict and not force_all:
                auto_ok += 1
                print(
                    f"AUTO OK: {img_path.name} {base_key} → number={suggestion_num} ({score_num:.2f}), "
                    f"suit={suggestion_suit} ({score_suit:.2f})"
                )
                continue

            # On garde le sample si au moins une partie n’atteint pas le strict
            samples.append(
                CardSample(
                    source_path=img_path,
                    base_key=base_key,
                    number_patch=num_patch,
                    suit_patch=suit_patch,
                    number_suggestion=suggestion_num,
                    suit_suggestion=suggestion_suit,
                    number_score=float(score_num),
                    suit_score=float(score_suit),
                    num_known=num_strict,   # connu = seuil strict
                    suit_known=suit_strict,
                )
            )

    return samples, total_cards, auto_ok


class DatasetWriter:
    def __init__(self, idx: TemplateIndex, cards_root: Path, trim_border: int) -> None:
        self.idx = idx
        self.cards_root = Path(cards_root)
        self.cards_root.mkdir(parents=True, exist_ok=True)
        (self.cards_root / "numbers").mkdir(parents=True, exist_ok=True)
        (self.cards_root / "suits").mkdir(parents=True, exist_ok=True)
        self.trim_border = trim_border
        self.counter = itertools.count(1)
        self.saved: List[Tuple[Path, Path]] = []

    def save(
        self,
        sample: CardSample,
        number_label: str,
        suit_label: str,
        *,
        save_number: bool = True,
        save_suit: bool = True,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """Enregistre les patches rognés; possibilité de ne sauver que l’inconnu."""
        n_img = sample.trimmed_number(self.trim_border)
        s_img = sample.trimmed_suit(self.trim_border)
        idx = next(self.counter)
        base_name = f"{sample.base_key}_{sample.source_path.stem}_{idx:04d}"
        num_path = self.cards_root / "numbers" / number_label / f"{base_name}.png"
        suit_path = self.cards_root / "suits" / suit_label / f"{base_name}.png"
        np_out: Optional[Path] = None
        sp_out: Optional[Path] = None
        if save_number:
            num_path.parent.mkdir(parents=True, exist_ok=True)
            n_img.save(num_path)
            self._update_index(number_label, n_img, is_number=True)
            np_out = num_path
        if save_suit:
            suit_path.parent.mkdir(parents=True, exist_ok=True)
            s_img.save(suit_path)
            self._update_index(suit_label, s_img, is_number=False)
            sp_out = suit_path
        self.saved.append((np_out or Path(), sp_out or Path()))
        return np_out, sp_out

    def _update_index(self, label: str, img: Image.Image, *, is_number: bool) -> None:
        gray = np.array(img.convert("L"))
        arr = TemplateIndex._prep(gray)
        store = self.idx.numbers if is_number else self.idx.suits
        store.setdefault(label, []).append(arr)


class LabelingApp:
    def __init__(
        self,
        samples: Sequence[CardSample],
        writer: DatasetWriter,
        *,
        number_choices: Sequence[str],
        suit_choices: Sequence[str],
        accept_threshold: float,
    ) -> None:
        self.samples = list(samples)
        self.writer = writer
        self.index = 0
        self.number_choices = list(number_choices)
        self.suit_choices = list(suit_choices)
        self.accept_threshold = float(accept_threshold)
        self.photo: Optional[ImageTk.PhotoImage] = None

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Labellisation des cartes")
        self.root.geometry("720x560")

        self.number_var = tk.StringVar(value=self.number_choices[0] if self.number_choices else "")
        self.suit_var = tk.StringVar(value=self.suit_choices[0] if self.suit_choices else "")

        self._build_ui()
        self._show_current()

    def _build_ui(self) -> None:
        top = ctk.CTkFrame(self.root)
        top.pack(fill="x", padx=12, pady=8)

        self.status_label = ctk.CTkLabel(top, text="")
        self.status_label.pack(side="left", padx=6)

        self.suggestion_label = ctk.CTkLabel(top, text="", anchor="e")
        self.suggestion_label.pack(side="right", padx=6)

        body = ctk.CTkFrame(self.root)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.image_label = ctk.CTkLabel(body, text="", compound="top")
        self.image_label.pack(side="top", pady=12)

        form = ctk.CTkFrame(body)
        form.pack(side="top", pady=8)

        # Lignes (on les cache/affiche dynamiquement)
        self.lbl_number = ctk.CTkLabel(form, text="Valeur")
        self.lbl_number.grid(row=0, column=0, padx=8, pady=6)
        self.number_menu = ctk.CTkOptionMenu(
            form,
            values=self.number_choices if self.number_choices else ["?"],
            variable=self.number_var,
        )
        self.number_menu.grid(row=0, column=1, padx=8, pady=6)

        self.lbl_suit = ctk.CTkLabel(form, text="Couleur")
        self.lbl_suit.grid(row=1, column=0, padx=8, pady=6)
        self.suit_menu = ctk.CTkOptionMenu(
            form,
            values=self.suit_choices if self.suit_choices else ["?"],
            variable=self.suit_var,
        )
        self.suit_menu.grid(row=1, column=1, padx=8, pady=6)

        btns = ctk.CTkFrame(body)
        btns.pack(side="bottom", pady=16)

        self.save_btn = ctk.CTkButton(btns, text="Enregistrer", command=self._on_save)
        self.save_btn.pack(side="left", padx=10)

        self.skip_btn = ctk.CTkButton(btns, text="Ignorer", command=self._on_skip)
        self.skip_btn.pack(side="left", padx=10)

    def run(self) -> None:
        self.root.mainloop()

    def _on_save(self) -> None:
        if self.index >= len(self.samples):
            self.root.destroy()
            return
        sample = self.samples[self.index]

        # fige/complète avec les suggestions quand c’est "connu"
        number_label = sample.number_suggestion if sample.num_known else self.number_var.get().strip()
        suit_label = sample.suit_suggestion if sample.suit_known else self.suit_var.get().strip()

        if (not number_label or number_label == "?") and (not sample.num_known):
            messagebox.showwarning("Label manquant", "Sélectionnez la valeur de la carte.")
            return
        if (not suit_label or suit_label == "?") and (not sample.suit_known):
            messagebox.showwarning("Label manquant", "Sélectionnez la couleur de la carte.")
            return

        # Sauve uniquement la partie inconnue
        save_number = not sample.num_known
        save_suit = not sample.suit_known
        np_out, sp_out = self.writer.save(
            sample, number_label, suit_label, save_number=save_number, save_suit=save_suit
        )

        if sample.num_known and sample.suit_known:
            print(f"[SAVE] (déjà connus) {sample.base_key} → rien à sauver")
        else:
            print(
                f"[SAVE] {sample.base_key} → number={number_label if save_number else '—'}; "
                f"suit={suit_label if save_suit else '—'}"
            )

        self.index += 1
        self._show_current()

    def _on_skip(self) -> None:
        if self.index >= len(self.samples):
            self.root.destroy()
            return
        self.index += 1
        self._show_current()

    def _show_current(self) -> None:
        if self.index >= len(self.samples):
            self.status_label.configure(text="Labellisation terminée")
            self.suggestion_label.configure(text=f"{len(self.writer.saved)} entrées ajoutées")
            self.image_label.configure(image=None, text="")
            self.save_btn.configure(text="Fermer", command=self.root.destroy)
            self.skip_btn.configure(text="Fermer", command=self.root.destroy)
            return
        sample = self.samples[self.index]
        self.status_label.configure(
            text=f"Carte {self.index + 1}/{len(self.samples)} — {sample.base_key} ({sample.source_path.name})"
        )
        self._update_menu(self.number_menu, self.number_choices, sample.number_suggestion, self.number_var)
        self._update_menu(self.suit_menu, self.suit_choices, sample.suit_suggestion, self.suit_var)
        suggestion_text = _format_suggestion(
            sample.number_suggestion, sample.suit_suggestion, sample.number_score, sample.suit_score
        )
        self.suggestion_label.configure(text=suggestion_text)

        # Affiche UNIQUEMENT ce qui est à renseigner
        if sample.num_known:
            self.lbl_number.grid_remove(); self.number_menu.grid_remove()
        else:
            self.lbl_number.grid(row=0, column=0, padx=8, pady=6)
            self.number_menu.grid(row=0, column=1, padx=8, pady=6)
        if sample.suit_known:
            self.lbl_suit.grid_remove(); self.suit_menu.grid_remove()
        else:
            self.lbl_suit.grid(row=1, column=0, padx=8, pady=6)
            self.suit_menu.grid(row=1, column=1, padx=8, pady=6)

        preview = _make_preview(
            sample.trimmed_number(self.writer.trim_border), sample.trimmed_suit(self.writer.trim_border)
        )
        self.photo = ImageTk.PhotoImage(preview)
        self.image_label.configure(image=self.photo)

    @staticmethod
    def _update_menu(menu: ctk.CTkOptionMenu, choices: Sequence[str], suggestion: Optional[str], var: tk.StringVar) -> None:
        values = list(choices) if choices else ["?"]
        menu.configure(values=values)
        if suggestion and suggestion in values:
            var.set(suggestion)
        else:
            var.set(values[0])


def _format_suggestion(num: Optional[str], suit: Optional[str], score_num: float, score_suit: float) -> str:
    if not num and not suit:
        return ""
    parts = []
    if num:
        parts.append(f"{num} ({score_num:.2f})")
    if suit:
        parts.append(f"{suit} ({score_suit:.2f})")
    return " / ".join(parts)


def _make_preview(num_img: Image.Image, suit_img: Image.Image) -> Image.Image:
    num_w, num_h = num_img.size
    suit_w, suit_h = suit_img.size
    width = max(num_w, suit_w)
    spacer = 6
    preview = Image.new("RGB", (width, num_h + suit_h + spacer), color="#f0f0f0")
    preview.paste(num_img, ((width - num_w) // 2, 0))
    preview.paste(suit_img, ((width - suit_w) // 2, num_h + spacer))
    return preview


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interface de labellisation des cartes")
    parser.add_argument("--game", default="PMU", help="Identifiant du jeu (dossier dans config/)")
    parser.add_argument("--crops-dir", help="Dossier contenant les captures à analyser")
    parser.add_argument("--threshold", type=float, default=0.92, help="Score min (0-1) pour accepter une reco auto")
    parser.add_argument("--strict", type=float, default=0.985, help="Score min (0-1) pour autoskip sans UI")
    parser.add_argument("--trim", type=int, default=6, help="Bordure rognée avant sauvegarde (px)")
    parser.add_argument("--force-all", action="store_true", help="Inclut toutes les cartes même reconnues")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    game_dir = Path("config") / args.game
    coords_path = game_dir / "coordinates.json"
    if not coords_path.exists():
        print(f"ERREUR: fichier de coordonnées introuvable ({coords_path})")
        return 2

    crops_dir = Path(args.crops_dir) if args.crops_dir else game_dir / "debug" / "crops"
    if not crops_dir.exists():
        print(f"ERREUR: dossier de captures introuvable ({crops_dir})")
        return 2

    regions, _, _ = load_coordinates(coords_path)
    cards_root = game_dir / "cards"
    idx = TemplateIndex(cards_root)
    idx.load()

    crop_paths = list(_iter_capture_files(crops_dir))
    if not crop_paths:
        print(f"Aucune capture trouvée dans {crops_dir}")
        return 0

    samples, total_cards, auto_ok = collect_card_samples(
        crop_paths,
        regions,
        idx,
        trim_border=int(args.trim),
        accept_threshold=float(args.threshold),
        strict_threshold=float(args.strict),
        force_all=bool(args.force_all),
    )

    print(
        f"Cartes analysées: {total_cards} — auto reconnues (strict): {auto_ok} — à labelliser: {len(samples)}"
    )
    if not samples:
        print("Rien à labelliser, base à jour ✅")
        return 0

    number_choices = _unique_sorted(
        [s.number_suggestion for s in samples if s.number_suggestion], DEFAULT_NUMBERS
    )
    suit_choices = _unique_sorted(
        [s.suit_suggestion for s in samples if s.suit_suggestion], DEFAULT_SUITS
    )

    writer = DatasetWriter(idx, cards_root, int(args.trim))
    app = LabelingApp(
        samples,
        writer,
        number_choices=number_choices,
        suit_choices=suit_choices,
        accept_threshold=float(args.threshold),
    )
    app.run()

    if writer.saved:
        print(f"{len(writer.saved)} entrées ajoutées dans {cards_root}")
    else:
        print("Aucune nouvelle entrée enregistrée.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Sequence, Dict
import itertools
import time

import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk
import tkinter as tk

# Dépend de votre module existant
# capture_cards: TemplateIndex, recognize_number_and_suit, collect_card_patches, is_card_present
from objet.scanner.cards_recognition import (
    TemplateIndex,
    recognize_number_and_suit,
    is_card_present,
)
from objet.utils.calibration import collect_card_patches

DEFAULT_NUMBERS: Sequence[str] = (
    "?",
    "A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2",
)
DEFAULT_SUITS: Sequence[str] = ("?", "spades", "hearts", "diamonds", "clubs")


def _trim(img: Image.Image, border: int) -> Image.Image:
    if border <= 0:
        return img
    w, h = img.size
    if w <= border * 2 or h <= border * 2:
        return img
    return img.crop((border, border, w - border, h - border))


@dataclass
class IdentifyResult:
    number: str
    suit: str
    meta: Dict[str, object]


class _SingleCardDialog:
    """Boîte de dialogue minimale pour compléter **uniquement** la partie inconnue.
    - Si l'une des deux (valeur/couleur) est déjà fiable, on ne montre que l'autre champ.
    - Retourne (number, suit) choisis par l'utilisateur.
    """

    def __init__(
        self,
        number_img: Image.Image,
        suit_img: Image.Image,
        *,
        missing_number: bool,
        missing_suit: bool,
        number_choices: Sequence[str],
        suit_choices: Sequence[str],
        suggested_number: Optional[str] = None,
        suggested_suit: Optional[str] = None,
    ) -> None:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Identifier la carte")
        self.root.geometry("560x480")

        self.result: Optional[Tuple[str, str]] = None

        self.number_var = tk.StringVar(value=(suggested_number or number_choices[0] if number_choices else "?"))
        self.suit_var = tk.StringVar(value=(suggested_suit or suit_choices[0] if suit_choices else "?"))

        top = ctk.CTkFrame(self.root)
        top.pack(fill="both", expand=True, padx=12, pady=12)

        # Aperçu empilé
        prev = self._make_preview(number_img, suit_img)
        self.photo = ImageTk.PhotoImage(prev)
        self.img_lbl = ctk.CTkLabel(top, image=self.photo, text="", compound="top")
        self.img_lbl.pack(pady=8)

        form = ctk.CTkFrame(top)
        form.pack(pady=8)

        row = 0
        if missing_number:
            ctk.CTkLabel(form, text="Valeur").grid(row=row, column=0, padx=8, pady=6)
            self.num_menu = ctk.CTkOptionMenu(form, values=list(number_choices) or ["?"], variable=self.number_var)
            self.num_menu.grid(row=row, column=1, padx=8, pady=6)
            row += 1
        if missing_suit:
            ctk.CTkLabel(form, text="Couleur").grid(row=row, column=0, padx=8, pady=6)
            self.suit_menu = ctk.CTkOptionMenu(form, values=list(suit_choices) or ["?"], variable=self.suit_var)
            self.suit_menu.grid(row=row, column=1, padx=8, pady=6)
            row += 1

        btns = ctk.CTkFrame(top)
        btns.pack(pady=10)
        ctk.CTkButton(btns, text="Valider", command=self._on_save).pack(side="left", padx=8)
        ctk.CTkButton(btns, text="Annuler", command=self._on_cancel).pack(side="left", padx=8)

    def run(self) -> Optional[Tuple[str, str]]:
        self.root.mainloop()
        return self.result

    def _on_save(self) -> None:
        self.result = (self.number_var.get().strip() or "?", self.suit_var.get().strip() or "?")
        self.root.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.root.destroy()

    @staticmethod
    def _make_preview(num_img: Image.Image, suit_img: Image.Image) -> Image.Image:
        num_w, num_h = num_img.size
        suit_w, suit_h = suit_img.size
        w = max(num_w, suit_w)
        spacer = 6
        canvas = Image.new("RGB", (w, num_h + suit_h + spacer), "#f0f0f0")
        canvas.paste(num_img, ((w - num_w) // 2, 0))
        canvas.paste(suit_img, ((w - suit_w) // 2, num_h + spacer))
        return canvas


class CardIdentifier:
    """Service réutilisable : identifier (valeur, couleur) à partir de 2 patches.

    - Utilise l'index `cards/` du jeu (TemplateIndex).
    - Tente la reco. Si >= strict → renvoie sans UI.
    - Sinon, si `interactive=True`, ouvre une mini-UI qui ne demande que la partie
      inconnue, **sauvegarde** cette partie et met à jour l'index.
    - Retourne toujours (number, suit, meta).
    """

    def __init__(
        self,
        game_dir: Path | str,
        *,
        trim: int = 6,
        threshold: float = 0.92,
        strict: float = 0.8,
        number_choices: Sequence[str] = DEFAULT_NUMBERS,
        suit_choices: Sequence[str] = DEFAULT_SUITS,
    ) -> None:
        self.game_dir = Path(game_dir)
        self.cards_root = self.game_dir / "cards"
        self.trim = int(trim)
        self.threshold = float(threshold)
        self.strict = float(strict)
        self.number_choices = list(number_choices)
        self.suit_choices = list(suit_choices)
        self.idx = TemplateIndex(self.cards_root)
        self.idx.load()
        self._counter = itertools.count(1)

    # ---------- API principale (patches) ----------
    def identify_from_patches(
        self,
        number_patch: Image.Image,
        suit_patch: Image.Image,
        *,
        base_key: str = "live",
        interactive: bool = True,
        force_all: bool = False,
    ) -> IdentifyResult:
        # 1) Trim puis tentative de reco
        tnum = _trim(number_patch, self.trim)
        tsuit = _trim(suit_patch, self.trim)
        num_s, suit_s, s_num, s_suit = recognize_number_and_suit(tnum, tsuit, self.idx)

        num_known_strict = bool(num_s) and float(s_num) >= self.strict
        suit_known_strict = bool(suit_s) and float(s_suit) >= self.strict

        if num_known_strict and suit_known_strict and not force_all:
            return IdentifyResult(num_s, suit_s, {
                "source": "auto",
                "score_number": float(s_num),
                "score_suit": float(s_suit),
            })

        # 2) Si interactif: ne demander que la partie inconnue
        if interactive:
            missing_number = not num_known_strict
            missing_suit = not suit_known_strict
            dialog = _SingleCardDialog(
                tnum,
                tsuit,
                missing_number=missing_number,
                missing_suit=missing_suit,
                number_choices=self.number_choices,
                suit_choices=self.suit_choices,
                suggested_number=num_s,
                suggested_suit=suit_s,
            )
            out = dialog.run()
            if out is None:
                # annulé → renvoyer meilleure info connue (ou "?")
                return IdentifyResult(num_s or "?", suit_s or "?", {
                    "source": "cancel",
                    "score_number": float(s_num),
                    "score_suit": float(s_suit),
                })
            lab_num, lab_suit = out
            save_number = missing_number and lab_num not in {"", "?"}
            save_suit = missing_suit and lab_suit not in {"", "?"}
            self._save_if_missing(tnum, tsuit, lab_num, lab_suit, save_number, save_suit, base_key)
            # MAJ index pour la session courante
            if save_number:
                self._update_index(lab_num, tnum, is_number=True)
            if save_suit:
                self._update_index(lab_suit, tsuit, is_number=False)
            return IdentifyResult(lab_num or (num_s or "?"), lab_suit or (suit_s or "?"), {
                "source": "labeled",
                "score_number": float(s_num),
                "score_suit": float(s_suit),
            })

        # 3) Non interactif → renvoyer la meilleure hypothèse (ou "?")
        return IdentifyResult(num_s or "?", suit_s or "?", {
            "source": "guess",
            "score_number": float(s_num),
            "score_suit": float(s_suit),
        })

    # ---------- API pratique (image table + base_key) ----------
    def identify_from_table(
        self,
        table_img: Image.Image,
        regions: Dict[str, object],
        base_key: str,
        *,
        interactive: bool = True,
        force_all: bool = False,
    ) -> IdentifyResult:
        pairs = collect_card_patches(table_img.convert("RGB"), regions, pad=0)
        if base_key not in pairs:
            return IdentifyResult("?", "?", {"source": "error", "reason": "region-missing"})
        num_patch, suit_patch = pairs[base_key]
        if not is_card_present(num_patch, threshold=215, min_ratio=0.04):
            return IdentifyResult("?", "?", {"source": "empty", "reason": "no-card"})
        return self.identify_from_patches(
            num_patch,
            suit_patch,
            base_key=base_key,
            interactive=interactive,
            force_all=force_all,
        )

    # ---------- Sauvegarde minimale des nouveaux exemples ----------
    def _save_if_missing(
        self,
        num_img: Image.Image,
        suit_img: Image.Image,
        number_label: str,
        suit_label: str,
        save_number: bool,
        save_suit: bool,
        base_key: str,
    ) -> None:
        ts = int(time.time())
        idx = next(self._counter)
        base = f"{base_key}_{ts}_{idx:04d}"
        if save_number:
            p = self.cards_root / "numbers" / number_label / f"{base}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            num_img.save(p)
        if save_suit:
            p = self.cards_root / "suits" / suit_label / f"{base}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            suit_img.save(p)

    def _update_index(self, label: str, img: Image.Image, *, is_number: bool) -> None:
        gray = np.array(img.convert("L"))
        arr = TemplateIndex._prep(gray)
        store = self.idx.numbers if is_number else self.idx.suits
        store.setdefault(label, []).append(arr)



```
### scripts/position_zones.py
```python
# position_zones_ctk.py — Zone Editor multi‑jeux (CustomTkinter)
# Auteur: ChatGPT (portage puis refonte "templates + top_left")
# ---------------------------------------------------------------
# Nouveautés vs version précédente :
#  - Schéma JSON simplifié `templates` (tailles par groupe) + `regions` (top_left + group).
#  - Pré‑chargement des rectangles aux positions des `regions`.
#  - Clamp auto si dépassement image.
#  - Édition : sélection, renommage, changement de groupe, X/Y.
#  - Changement de TAILLE au niveau GROUPE → propage à toutes les zones du groupe.
#  - ✅ Déplacement "à la mano" des rectangles : cliquer‑glisser pour bouger.
#  - ✅ Zoom : slider (25%–300%), Ctrl+molette, boutons Ajuster/100%.
#  - Robustesse : garde‑fous contre groupes manquants et conversions.
#  - Sauvegarde au format `templates + regions`.
#
# Dépendances: customtkinter, pillow
#   pip install customtkinter pillow
# ---------------------------------------------------------------

import os, json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Dict, Any, Optional, Tuple

import customtkinter as ctk
from PIL import Image, ImageTk

from objet.services.game import Game
from _utils import clamp_top_left, coerce_int, resolve_templates

APP_TITLE = "Zone Editor (CustomTkinter) — Multi‑jeux"
MAX_CANVAS_W = 1280
MAX_CANVAS_H = 800

# ---------------------------------------------------------------
# Chargeurs JSON
# ---------------------------------------------------------------

def _load_templated(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if isinstance(data, dict) and "templates" in data and "regions" in data:
        return data
    return None


# ---------------------------------------------------------------
# Application principale
# ---------------------------------------------------------------

class ZoneEditorCTK:
    def __init__(self, base_dir: Optional[str] = None):
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title(APP_TITLE)
        self.root.geometry("1700x1000")

        # État
        self.game = Game.for_script(Path(__file__).name)
        self.base_dir = self._default_base_dir(base_dir)
        self.current_game: Optional[str] = None
        self.img_path: Optional[str] = None
        self.img_pil_original: Optional[Image.Image] = None
        self.img_display: Optional[Image.Image] = None
        self.tk_img: Optional[ImageTk.PhotoImage] = None
        self.base_scale: float = 1.0  # échelle "fit"
        self.user_zoom: float = 1.0   # multiplicateur utilisateur
        self.scale: float = 1.0       # base_scale * user_zoom

        # Schéma courant
        self.table_capture: Dict[str, Any] = self.game.table.captures.table_capture
        self.templates: Dict[str, Dict[str, Any]] = self.game.table.captures.templates
        self.templates_resolved: Dict[str, Dict[str, Any]] = {}
        # regions: key -> {group, top_left:[x,y], value, label}
        self.regions: Dict[str, Dict[str, Any]] = self.game.table.captures.regions

        # Dessin
        self.rect_items: Dict[str, int] = {}   # key -> canvas rect id
        self.text_items: Dict[str, int] = {}   # key -> canvas text id
        self.dragging_key: Optional[str] = None
        self.drag_offset: Tuple[int,int] = (0, 0)

        self._build_ui()
        self._bind_canvas_events()
        self._refresh_games_list()
        self._update_info()

    # ---------------- UI ----------------
    def _build_ui(self):
        # Topbar
        self.topbar = ctk.CTkFrame(self.root, corner_radius=0)
        self.topbar.pack(side="top", fill="x")

        # Base dir chooser
        self.lbl_root = ctk.CTkLabel(self.topbar, text="Racine:")
        self.lbl_root.pack(side="left", padx=(8,4), pady=8)

        self.root_entry = ctk.CTkEntry(self.topbar, width=420)
        self.root_entry.insert(0, self.base_dir)
        self.root_entry.pack(side="left", padx=4, pady=8)

        self.btn_browse_root = ctk.CTkButton(self.topbar, text="Parcourir…", command=self._choose_base_dir)
        self.btn_browse_root.pack(side="left", padx=6, pady=8)

        # Games dropdown
        self.lbl_game = ctk.CTkLabel(self.topbar, text="Jeu:")
        self.lbl_game.pack(side="left", padx=(16,4), pady=8)

        self.game_var = ctk.StringVar(value="(aucun)")
        self.game_menu = ctk.CTkOptionMenu(self.topbar, values=["(aucun)"], variable=self.game_var, command=self._on_select_game)
        self.game_menu.pack(side="left", padx=4, pady=8)

        self.btn_refresh = ctk.CTkButton(self.topbar, text="Rafraîchir", command=self._refresh_games_list)
        self.btn_refresh.pack(side="left", padx=(4,10), pady=8)

        # Zoom controls
        ctk.CTkLabel(self.topbar, text="Zoom").pack(side="left", padx=(16,4))
        self.zoom_var = tk.DoubleVar(value=1.0)
        self.zoom_slider = ctk.CTkSlider(self.topbar, from_=0.25, to=3.0, number_of_steps=55, command=lambda v: self._on_zoom_slider(float(v)))
        self.zoom_slider.set(1.0)
        self.zoom_slider.pack(side="left", padx=6, pady=8)
        self.zoom_pct_label = ctk.CTkLabel(self.topbar, text="100%")
        self.zoom_pct_label.pack(side="left", padx=(6,2))
        self.btn_fit = ctk.CTkButton(self.topbar, text="Ajuster", width=80, command=self._zoom_fit)
        self.btn_fit.pack(side="left", padx=4)
        self.btn_100 = ctk.CTkButton(self.topbar, text="100%", width=70, command=self._zoom_100)
        self.btn_100.pack(side="left", padx=4)

        # Actions globales
        self.btn_save = ctk.CTkButton(self.topbar, text="Enregistrer", command=self.save_json, state="disabled")
        self.btn_save.pack(side="left", padx=8, pady=8)

        self.info_label = ctk.CTkLabel(self.topbar, text="Aucune image", anchor="w")
        self.info_label.pack(side="left", padx=16, pady=8)

        # Main area split
        self.main = ctk.CTkFrame(self.root)
        self.main.pack(side="top", fill="both", expand=True)

        # Canvas
        self.canvas_frame = ctk.CTkFrame(self.main)
        self.canvas_frame.pack(side="left", fill="both", expand=True, padx=(10,5), pady=10)
        self.canvas = tk.Canvas(self.canvas_frame, bg="#F2F2F2", highlightthickness=0, width=MAX_CANVAS_W, height=MAX_CANVAS_H)
        self.canvas.pack(fill="both", expand=True)

        # Sidebar (édition)
        self.sidebar = ctk.CTkFrame(self.main, width=380)
        self.sidebar.pack(side="right", fill="y", padx=(5,10), pady=10)

        # Liste des régions
        ctk.CTkLabel(self.sidebar, text="Régions", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=12, pady=(12,6))
        self.listbox = tk.Listbox(self.sidebar, height=12)
        self.listbox.pack(fill="x", padx=12)
        self.listbox.bind("<<ListboxSelect>>", self._on_select_region_from_list)

        # Champs d'édition
        frm = ctk.CTkFrame(self.sidebar)
        frm.pack(fill="x", padx=12, pady=12)

        # Nom (clé)
        ctk.CTkLabel(frm, text="Nom (clé)").grid(row=0, column=0, sticky="w")
        self.entry_name = ctk.CTkEntry(frm)
        self.entry_name.grid(row=0, column=1, sticky="ew", padx=(8,0))
        frm.grid_columnconfigure(1, weight=1)

        # Groupe
        ctk.CTkLabel(frm, text="Groupe").grid(row=1, column=0, sticky="w", pady=(6,0))
        self.group_var = ctk.StringVar(value="")
        self.group_menu = ctk.CTkOptionMenu(frm, values=[""], variable=self.group_var)
        self.group_menu.grid(row=1, column=1, sticky="ew", padx=(8,0), pady=(6,0))

        # Position X/Y
        ctk.CTkLabel(frm, text="X").grid(row=2, column=0, sticky="w", pady=(6,0))
        self.entry_x = ctk.CTkEntry(frm, width=90)
        self.entry_x.grid(row=2, column=1, sticky="w", padx=(8,0), pady=(6,0))

        ctk.CTkLabel(frm, text="Y").grid(row=3, column=0, sticky="w")
        self.entry_y = ctk.CTkEntry(frm, width=90)
        self.entry_y.grid(row=3, column=1, sticky="w", padx=(8,0))

        # Taille du GROUPE (propagé)
        ctk.CTkLabel(frm, text="Largeur (groupe)").grid(row=4, column=0, sticky="w", pady=(10,0))
        self.entry_w = ctk.CTkEntry(frm, width=90)
        self.entry_w.grid(row=4, column=1, sticky="w", padx=(8,0), pady=(10,0))

        ctk.CTkLabel(frm, text="Hauteur (groupe)").grid(row=5, column=0, sticky="w")
        self.entry_h = ctk.CTkEntry(frm, width=90)
        self.entry_h.grid(row=5, column=1, sticky="w", padx=(8,0))

        # Boutons édition
        btns = ctk.CTkFrame(self.sidebar)
        btns.pack(fill="x", padx=12, pady=(0,12))
        self.btn_apply = ctk.CTkButton(btns, text="Appliquer", command=self.apply_changes, state="disabled")
        self.btn_apply.pack(side="left", padx=6)
        self.btn_add = ctk.CTkButton(btns, text="Ajouter", command=self.add_region, state="disabled")
        self.btn_add.pack(side="left", padx=6)
        self.btn_delete = ctk.CTkButton(btns, text="Supprimer", command=self.delete_region, state="disabled")
        self.btn_delete.pack(side="left", padx=6)

        # Status
        self.status = ctk.CTkLabel(self.root, text="Prêt", anchor="w")
        self.status.pack(side="bottom", fill="x", padx=8, pady=6)

    def _bind_canvas_events(self):
        # Sélection + drag & drop
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        # Zoom à la molette avec Ctrl
        self.root.bind("<Control-MouseWheel>", self._on_ctrl_wheel)
        # Raccourcis zoom
        self.root.bind("<Control-Key-0>", lambda e: self._zoom_fit())
        self.root.bind("<Control-Key-1>", lambda e: self._zoom_100())

    # ---------------- Base dir & jeux ----------------
    def _default_base_dir(self, base_dir: Optional[str]) -> str:
        if base_dir:
            return os.path.abspath(base_dir)
        here = os.path.dirname(os.path.abspath(__file__))
        guess = os.path.abspath(os.path.join(here, "..", "config"))
        return guess

    def _choose_base_dir(self):
        path = filedialog.askdirectory(title="Choisir le dossier racine (config)", initialdir=self.base_dir)
        if not path:
            return
        self.base_dir = path
        self.root_entry.delete(0, tk.END)
        self.root_entry.insert(0, self.base_dir)
        self._refresh_games_list()
        self.status.configure(text=f"Racine: {self.base_dir}")

    def _refresh_games_list(self):
        base = os.path.abspath(self.root_entry.get().strip() or self.base_dir)
        games: List[str] = []
        if os.path.isdir(base):
            try:
                for name in sorted(os.listdir(base)):
                    full = os.path.join(base, name)
                    if os.path.isdir(full) and os.path.isfile(os.path.join(full, "test_crop_result.png")):
                        games.append(name)
            except Exception:
                pass
        if not games:
            games = ["(aucun)"]
        self.game_menu.configure(values=games)
        if self.current_game in games:
            self.game_var.set(self.current_game)
        else:
            self.game_var.set(games[0])
            if games[0] != "(aucun)":
                self._on_select_game(games[0])

    def _on_select_game(self, game_name: str):
        if not game_name or game_name == "(aucun)":
            return
        self.current_game = game_name
        base = os.path.abspath(self.root_entry.get().strip() or self.base_dir)
        folder = os.path.join(base, game_name)
        img_path = os.path.join(folder, "test_crop_result.png")
        coord_path = os.path.join(folder, "coordinates.json")

        if not os.path.isfile(img_path):
            messagebox.showerror("Erreur", f"Image introuvable:\n{img_path}")
            return
        try:
            img = Image.open(img_path).convert("RGBA")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'ouvrir l'image:\n{e}")
            return

        self.img_path = img_path
        self.img_pil_original = img
        self._prepare_display_image(img)

        # Par défaut pour table_capture
        self.table_capture.clear()
        self.table_capture.update({"enabled": True, "relative_bounds": [0, 0, img.width, img.height]})

        # Charger JSON
        self.templates.clear()
        self.templates_resolved = {}
        self.regions.clear()

        templated = _load_templated(coord_path) if os.path.isfile(coord_path) else None
        if templated:
            self.table_capture.update(templated.get("table_capture", {}))
            self.templates.update(templated.get("templates", {}))
            self.templates_resolved = resolve_templates(self.templates)
            reg = templated.get("regions", {})
            for key, r in reg.items():
                group = r.get("group", "")
                tl = r.get("top_left", [0, 0])
                val = r.get("value", None)
                label = r.get("label", key)
                self.regions[key] = {"group": group, "top_left": [int(tl[0]), int(tl[1])], "value": val, "label": label}
            self.status.configure(text=f"{game_name}: image + {len(self.regions)} région(s) chargées")
        else:
            # pas de JSON → base minimale
            self.templates.clear()
            self.templates.update({"action_button": {"size": [165, 70], "type": "texte"}})
            self.templates_resolved = resolve_templates(self.templates)
            self.status.configure(text=f"{game_name}: image chargée (coordinates.json absent)")

        self._enable_ui_after_load()
        self._reset_canvas_for_image()
        self._redraw_all()
        self._populate_regions_list()
        self._populate_group_menu()
        self._update_info()
        self._sync_game_capture()

    # ---------------- Image/Canvas ----------------
    def _prepare_display_image(self, img: Image.Image):
        w, h = img.size
        self.base_scale = min(MAX_CANVAS_W / w, MAX_CANVAS_H / h, 1.0)
        if self.base_scale <= 0:
            self.base_scale = 1.0
        self._update_display_image()

    def _update_display_image(self):
        if not self.img_pil_original:
            return
        self.scale = max(0.05, self.base_scale * max(0.1, float(self.user_zoom)))
        disp_w = int(self.img_pil_original.width * self.scale)
        disp_h = int(self.img_pil_original.height * self.scale)
        self.img_display = self.img_pil_original.resize((disp_w, disp_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(self.img_display)
        pct = int(round(self.scale / self.base_scale * 100)) if self.base_scale else int(self.scale*100)
        self.zoom_pct_label.configure(text=f"{pct}%") if hasattr(self, 'zoom_pct_label') else None
        self._reset_canvas_for_image()
        self._update_info()

    def _reset_canvas_for_image(self):
        self.canvas.delete("all")
        self.rect_items.clear()
        self.text_items.clear()
        self.canvas.config(width=max(self.tk_img.width(), MAX_CANVAS_W), height=max(self.tk_img.height(), MAX_CANVAS_H))
        # Place l'image à l'origine (0,0) pour aligner les rectangles sans offset
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    def _redraw_all(self):
        for rid in list(self.rect_items.values()):
            self.canvas.delete(rid)
        for tid in list(self.text_items.values()):
            self.canvas.delete(tid)
        self.rect_items.clear(); self.text_items.clear()

        if not self.img_pil_original:
            return
        W, H = self.img_pil_original.size
        s = self.scale if self.scale else 1.0

        for key, r in self.regions.items():
            group = r.get("group", "")
            label = r.get("label", key)
            tl = r.get("top_left", [0, 0])
            size = self.templates_resolved.get(group, {}).get("size", [60, 40])
            w, h = coerce_int(size[0], 60), coerce_int(size[1], 40)
            x, y = clamp_top_left(coerce_int(tl[0]), coerce_int(tl[1]), w, h, W, H)
            r["top_left"] = [x, y]
            dx0, dy0 = int(x * s), int(y * s)
            dx1, dy1 = int((x + w) * s), int((y + h) * s)
            rid = self.canvas.create_rectangle(dx0, dy0, dx1, dy1, outline="#0ea5e9", width=2)
            tid = self.canvas.create_text(dx0 + 6, dy0 + 6, anchor="nw", text=str(label), fill="#0ea5e9", font=("Segoe UI", 10, "bold"))
            self.rect_items[key] = rid
            self.text_items[key] = tid

    def _update_info(self):
        if not self.img_pil_original:
            self.info_label.configure(text="Aucune image")
            return
        name = os.path.basename(self.img_path or "?")
        w, h = self.img_pil_original.size
        pct = int(round(self.scale / self.base_scale * 100)) if self.base_scale else int(self.scale*100)
        self.info_label.configure(text=f"{name}  |  {w}×{h}  |  zoom {pct}%")

    def _sync_game_capture(self) -> None:
        self.game.update_from_capture(
            table_capture=self.table_capture,
            regions=self.regions,
            templates=self.templates,
            reference_path=self.img_path,
        )

    # ---------------- Liste & sélection ----------------
    def _populate_regions_list(self):
        self.listbox.delete(0, tk.END)
        for k in sorted(self.regions.keys()):
            lab = self.regions[k].get("label", k)
            self.listbox.insert(tk.END, f"{k}  —  {lab}")

    def _populate_group_menu(self):
        # Reset simple (utilisé au chargement)
        groups = sorted(list(self.templates_resolved.keys())) or [""]
        self.group_menu.configure(values=groups)
        if groups:
            self.group_var.set(groups[0])

    def _populate_group_menu_keep_current(self):
        # Version qui conserve la sélection actuelle
        current = self.group_var.get()
        groups = sorted(list(self.templates_resolved.keys())) or [""]
        self.group_menu.configure(values=groups)
        if current in groups:
            self.group_var.set(current)
        elif groups:
            self.group_var.set(groups[0])

    def _current_selection_key(self) -> Optional[str]:
        sel = self.listbox.curselection()
        if not sel:
            return None
        line = self.listbox.get(sel[0])
        key = line.split("  —  ", 1)[0]
        return key

    def _select_key_in_listbox(self, key: str):
        keys_sorted = sorted(self.regions.keys())
        if key in keys_sorted:
            idx = keys_sorted.index(key)
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(idx)
            self.listbox.activate(idx)

    def _on_select_region_from_list(self, event=None):
        key = self._current_selection_key()
        if not key:
            return
        r = self.regions.get(key, {})
        self.entry_name.delete(0, tk.END); self.entry_name.insert(0, key)
        # fixe le menu de groupe sur la valeur de la région (sans reset global)
        group = r.get("group", "")
        if group not in self.templates_resolved:
            # créer un groupe par défaut si besoin
            self.templates[group] = {"size": [60, 40], "type": "mix"}
            self.templates_resolved = resolve_templates(self.templates)
            self._populate_group_menu_keep_current()
        self.group_var.set(group)
        tl = r.get("top_left", [0, 0])
        self.entry_x.delete(0, tk.END); self.entry_x.insert(0, str(int(tl[0])))
        self.entry_y.delete(0, tk.END); self.entry_y.insert(0, str(int(tl[1])))
        gsize = self.templates_resolved.get(group, {}).get("size", [60, 40])
        self.entry_w.delete(0, tk.END); self.entry_w.insert(0, str(int(gsize[0])))
        self.entry_h.delete(0, tk.END); self.entry_h.insert(0, str(int(gsize[1])))
        self.btn_apply.configure(state="normal")
        self.btn_delete.configure(state="normal")

    # ---------------- Détection & Drag ----------------
    def _region_at_point(self, x: int, y: int) -> Optional[str]:
        for key, r in self.regions.items():
            g = r.get("group", "")
            w, h = self.templates_resolved.get(g, {}).get("size", [0, 0])
            tlx, tly = r.get("top_left", [0, 0])
            if tlx <= x <= tlx + w and tly <= y <= tly + h:
                return key
        return None

    def _on_mouse_down(self, event):
        if not self.img_pil_original:
            return
        s = self.scale if self.scale else 1.0
        x, y = int(event.x / s), int(event.y / s)
        key = self._region_at_point(x, y)
        if key:
            self.dragging_key = key
            tlx, tly = self.regions[key].get("top_left", [0, 0])
            self.drag_offset = (x - tlx, y - tly)
            self._select_key_in_listbox(key)
            self._on_select_region_from_list()
        else:
            self.dragging_key = None

    def _on_mouse_drag(self, event):
        if not self.dragging_key or not self.img_pil_original:
            return
        key = self.dragging_key
        r = self.regions.get(key, {})
        g = r.get("group", "")
        w, h = self.templates_resolved.get(g, {}).get("size", [0, 0])
        if w <= 0 or h <= 0:
            return
        s = self.scale if self.scale else 1.0
        W, H = self.img_pil_original.size
        x = int(event.x / s) - self.drag_offset[0]
        y = int(event.y / s) - self.drag_offset[1]
        x, y = clamp_top_left(x, y, w, h, W, H)
        r["top_left"] = [x, y]
        dx0, dy0 = int(x * s), int(y * s)
        dx1, dy1 = int((x + w) * s), int((y + h) * s)
        rid = self.rect_items.get(key)
        tid = self.text_items.get(key)
        if rid: self.canvas.coords(rid, dx0, dy0, dx1, dy1)
        if tid: self.canvas.coords(tid, dx0 + 6, dy0 + 6)
        self.entry_x.delete(0, tk.END); self.entry_x.insert(0, str(x))
        self.entry_y.delete(0, tk.END); self.entry_y.insert(0, str(y))

    def _on_mouse_up(self, event):
        self.dragging_key = None

    # ---------------- Actions d'édition ----------------
    def apply_changes(self):
        key = self._current_selection_key()
        if not key:
            return
        r = self.regions.get(key, {})
        if not r:
            return
        old_key = key
        old_group = r.get("group", "")

        # Nouveau nom (clé)
        new_key = (self.entry_name.get() or old_key).strip()
        if new_key and new_key != old_key and new_key not in self.regions:
            self.regions[new_key] = r
            del self.regions[old_key]
            if old_key in self.rect_items: self.rect_items[new_key] = self.rect_items.pop(old_key)
            if old_key in self.text_items: self.text_items[new_key] = self.text_items.pop(old_key)
            key = new_key
            r = self.regions[key]
        r["label"] = key

        # Groupe
        g = (self.group_var.get() or old_group).strip()
        if g and g not in self.templates:
            # crée le groupe s'il n'existe pas
            self.templates[g] = {"size": [60, 40], "type": "mix"}
        r["group"] = g
        self.templates_resolved = resolve_templates(self.templates)

        # Position
        try:
            x = int(self.entry_x.get()); y = int(self.entry_y.get())
        except Exception:
            x, y = r.get("top_left", [0, 0])
        W, H = self.img_pil_original.size if self.img_pil_original else (0, 0)
        gw, gh = self.templates_resolved.get(r["group"], {}).get("size", [60, 40])
        x, y = clamp_top_left(x, y, gw, gh, W, H)
        r["top_left"] = [x, y]

        # Taille de GROUPE (propagation)
        try:
            nw = int(self.entry_w.get()); nh = int(self.entry_h.get())
            if nw > 0 and nh > 0:
                self.templates[r["group"]] = {**self.templates.get(r["group"], {}), "size": [nw, nh]}
                self.templates_resolved = resolve_templates(self.templates)
        except Exception:
            pass

        # Redraw et re‑sélection — conserve la sélection de groupe
        self._populate_group_menu_keep_current()
        self._populate_regions_list()
        self._update_display_image()  # garde l'échelle actuelle
        self._redraw_all()
        self._select_key_in_listbox(key)
        self._on_select_region_from_list()
        self.status.configure(text=f"Modifié: {key}")

    def add_region(self):
        if not self.img_pil_original:
            return
        g = self.group_var.get()
        if not g or g not in self.templates_resolved:
            messagebox.showinfo("Info", "Choisis un groupe valide avant d'ajouter.")
            return
        W, H = self.img_pil_original.size
        gw, gh = self.templates_resolved[g]["size"]
        x, y = clamp_top_left(W // 2 - gw // 2, H // 2 - gh // 2, gw, gh, W, H)
        base = f"{g}_"; idx = 1
        while f"{base}{idx}" in self.regions:
            idx += 1
        key = f"{g}_{idx}"
        self.regions[key] = {"group": g, "top_left": [x, y], "value": None, "label": key}
        self._populate_regions_list()
        self._redraw_all()
        self._select_key_in_listbox(key)
        self._on_select_region_from_list()
        self.status.configure(text=f"Ajouté: {key}")

    def delete_region(self):
        key = self._current_selection_key()
        if not key:
            return
        if messagebox.askyesno("Confirmation", f"Supprimer la région '{key}' ?"):
            self.regions.pop(key, None)
            rid = self.rect_items.pop(key, None)
            tid = self.text_items.pop(key, None)
            if rid: self.canvas.delete(rid)
            if tid: self.canvas.delete(tid)
            self._populate_regions_list()
            self.entry_name.delete(0, tk.END)
            self.entry_x.delete(0, tk.END); self.entry_y.delete(0, tk.END)
            self.entry_w.delete(0, tk.END); self.entry_h.delete(0, tk.END)
            self.btn_delete.configure(state="disabled")
            self.status.configure(text=f"Supprimé: {key}")

    def _enable_ui_after_load(self):
        self.btn_save.configure(state="normal")
        self.btn_apply.configure(state="normal")
        self.btn_add.configure(state="normal")
        self.btn_delete.configure(state="disabled")

    # ---------------- Sauvegarde ----------------
    def save_json(self):
        if not self.img_pil_original:
            messagebox.showinfo("Info", "Charge une image avant de sauvegarder.")
            return
        base = os.path.abspath(self.root_entry.get().strip() or self.base_dir)
        folder = os.path.join(base, self.current_game or "")
        default_name = "coordinates.json"
        init_dir = folder if os.path.isdir(folder) else os.path.dirname(self.img_path or "")
        out_path = filedialog.asksaveasfilename(
            title="Enregistrer",
            defaultextension=".json",
            initialdir=init_dir,
            initialfile=default_name,
            filetypes=[("JSON", "*.json")]
        )
        if not out_path:
            return

        W, H = self.img_pil_original.size
        tc = self.table_capture or {"enabled": True, "relative_bounds": [0, 0, W, H]}

        payload = {
            "table_capture": tc,
            "templates": self.templates,
            "regions": {}
        }
        for key, r in self.regions.items():
            payload["regions"][key] = {
                "group": r.get("group", ""),
                "top_left": [int(r.get("top_left", [0, 0])[0]), int(r.get("top_left", [0, 0])[1])],
                "value": r.get("value", None),
                "label": r.get("label", key)
            }
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            self.status.configure(text=f"Sauvegardé: {out_path}")
            messagebox.showinfo("Succès", f"Enregistré :\n{out_path}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'écrire le fichier:\n{e}")

    # ---------------- Zoom helpers ----------------
    def _on_zoom_slider(self, value: float):
        self.user_zoom = float(value)
        self._update_display_image()
        self._redraw_all()

    def _zoom_fit(self):
        self.user_zoom = 1.0
        self.zoom_slider.set(1.0)
        self._update_display_image()
        self._redraw_all()

    def _zoom_100(self):
        if self.base_scale == 0:
            return
        self.user_zoom = 1.0 / self.base_scale
        # contraindre au range du slider
        self.user_zoom = min(max(self.user_zoom, 0.25), 3.0)
        self.zoom_slider.set(self.user_zoom)
        self._update_display_image()
        self._redraw_all()

    def _on_ctrl_wheel(self, event):
        # delta>0 zoom in ; delta<0 zoom out
        step = 1.1 if event.delta > 0 else 1/1.1
        self.user_zoom = min(max(self.user_zoom * step, 0.25), 3.0)
        self.zoom_slider.set(self.user_zoom)
        self._update_display_image()
        self._redraw_all()

    # ---------------- Run ----------------
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = ZoneEditorCTK()
    app.run()

```
### scripts/position_zones_ctk.py
```python
# position_zones_ctk.py — UI simplifiée (CustomTkinter) avec alignement au démarrage
# UI uniquement — Logique dans zone_project.py
# Changements clés :
#  - UI épurée (Pas de Rafraîchir/Ajouter/Supprimer, pas de nom d'image en haut)
#  - Zoom (slider, Ctrl+molette, Ajuster/100%)
#  - Sélection robuste (exportselection=False, _last_key)
#  - "Appliquer" lit d'abord le champ Nom (clé), puis fallback sur last_key/1ère clé + logs
#  - Drag & drop + MAJ de tout le groupe si lock_same_y
#  - **Alignement au démarrage** pour les groupes avec templates.layout.lock_same_y (ancre = 1er élément trié)

from __future__ import annotations
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional, Tuple, List

import customtkinter as ctk
from PIL import ImageTk, Image

# Allow running this script directly (ensure project root on sys.path)
import sys
try:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
except Exception:
    pass

# Disable save-as dialog for direct save to suggested path
def _direct_save_as_filename(**kwargs):
    try:
        initialdir = kwargs.get("initialdir") or os.getcwd()
        initialfile = kwargs.get("initialfile") or "coordinates.json"
        return os.path.join(initialdir, initialfile)
    except Exception:
        # Fallback: default file in CWD
        return os.path.join(os.getcwd(), "coordinates.json")

filedialog.asksaveasfilename = _direct_save_as_filename

from objet.services.game import Game
from zone_project import ZoneProject

APP_TITLE = "Zone Editor (CustomTkinter) — Multi-jeux (UI simplifiée)"
MAX_CANVAS_W = 1280
MAX_CANVAS_H = 800

class ZoneEditorCTK:
    def __init__(self, base_dir: Optional[str] = None):
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title(APP_TITLE)
        self.root.geometry("1600x940")

        # Modèle
        self.project = ZoneProject(Game.for_script(Path(__file__).name))
        self.base_dir = self._default_base_dir(base_dir)

        # Image affichée
        self.tk_img: Optional[ImageTk.PhotoImage] = None
        self.base_scale: float = 1.0
        self.user_zoom: float = 1.0
        self.scale: float = 1.0

        # Dessin
        self.rect_items: dict[str, int] = {}
        self.text_items: dict[str, int] = {}
        self.dragging_key: Optional[str] = None
        self.drag_offset: Tuple[int,int] = (0, 0)
        self._last_key: Optional[str] = None

        self._build_ui()
        self._bind_canvas_events()
        self._refresh_games_list()

    # ---------- UI ----------
    def _build_ui(self):
        # Topbar
        top = ctk.CTkFrame(self.root, corner_radius=0)
        top.pack(side="top", fill="x")

        # Racine
        ctk.CTkLabel(top, text="Racine:").pack(side="left", padx=(8,4), pady=8)
        self.root_entry = ctk.CTkEntry(top, width=420)
        self.root_entry.insert(0, self.base_dir)
        self.root_entry.pack(side="left", padx=4, pady=8)
        ctk.CTkButton(top, text="Parcourir…", command=self._choose_base_dir).pack(side="left", padx=6, pady=8)

        # Jeux
        ctk.CTkLabel(top, text="Jeu:").pack(side="left", padx=(16,4), pady=8)
        self.game_var = tk.StringVar(value="(aucun)")
        self.game_menu = ctk.CTkOptionMenu(top, values=["(aucun)"], variable=self.game_var, command=self._on_select_game)
        self.game_menu.pack(side="left", padx=4, pady=8)

        # Zoom
        ctk.CTkLabel(top, text="Zoom").pack(side="left", padx=(16,4))
        self.zoom_slider = ctk.CTkSlider(top, from_=0.25, to=3.0, number_of_steps=55, command=lambda v: self._on_zoom_slider(float(v)))
        self.zoom_slider.set(1.0)
        self.zoom_slider.pack(side="left", padx=6, pady=8)
        self.zoom_pct_label = ctk.CTkLabel(top, text="100%")
        self.zoom_pct_label.pack(side="left", padx=(6,2))
        ctk.CTkButton(top, text="Ajuster", width=80, command=self._zoom_fit).pack(side="left", padx=4)
        ctk.CTkButton(top, text="100%", width=70, command=self._zoom_100).pack(side="left", padx=4)

        # Enregistrer
        self.btn_save = ctk.CTkButton(top, text="Enregistrer", command=self._save_json, state="disabled")
        self.btn_save.pack(side="left", padx=8, pady=8)

        # Main split
        main = ctk.CTkFrame(self.root); main.pack(side="top", fill="both", expand=True)

        # Canvas
        cf = ctk.CTkFrame(main); cf.pack(side="left", fill="both", expand=True, padx=(10,5), pady=10)
        self.canvas = tk.Canvas(cf, bg="#F2F2F2", highlightthickness=0, width=MAX_CANVAS_W, height=MAX_CANVAS_H)
        self.canvas.pack(fill="both", expand=True)

        # Sidebar
        side = ctk.CTkFrame(main, width=360); side.pack(side="right", fill="y", padx=(5,10), pady=10)
        ctk.CTkLabel(side, text="Régions", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=12, pady=(12,6))
        self.listbox = tk.Listbox(side, height=14, exportselection=False)  # conserve sélection
        self.listbox.pack(fill="x", padx=12)
        self.listbox.bind("<<ListboxSelect>>", self._on_select_region_from_list)

        frm = ctk.CTkFrame(side); frm.pack(fill="x", padx=12, pady=12)
        ctk.CTkLabel(frm, text="Nom (clé)").grid(row=0, column=0, sticky="w")
        self.entry_name = ctk.CTkEntry(frm); self.entry_name.grid(row=0, column=1, sticky="ew", padx=(8,0)); frm.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(frm, text="Groupe").grid(row=1, column=0, sticky="w", pady=(6,0))
        self.group_var = tk.StringVar(value="")
        self.group_menu = ctk.CTkOptionMenu(frm, values=[""], variable=self.group_var)
        self.group_menu.grid(row=1, column=1, sticky="ew", padx=(8,0), pady=(6,0))

        ctk.CTkLabel(frm, text="X").grid(row=2, column=0, sticky="w", pady=(6,0))
        self.entry_x = ctk.CTkEntry(frm, width=90); self.entry_x.grid(row=2, column=1, sticky="w", padx=(8,0), pady=(6,0))
        ctk.CTkLabel(frm, text="Y").grid(row=3, column=0, sticky="w")
        self.entry_y = ctk.CTkEntry(frm, width=90); self.entry_y.grid(row=3, column=1, sticky="w", padx=(8,0))

        ctk.CTkLabel(frm, text="Largeur (groupe)").grid(row=4, column=0, sticky="w", pady=(10,0))
        self.entry_w = ctk.CTkEntry(frm, width=90); self.entry_w.grid(row=4, column=1, sticky="w", padx=(8,0), pady=(10,0))
        ctk.CTkLabel(frm, text="Hauteur (groupe)").grid(row=5, column=0, sticky="w")
        self.entry_h = ctk.CTkEntry(frm, width=90); self.entry_h.grid(row=5, column=1, sticky="w", padx=(8,0))

        # Appliquer seulement
        btns = ctk.CTkFrame(side); btns.pack(fill="x", padx=12, pady=(0,12))
        self.btn_apply = ctk.CTkButton(btns, text="Appliquer", command=self._apply_changes, state="disabled")
        self.btn_apply.pack(side="left", padx=6)

        # Entrée ↵ applique
        for e in (self.entry_name, self.entry_x, self.entry_y, self.entry_w, self.entry_h):
            e.bind("<Return>", lambda _e: self._apply_changes())

        # Status
        self.status = ctk.CTkLabel(self.root, text="Prêt", anchor="w"); self.status.pack(side="bottom", fill="x", padx=8, pady=6)

    def _bind_canvas_events(self):
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.root.bind("<Control-MouseWheel>", self._on_ctrl_wheel)
        self.root.bind("<Control-Key-0>", lambda e: self._zoom_fit())
        self.root.bind("<Control-Key-1>", lambda e: self._zoom_100())

    # ---------- Jeux / fichiers ----------
    def _default_base_dir(self, base_dir: Optional[str]) -> str:
        if base_dir:
            return os.path.abspath(base_dir)
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(here, "..", "config"))

    def _choose_base_dir(self):
        path = filedialog.askdirectory(title="Choisir le dossier racine (config)", initialdir=self.base_dir)
        if not path: return
        self.base_dir = path
        self.root_entry.delete(0, tk.END); self.root_entry.insert(0, self.base_dir)
        self._refresh_games_list()
        self.status.configure(text=f"Racine: {self.base_dir}")

    def _refresh_games_list(self):
        base = os.path.abspath(self.root_entry.get().strip() or self.base_dir)
        games = ZoneProject.list_games(base)
        if not games: games = ["(aucun)"]
        self.game_menu.configure(values=games)
        self.game_var.set(games[0])
        if games[0] != "(aucun)":
            self._on_select_game(games[0])

    def _on_select_game(self, game_name: str):
        if not game_name or game_name == "(aucun)": return
        base = os.path.abspath(self.root_entry.get().strip() or self.base_dir)
        try:
            self.project.load_game(base, game_name)
        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            return

        # Alignement au démarrage : pour chaque groupe lock_same_y, on ancre sur la 1ère clé triée
        self._startup_align_groups()

        self._prepare_display_image()
        self._reset_canvas()
        self._redraw_all()
        self._populate_regions_list()
        self._populate_group_menu()
        self._enable_after_load()
        self.status.configure(text=f"{game_name}: {len(self.project.regions)} région(s)")

        # Auto-sélectionne la 1ère région
        keys = sorted(self.project.regions.keys())
        if keys:
            first = keys[0]
            self._last_key = first
            self._select_key_in_list(first)
            self._on_select_region_from_list()

    def _startup_align_groups(self):
        # Regroupe par groupe
        groups: dict[str, List[str]] = {}
        for k, r in self.project.regions.items():
            g = r.get("group", "")
            groups.setdefault(g, []).append(k)
        # Pour chaque groupe lock_same_y → ancre sur la 1ère clé triée
        for g, keys in groups.items():
            if not self.project.group_has_lock_same_y(g):
                continue
            if not keys:
                continue
            keys_sorted = sorted(keys)
            anchor_key = keys_sorted[0]
            y = self.project.regions[anchor_key]["top_left"][1]
            # Déclenche la propagation via set_region_pos (gère clamp + align)
            x = self.project.regions[anchor_key]["top_left"][0]
            self.project.set_region_pos(anchor_key, x, y)

    # ---------- Image / zoom ----------
    def _prepare_display_image(self):
        W, H = self.project.image_size
        if W == 0 or H == 0:
            return
        self.base_scale = min(MAX_CANVAS_W / W, MAX_CANVAS_H / H, 1.0) or 1.0
        self._update_display_image()

    def _update_display_image(self):
        img = self.project.image
        if img is None: return
        self.scale = max(0.05, self.base_scale * max(0.1, float(self.user_zoom)))
        disp = img.resize((int(img.width * self.scale), int(img.height * self.scale)), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(disp)
        pct = int(round(self.scale / self.base_scale * 100)) if self.base_scale else int(self.scale*100)
        self.zoom_pct_label.configure(text=f"{pct}%")
        self._reset_canvas()

    def _reset_canvas(self):
        self.canvas.delete("all")
        self.rect_items.clear(); self.text_items.clear()
        w = self.tk_img.width() if self.tk_img else MAX_CANVAS_W
        h = self.tk_img.height() if self.tk_img else MAX_CANVAS_H
        self.canvas.config(width=max(w, MAX_CANVAS_W), height=max(h, MAX_CANVAS_H))
        if self.tk_img:
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    # ---------- Dessin ----------
    def _redraw_all(self):
        for rid in list(self.rect_items.values()): self.canvas.delete(rid)
        for tid in list(self.text_items.values()): self.canvas.delete(tid)
        self.rect_items.clear(); self.text_items.clear()
        W, H = self.project.image_size
        if W == 0: return
        s = self.scale
        for key, r in self.project.regions.items():
            group = r.get("group", "")
            w, h = self.project.get_group_size(group)
            x, y = r.get("top_left", [0, 0])
            dx0, dy0 = int(x * s), int(y * s)
            dx1, dy1 = int((x + w) * s), int((y + h) * s)
            rid = self.canvas.create_rectangle(dx0, dy0, dx1, dy1, outline="#0ea5e9", width=2)
            tid = self.canvas.create_text(dx0 + 6, dy0 + 6, anchor="nw", text=str(r.get("label", key)), fill="#0ea5e9", font=("Segoe UI", 10, "bold"))
            self.rect_items[key] = rid; self.text_items[key] = tid

    # ---------- Liste & champs ----------
    def _populate_regions_list(self):
        self.listbox.delete(0, tk.END)
        for k in sorted(self.project.regions.keys()):
            lab = self.project.regions[k].get("label", k)
            self.listbox.insert(tk.END, f"{k}  —  {lab}")

    def _populate_group_menu(self):
        groups = sorted(list(self.project.templates_resolved.keys())) or [""]
        self.group_menu.configure(values=groups)
        if groups: self.group_var.set(groups[0])

    def _populate_group_menu_keep_current(self):
        current = self.group_var.get()
        groups = sorted(list(self.project.templates_resolved.keys())) or [""]
        self.group_menu.configure(values=groups)
        if current in groups: self.group_var.set(current)
        elif groups: self.group_var.set(groups[0])

    def _current_selection_key(self) -> Optional[str]:
        sel = self.listbox.curselection()
        if not sel: return None
        line = self.listbox.get(sel[0])
        return line.split("  —  ", 1)[0]

    def _select_key_in_list(self, key: str):
        keys = sorted(self.project.regions.keys())
        if key in keys:
            idx = keys.index(key)
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(idx)
            self.listbox.activate(idx)

    def _on_select_region_from_list(self, event=None):
        key = self._current_selection_key()
        if not key: return
        r = self.project.regions.get(key, {})
        self.entry_name.delete(0, tk.END); self.entry_name.insert(0, key)
        group = r.get("group", "")
        if group not in self.project.templates:
            self.project.templates[group] = {"size": [60, 40], "type": "mix"}
        self._populate_group_menu_keep_current()
        self.group_var.set(group)
        x, y = r.get("top_left", [0, 0])
        self.entry_x.delete(0, tk.END); self.entry_x.insert(0, str(int(x)))
        self.entry_y.delete(0, tk.END); self.entry_y.insert(0, str(int(y)))
        gw, gh = self.project.get_group_size(group)
        self.entry_w.delete(0, tk.END); self.entry_w.insert(0, str(int(gw)))
        self.entry_h.delete(0, tk.END); self.entry_h.insert(0, str(int(gh)))
        self.btn_apply.configure(state="normal")
        self._last_key = key

    # ---------- Drag & drop ----------
    def _region_at_point(self, x: int, y: int) -> Optional[str]:
        for key, r in self.project.regions.items():
            gw, gh = self.project.get_group_size(r.get("group", ""))
            px, py = r.get("top_left", [0, 0])
            if px <= x <= px + gw and py <= y <= py + gh:
                return key
        return None

    def _on_mouse_down(self, event):
        s = self.scale if self.scale else 1.0
        x, y = int(event.x / s), int(event.y / s)
        key = self._region_at_point(x, y)
        if key:
            self.dragging_key = key
            tlx, tly = self.project.regions[key]["top_left"]
            self.drag_offset = (x - tlx, y - tly)
            self._select_key_in_list(key); self._on_select_region_from_list()
            self._last_key = key
        else:
            self.dragging_key = None

    def _on_mouse_drag(self, event):
        if not self.dragging_key: return
        key = self.dragging_key
        s = self.scale if self.scale else 1.0
        x = int(event.x / s) - self.drag_offset[0]
        y = int(event.y / s) - self.drag_offset[1]
        # Déplace + aligne si lock_same_y (géré par le modèle)
        self.project.set_region_pos(key, x, y)
        # Redessine le groupe impacté (plus efficient que tout redessiner)
        self._redraw_group(self.project.regions[key]["group"]) 
        # MAJ des champs X/Y
        px, py = self.project.regions[key]["top_left"]
        self.entry_x.delete(0, tk.END); self.entry_x.insert(0, str(px))
        self.entry_y.delete(0, tk.END); self.entry_y.insert(0, str(py))

    def _on_mouse_up(self, event):
        self.dragging_key = None

    def _redraw_group(self, group: str):
        s = self.scale if self.scale else 1.0
        # Trouver les clés du groupe
        keys = [k for k, r in self.project.regions.items() if r.get("group") == group]
        for k in keys:
            r = self.project.regions[k]
            w, h = self.project.get_group_size(group)
            x, y = r.get("top_left", [0, 0])
            dx0, dy0 = int(x * s), int(y * s)
            dx1, dy1 = int((x + w) * s), int((y + h) * s)
            rid = self.rect_items.get(k); tid = self.text_items.get(k)
            if rid: self.canvas.coords(rid, dx0, dy0, dx1, dy1)
            if tid: self.canvas.coords(tid, dx0 + 6, dy0 + 6)

    # ---------- Actions ----------
    def _apply_changes(self):
        print("[APPLY] start")
        try:
            key_from_entry = (self.entry_name.get() or "").strip()
            fallback_key = self._last_key
            keys_sorted = sorted(self.project.regions.keys())
            first_key = keys_sorted[0] if keys_sorted else None
            key = key_from_entry or fallback_key or first_key
            if not key:
                print("[APPLY] no regions in project")
                messagebox.showinfo("Info", "Sélectionne une région (clic dans la liste ou sur un rectangle).")
                return
            if key not in self.project.regions:
                print("[APPLY] key not found, using fallback order:", key)
                key = fallback_key or first_key
                if not key or key not in self.project.regions:
                    print("[APPLY] fallback also invalid")
                    return
            r = self.project.regions[key]
            renamed = False

            # rename
            try:
                new_key = (self.entry_name.get() or key).strip()
                if new_key and new_key != key:
                    new_key = self.project.rename_region(key, new_key)
                    key = new_key
                    r = self.project.regions[key]
                    self._last_key = key
                    renamed = True
            except Exception as e:
                print("[APPLY][rename] exception:", e)

            # group
            try:
                g = (self.group_var.get() or r.get("group", "")).strip()
                self.project.set_region_group(key, g)
            except Exception as e:
                print("[APPLY][group] exception:", e)

            # pos
            try:
                x = int(self.entry_x.get()); y = int(self.entry_y.get())
                self.project.set_region_pos(key, x, y)
            except Exception as e:
                print("[APPLY][pos] exception:", e)

            # size (propagé au groupe)
            try:
                nw = int(self.entry_w.get()); nh = int(self.entry_h.get())
                if nw > 0 and nh > 0:
                    g = self.project.regions[key]["group"]
                    self.project.set_group_size(g, nw, nh)
            except Exception as e:
                print("[APPLY][size] exception:", e)

            # MAJ UI (évite de casser la sélection si pas renommé)
            if renamed:
                self._populate_regions_list()
            # Redessine le groupe (si lock_same_y, plusieurs zones peuvent bouger)
            g = self.project.regions[key]["group"]
            self._redraw_group(g)
            self._select_key_in_list(key)
            self._on_select_region_from_list()
            self.status.configure(text=f"Modifié: {key}")
        except Exception as e:
            print("[APPLY] FATAL exception:", e)

    def _enable_after_load(self):
        self.btn_save.configure(state="normal")
        self.btn_apply.configure(state="normal")

    def _save_json(self):
        if self.project.image is None:
            messagebox.showinfo("Info", "Charge une image avant de sauvegarder.")
            return
        base = os.path.abspath(self.root_entry.get().strip() or self.base_dir)
        folder = os.path.join(base, self.project.current_game or "")
        init_dir = folder if os.path.isdir(folder) else os.path.dirname(self.project.image_path or "")
        out_path = filedialog.asksaveasfilename(
            title="Enregistrer",
            defaultextension=".json",
            initialdir=init_dir,
            initialfile="coordinates.json",
            filetypes=[("JSON", "*.json")]
        )
        if not out_path: return
        try:
            self.project.save_to(out_path)
            self.status.configure(text=f"Sauvegardé: {out_path}")
            messagebox.showinfo("Succès", f"Enregistré :\n{out_path}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'écrire le fichier:\n{e}")

    # ---------- Zoom ----------
    def _on_zoom_slider(self, value: float):
        self.user_zoom = float(value); self._update_display_image(); self._redraw_all()

    def _zoom_fit(self):
        self.user_zoom = 1.0; self.zoom_slider.set(1.0)
        self._update_display_image(); self._redraw_all()

    def _zoom_100(self):
        if self.base_scale == 0: return
        self.user_zoom = min(max(1.0 / self.base_scale, 0.25), 3.0)
        self.zoom_slider.set(self.user_zoom)
        self._update_display_image(); self._redraw_all()

    def _on_ctrl_wheel(self, event):
        step = 1.1 if event.delta > 0 else 1/1.1
        self.user_zoom = min(max(self.user_zoom * step, 0.25), 3.0)
        self.zoom_slider.set(self.user_zoom)
        self._update_display_image(); self._redraw_all()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ZoneEditorCTK()
    app.run()

```
### scripts/pyauto_helpers.py
```python
"""Thin wrapper around :mod:`objet.utils.pyauto` for backwards compatibility."""
from __future__ import annotations

from objet.utils.pyauto import locate_in_image

__all__ = ["locate_in_image"]

```
### scripts/quick_setup.py
```python
#!/usr/bin/env python3
"""quick_setup.py — pipeline de configuration rapide pour la capture des cartes.

Ce script enchaîne les étapes manuelles existantes :
  1. Éditer les zones via l'UI CustomTkinter.
  2. Rogner une vidéo test pour générer des crops.
  3. Identifier/labelliser les cartes manquantes.
  4. Valider la capture complète sur une vidéo.

Chaque étape peut être sautée avec --skip-*."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _print_header(title: str) -> None:
    line = "=" * max(10, len(title) + 4)
    print(f"\n{line}\n  {title}\n{line}")


def _run_zone_editor(game: str, config_root: Path) -> int:
    from position_zones_ctk import ZoneEditorCTK

    app = ZoneEditorCTK(base_dir=str(config_root))
    if game:
        try:
            app.game_var.set(game)
            app._on_select_game(game)  # type: ignore[attr-defined]
            print(f"ZoneEditor prêt sur le jeu '{game}'. Fermez la fenêtre pour continuer…")
        except Exception as exc:  # pragma: no cover - dépend de l'état local
            print(f"[WARN] Impossible de précharger le jeu '{game}': {exc}")
    app.run()
    return 0


def _run_crop(game_dir: Path, video: Optional[str], interval: float, out_dir: Optional[Path]) -> int:
    from Crop_Video_Frames import main as crop_main

    argv: List[str] = ["--game-dir", str(game_dir)]
    if video:
        argv += ["--video", video]
    if interval is not None:
        argv += ["--interval", str(interval)]
    if out_dir is not None:
        argv += ["--out", str(out_dir)]
    return int(crop_main(argv))


def _run_identify(game: str, crops_dir: Optional[Path], threshold: float, strict: float, trim: int, force_all: bool) -> int:
    from identify_card import main as identify_main

    argv: List[str] = ["--game", game, "--threshold", str(threshold), "--strict", str(strict), "--trim", str(trim)]
    if crops_dir is not None:
        argv += ["--crops-dir", str(crops_dir)]
    if force_all:
        argv.append("--force-all")
    return int(identify_main(argv))


def _run_capture_video(
    game: str,
    game_dir: Path,
    video: Optional[str],
    stride: int,
    num_th: float,
    suit_th: float,
    require_k: int,
) -> int:
    from capture_cards import main as capture_main

    argv: List[str] = [
        "--game",
        game,
        "--game-dir",
        str(game_dir),
        "--stride",
        str(stride),
        "--num-th",
        str(num_th),
        "--suit-th",
        str(suit_th),
        "--require-k",
        str(require_k),
    ]
    if video:
        argv += ["--video", video]
    else:
        raise SystemExit("La validation vidéo nécessite --video.")
    return int(capture_main(argv))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assistant de configuration rapide (zones → crops → cartes → capture)")
    parser.add_argument("--game", default="PMU", help="Nom du jeu (dossier dans config/)")
    parser.add_argument("--config-root", help="Chemin vers le dossier config/ (défaut: auto)")
    parser.add_argument("--video", help="Vidéo utilisée pour le crop et la validation")
    parser.add_argument("--crops-dir", help="Dossier de sortie des crops (défaut: config/<game>/debug/crops)")
    parser.add_argument("--crop-interval", type=float, default=3.0, help="Intervalle (s) entre deux crops vidéo")
    parser.add_argument("--identify-threshold", type=float, default=0.92, help="Seuil reco acceptée")
    parser.add_argument("--identify-strict", type=float, default=0.985, help="Seuil autoskip strict")
    parser.add_argument("--identify-trim", type=int, default=6, help="Rognage autour des patches (px)")
    parser.add_argument("--identify-force-all", action="store_true", help="Forcer l'UI sur toutes les cartes")
    parser.add_argument("--capture-stride", type=int, default=3, help="Traiter un frame sur N pour la validation vidéo")
    parser.add_argument("--capture-num-th", type=float, default=0.65, help="Seuil reconnaissance des valeurs")
    parser.add_argument("--capture-suit-th", type=float, default=0.65, help="Seuil reconnaissance des couleurs")
    parser.add_argument("--capture-require-k", type=int, default=2, help="Frames nécessaires pour stabiliser")
    parser.add_argument("--skip-zone-editor", action="store_true", help="Sauter l'étape d'édition des zones")
    parser.add_argument("--skip-crop", action="store_true", help="Sauter l'étape de crop vidéo")
    parser.add_argument("--skip-identify", action="store_true", help="Sauter l'étape d'identification des cartes")
    parser.add_argument("--skip-capture", action="store_true", help="Sauter la validation capture")
    parser.add_argument("--continue-on-error", action="store_true", help="Continuer même si une étape échoue")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    config_root = Path(args.config_root).resolve() if args.config_root else (PROJECT_ROOT / "config").resolve()
    game_dir = (config_root / args.game).resolve()
    if not game_dir.exists():
        print(f"ERREUR: dossier du jeu introuvable ({game_dir})")
        return 2

    crops_dir = Path(args.crops_dir).resolve() if args.crops_dir else (game_dir / "debug" / "crops")

    steps: List[Tuple[str, Callable[[], int]]] = []
    if not args.skip_zone_editor:
        steps.append(("Édition des zones", lambda: _run_zone_editor(args.game, config_root)))
    if not args.skip_crop:
        steps.append((
            "Crop vidéo",
            lambda: _run_crop(game_dir, args.video, float(args.crop_interval), crops_dir),
        ))
    if not args.skip_identify:
        steps.append((
            "Identification des cartes",
            lambda: _run_identify(
                args.game,
                crops_dir,
                float(args.identify_threshold),
                float(args.identify_strict),
                int(args.identify_trim),
                bool(args.identify_force_all),
            ),
        ))
    if not args.skip_capture:
        steps.append((
            "Validation capture vidéo",
            lambda: _run_capture_video(
                args.game,
                game_dir,
                args.video,
                int(args.capture_stride),
                float(args.capture_num_th),
                float(args.capture_suit_th),
                int(args.capture_require_k),
            ),
        ))

    status = 0
    for title, func in steps:
        _print_header(title)
        try:
            status = func()
        except SystemExit as exc:
            status = int(exc.code) if isinstance(exc.code, int) else 1
        except Exception as exc:  # pragma: no cover - dépend de l'exécution temps réel
            print(f"[ERREUR] {title}: {exc}")
            status = 1
        if status != 0:
            print(f"[ECHEC] {title} (code {status})")
            if not args.continue_on_error:
                break
    else:
        print("\nConfiguration rapide terminée ✅")
        return 0 if status == 0 else status

    return status


if __name__ == "__main__":
    raise SystemExit(main())

```
### scripts/state_requirements.py
```python
"""Re-export of :mod:`objet.services.script_state` for CLI consumers."""
from __future__ import annotations

from objet.services.script_state import (
    SCRIPT_STATE_USAGE,
    StatePortion,
    ScriptStateUsage,
    describe_scripts,
)

__all__ = [
    "StatePortion",
    "ScriptStateUsage",
    "SCRIPT_STATE_USAGE",
    "describe_scripts",
]

```
### scripts/zone_project.py
```python
# zone_project.py
# Logique "métier" : modèle, opérations, lecture/écriture JSON, clamp
# Aucune dépendance UI. Dépend de pillow uniquement pour charger l'image.

from __future__ import annotations
import os, json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from collections import OrderedDict
from PIL import Image

from objet.services.game import Game
from _utils import clamp_top_left, coerce_int, resolve_templates

def _load_templated_json(coord_path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(coord_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "templates" in data and "regions" in data:
            return data
    except Exception:
        pass
    return None

class ZoneProject:
    """
    Représente un "projet" de zones pour un jeu (image + zones + templates).
    Gère : chargement/écriture JSON, manipulations des régions, tailles de groupe…
    """

    def __init__(self, game: Optional[Game] = None) -> None:
        self.game = game or Game.for_script(Path(__file__).name)
        self.base_dir: str = ""
        self.current_game: Optional[str] = None
        self.image_path: Optional[str] = None
        self.image: Optional[Image.Image] = None
        self.table_capture: Dict[str, Any] = self.game.table.captures.table_capture

        # Données décrites par le JSON
        self.templates: Dict[str, Any] = self.game.table.captures.templates
        self._templates_resolved: Dict[str, Any] = {}
        # regions : key -> {"group": str, "top_left":[x,y], "value": Any, "label": str}
        self.regions: "OrderedDict[str, Dict[str, Any]]" = self.game.table.captures.regions

    # ---------- Propriétés utiles ----------
    @property
    def image_size(self) -> Tuple[int, int]:
        if self.image is None:
            return 0, 0
        return self.image.width, self.image.height

    @property
    def templates_resolved(self) -> Dict[str, Any]:
        # recalcul léger à la demande (ou fais-le sur set)
        return resolve_templates(self.templates)

    def get_group_size(self, group: str) -> Tuple[int, int]:
        size = self.templates_resolved.get(group, {}).get("size", [60, 40])
        return coerce_int(size[0], 60), coerce_int(size[1], 40)

    def group_has_lock_same_y(self, group: str) -> bool:
        layout = self.templates_resolved.get(group, {}).get("layout", {})
        return bool(layout.get("lock_same_y", False))

    # ---------- Découverte ----------
    @staticmethod
    def _find_expected_image(folder: str) -> Optional[str]:
        """Returns path to test_crop_result with supported extensions, else None."""
        bases = ["test_crop_result"]
        exts = [".png", ".jpg", ".jpeg"]
        for base in bases:
            for ext in exts:
                p = os.path.join(folder, base + ext)
                if os.path.isfile(p):
                    return p
        return None

    @staticmethod
    def list_games(base_dir: str) -> List[str]:
        games = []
        try:
            for name in sorted(os.listdir(base_dir)):
                full = os.path.join(base_dir, name)
                if os.path.isdir(full) and ZoneProject._find_expected_image(full):
                    games.append(name)
        except Exception:
            pass
        return games

    # ---------- Chargement ----------
    def load_game(self, base_dir: str, game_name: str) -> None:
        self.base_dir = os.path.abspath(base_dir)
        self.current_game = game_name

        folder = os.path.join(self.base_dir, game_name)
        img_path = ZoneProject._find_expected_image(folder) or os.path.join(folder, "test_crop_result.png")
        coord_path = os.path.join(folder, "coordinates.json")

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image introuvable: {img_path}")

        self.image_path = img_path
        self.image = Image.open(img_path).convert("RGBA")
        W, H = self.image_size
        self.table_capture.clear()
        self.table_capture.update({"enabled": True, "relative_bounds": [0, 0, W, H]})
        self.templates.clear()
        self.regions.clear()

        data = _load_templated_json(coord_path) if os.path.isfile(coord_path) else None
        if data:
            self.table_capture.update(data.get("table_capture", {}))
            self.templates.update(data.get("templates", {}))
            regs = data.get("regions", {})
            for key, r in regs.items():
                group = r.get("group", "")
                tl = r.get("top_left", [0, 0])
                self.regions[key] = {
                    "group": group,
                    "top_left": [coerce_int(tl[0]), coerce_int(tl[1])],
                    "value": r.get("value"),
                    "label": r.get("label", key),
                }
            # clamp soft à l’ouverture (pas d’auto-align à ce stade)
            self._clamp_all()
        else:
            # base minimale si pas de JSON
            self.templates.update({"action_button": {"size": [165, 70], "type": "texte"}})

        self._sync_game_capture()

    def _sync_game_capture(self) -> None:
        self.game.update_from_capture(
            table_capture=self.table_capture,
            regions=self.regions,
            templates=self.templates,
            reference_path=self.image_path,
        )

    # ---------- Opérations régions ----------
    def list_regions(self) -> List[str]:
        return list(self.regions.keys())

    def get_region(self, key: str) -> Dict[str, Any]:
        return self.regions[key]

    def rename_region(self, old_key: str, new_key: str) -> str:
        if new_key == old_key or new_key in self.regions or old_key not in self.regions:
            return old_key
        r = self.regions.pop(old_key)
        r["label"] = new_key
        self.regions[new_key] = r
        return new_key

    def set_region_group(self, key: str, group: str) -> None:
        if key not in self.regions:
            return
        if group not in self.templates:
            # crée un groupe par défaut si inconnu
            self.templates[group] = {"size": [60, 40], "type": "mix"}
        self.regions[key]["group"] = group
        self._clamp_region(key)
        # NB: on ne déclenche pas d’alignement global ici pour éviter les surprises
        # (la contrainte lock_same_y s’applique surtout lors des déplacements/edition de Y)

    def set_region_pos(self, key: str, x: int, y: int) -> None:
        """Déplace une région. Si le groupe a lock_same_y, aligne Y de toutes les régions du groupe."""
        if key not in self.regions:
            return
        g = self.regions[key]["group"]
        gw, gh = self.get_group_size(g)
        W, H = self.image_size

        # clamp et pose la région cible
        x, y = clamp_top_left(x, y, gw, gh, W, H)
        self.regions[key]["top_left"] = [x, y]

        # si contrainte d'alignement, propage Y
        if self.group_has_lock_same_y(g):
            self._enforce_lock_same_y(g, anchor_y=y)

    def add_region(self, group: str, name: Optional[str] = None) -> str:
        if group not in self.templates:
            self.templates[group] = {"size": [60, 40], "type": "mix"}
        W, H = self.image_size
        gw, gh = self.get_group_size(group)
        x, y = clamp_top_left(W // 2 - gw // 2, H // 2 - gh // 2, gw, gh, W, H)
        base = name or f"{group}_"
        if not name:
            i = 1
            while f"{base}{i}" in self.regions:
                i += 1
            key = f"{base}{i}"
        else:
            key = name
            if key in self.regions:
                i = 2
                while f"{key}_{i}" in self.regions:
                    i += 1
                key = f"{key}_{i}"
        self.regions[key] = {"group": group, "top_left": [x, y], "value": None, "label": key}
        return key

    def delete_region(self, key: str) -> None:
        self.regions.pop(key, None)

    # ---------- Opérations groupes ----------
    def set_group_size(self, group: str, w: int, h: int) -> None:
        if w <= 0 or h <= 0:
            return
        base = self.templates.get(group, {"type": "mix"})
        base["size"] = [int(w), int(h)]
        self.templates[group] = base
        # re-clamp toutes les régions de ce groupe
        for k, r in self.regions.items():
            if r.get("group") == group:
                self._clamp_region(k)
        # si lock_same_y → réaligner le Y commun
        if self.group_has_lock_same_y(group):
            self._enforce_lock_same_y(group, anchor_y=None)

    # ---------- Sauvegarde ----------
    def export_payload(self) -> Dict[str, Any]:
        W, H = self.image_size
        tc = self.table_capture or {"enabled": True, "relative_bounds": [0, 0, W, H]}
        out = {"table_capture": tc, "templates": self.templates, "regions": {}}
        for key, r in self.regions.items():
            out["regions"][key] = {
                "group": r.get("group", ""),
                "top_left": [int(r.get("top_left", [0, 0])[0]), int(r.get("top_left", [0, 0])[1])],
                "value": r.get("value", None),
                "label": r.get("label", key),
            }
        return out

    def save_to(self, path: str) -> None:
        payload = self.export_payload()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    # ---------- internes ----------
    def _clamp_all(self) -> None:
        for k in list(self.regions.keys()):
            self._clamp_region(k)

    def _clamp_region(self, key: str) -> None:
        if key not in self.regions:
            return
        W, H = self.image_size
        g = self.regions[key]["group"]
        gw, gh = self.get_group_size(g)
        x, y = self.regions[key].get("top_left", [0, 0])
        x, y = clamp_top_left(coerce_int(x), coerce_int(y), gw, gh, W, H)
        self.regions[key]["top_left"] = [x, y]

    def _enforce_lock_same_y(self, group: str, anchor_y: Optional[int]) -> None:
        """
        Aligne toutes les régions du groupe sur un même Y:
        - si anchor_y est fourni → on l'utilise (puis clamp commun).
        - sinon → on prend le min des Y existants (puis clamp commun).
        Clamp du X conservé par région.
        """
        # Liste des clés du groupe
        keys = [k for k, r in self.regions.items() if r.get("group") == group]
        if not keys:
            return

        gw, gh = self.get_group_size(group)
        W, H = self.image_size
        # Y cible
        if anchor_y is None:
            current_ys = [coerce_int(self.regions[k]["top_left"][1], 0) for k in keys]
            target_y = min(current_ys) if current_ys else 0
        else:
            target_y = coerce_int(anchor_y, 0)

        # clamp commun
        target_y = max(0, min(target_y, max(0, H - gh)))

        # Applique à tout le groupe (en clampant X individuellement)
        for k in keys:
            x, _y = self.regions[k]["top_left"]
            x, y = clamp_top_left(coerce_int(x), target_y, gw, gh, W, H)
            self.regions[k]["top_left"] = [x, y]

```
### tool.py
```python
import numpy as np
import time
import pyautogui
    
def click_icone(path,boucle=10,wait=0.3,gris=True,confidence=0.95):
    while boucle > 0:
        time.sleep(np.random.uniform(wait, wait*3)) 
        find = pyautogui.locateOnScreen(path, grayscale=gris, confidence=confidence)
        if find is not None:
            find = pyautogui.center(find)
            pyautogui.leftClick(find,duration=0.3)
            return True
        boucle -=1
    print(f'Looking For {path[7:]}')
    return False

def convert_to_float(s):
    """
    Converts a string representing a number into a float by inserting a decimal point
    before the last two digits if it doesn't already contain one.
    
    Parameters:
    s (str): The input string representing the number.
    
    Returns:
    float: The converted float value.
    """
    if not s or not isinstance(s, str):
        return None  # Return None if s is None or not a string

    # Remove any whitespace and replace comma with period
    s = s.strip().replace(',', '.')

    # Remove unwanted characters, keeping only digits and periods
    s_clean = ''.join(c for c in s if c.isdigit() or c == '.')

    if not s_clean or s_clean == '.':
        return None  # Return None if s_clean is empty or just a peri
    
    # Keep only digit characters
    s_digits = ''.join(filter(str.isdigit, s))
    
    # Ensure there are at least two digits
    s_digits = s_digits.zfill(2)
    
    # Insert decimal point before the last two digits
    s_new = s_digits[:-2] + '.' + s_digits[-2:]
    
    try:
        return float(s_new)
    except ValueError:
        raise ValueError(f"Invalid number format after processing: {s_new}")
        return None 

```