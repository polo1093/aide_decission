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
"""Paquetage structuré en entités, services et utilitaires."""

__all__ = ["entities", "services", "utils"]

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
from objet.entities.buttons import Buttons
from objet.entities.card import Card, CardsState
from objet.entities.player import Players
from objet.services.table import Table
from objet.scanner.cards_recognition import CardObservation
from objet.utils.capture import CaptureState
from objet.utils.metrics import MetricsState

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
    "Buttons",
    "MetricsState",
    "CaptureState",
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

from objet.entities.buttons import Buttons
from objet.entities.card import CardsState
from objet.entities.player import Fond , Players
from objet.utils.capture import CaptureState
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
    buttons: Buttons = field(default_factory=Buttons)
    captures: CaptureState = field(default_factory=CaptureState)
    scan: ScanTable = field(default_factory=ScanTable)
    pot: Fond = field(default_factory=Fond)
    new_party_flag: bool = False
    players : Players = field(default_factory=Players)
    
    
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
### objet/utils/capture.py
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
        if not isinstance(self.table_capture, dict):
            return None
        size = self.table_capture.get("size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            return [int(size[0]), int(size[1])]
        bounds = self.bounds
        if bounds and len(bounds) == 4:
            x1, y1, x2, y2 = bounds
            return [int(x2 - x1), int(y2 - y1)]
        return None

    @property
    def ref_offset(self) -> Optional[List[int]]:
        if not isinstance(self.table_capture, dict):
            return None
        offset = self.table_capture.get("ref_offset")
        if isinstance(offset, (list, tuple)) and len(offset) == 2:
            return [int(offset[0]), int(offset[1])]
        origin = self.origin
        if origin and len(origin) == 2:
            return [int(origin[0]), int(origin[1])]
        return None

    @property
    def bounds(self) -> Optional[List[int]]:
        if not isinstance(self.table_capture, dict):
            return None
        bounds = self.table_capture.get("bounds")
        if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
            return [int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])]
        return None

    @property
    def origin(self) -> Optional[List[int]]:
        if not isinstance(self.table_capture, dict):
            return None
        origin = self.table_capture.get("origin")
        if isinstance(origin, (list, tuple)) and len(origin) == 2:
            return [int(origin[0]), int(origin[1])]
        bounds = self.bounds
        if bounds and len(bounds) == 4:
            return bounds[:2]
        return None


__all__ = ["CaptureState"]

```
### objet/utils/metrics.py
```python
"""Gestion des métriques numériques de la table."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import tool
from objet.utils.state_utils import extract_scan_value


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
            self.player_money[key] = tool.convert_to_float(
                extract_scan_value(scan_table, raw_key)
            )
        self.players_count = sum(
            1 for money in self.player_money.values() if money not in (None, 0)
        )


__all__ = ["MetricsState"]

```
### objet/utils/state_utils.py
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