"""
Nouvelle orchestration UI (Section A) — test.py
------------------------------------------------
Objectif :
- UI Tkinter non bloquante avec `root.after` (pas de boucle infinie).
- Boutons Start/Stop pour activer/désactiver le scan en continu.
- Afficher l'état courant : cartes du joueur, board, FPS, message debug.
- Ne PAS dépendre pour l’instant de l’OCR ni des boutons/joueurs.

Intégration :
1) Coller ce fichier comme `test.py` à la racine du projet ou dans le dossier d’exécution.
2) Par défaut, le code démarre en mode **DEMO** (aucune dépendance). 
   - Pour brancher le pipeline réel, voir la classe `RealPipeline` et remplacer `get_pipeline()`.
3) Si vous gardez DEMO, lancez : `python test.py`.

Points de branchement vers votre code réel :
- Remplacer `get_pipeline()` pour retourner une instance de `RealPipeline()`.
- Dans `RealPipeline.tick()`: appeler votre pipeline existant
  (capture écran -> find_table -> coords -> scan_cartes -> game.update_from_scan)
  et **retourner** un dictionnaire `scan_table` aux clés :
    * player_card_1_number / player_card_1_symbol
    * player_card_2_number / player_card_2_symbol
    * board_card_1_number / board_card_1_symbol
    * ... jusqu’à board_card_5_* 
  Les valeurs : number ∈ {"A","K","Q","J","10","9",...}, symbol ∈ {"spades","hearts","diamonds","clubs"} ou None.

Robustesse :
- Aucune dépendance aux boutons/players. Tout est optionnel.
- Exceptions capturées dans `tick()`, affichées en UI, scan stoppé proprement.
"""
from __future__ import annotations

import os
import random
import time
import tkinter as tk
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# =====================
# Utils (formatage UI)
# =====================
SUIT_TO_SYMBOL = {
    "spades": "♠",
    "hearts": "♥",
    "diamonds": "♦",
    "clubs": "♣",
}

CARD_ORDER = ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"]
SUITS = list(SUIT_TO_SYMBOL.keys())


def format_card(value: Optional[str], suit: Optional[str]) -> str:
    if not value or not suit:
        return "?"
    return f"{value}{SUIT_TO_SYMBOL.get(suit, '?')}"


# =====================
# Pipelines (DEMO / REAL)
# =====================
class BasePipeline:
    """Interface minimale d'un pipeline de scan.

    Doit fournir :
      - tick() -> Dict[str, Dict[str, Optional[str]]]
        Retourne un `scan_table` (clé -> {"value": <str|None>}).
    """

    def tick(self) -> Dict[str, Dict[str, Optional[str]]]:  # pragma: no cover - interface
        raise NotImplementedError


class DemoPipeline(BasePipeline):
    """Pipeline de démonstration (aucune dépendance à votre code).

    Génère aléatoirement des cartes pour simuler un scan live.
    """

    def __init__(self) -> None:
        self._last_update = 0.0
        self._state = self._empty_table()

    @staticmethod
    def _empty_table() -> Dict[str, Dict[str, Optional[str]]]:
        table: Dict[str, Dict[str, Optional[str]]] = {}
        # joueurs (2 cartes)
        for i in (1, 2):
            table[f"player_card_{i}_number"] = {"value": None}
            table[f"player_card_{i}_symbol"] = {"value": None}
        # board (5 cartes)
        for i in range(1, 6):
            table[f"board_card_{i}_number"] = {"value": None}
            table[f"board_card_{i}_symbol"] = {"value": None}
        return table

    def _random_card(self) -> Tuple[str, str]:
        return random.choice(CARD_ORDER), random.choice(SUITS)

    def tick(self) -> Dict[str, Dict[str, Optional[str]]]:
        now = time.time()
        # Met à jour environ toutes les 0.8s pour visualiser les changements
        if now - self._last_update > 0.8:
            self._last_update = now
            st = self._empty_table()
            # 70% de chance d'avoir des cartes joueur
            if random.random() < 0.7:
                v, s = self._random_card()
                st["player_card_1_number"]["value"] = v
                st["player_card_1_symbol"]["value"] = s
            if random.random() < 0.7:
                v, s = self._random_card()
                st["player_card_2_number"]["value"] = v
                st["player_card_2_symbol"]["value"] = s
            # board progressif
            board_count = random.randint(0, 5)
            for i in range(1, board_count + 1):
                v, s = self._random_card()
                st[f"board_card_{i}_number"]["value"] = v
                st[f"board_card_{i}_symbol"]["value"] = s
            self._state = st
        return self._state


class RealPipeline(BasePipeline):
    """Branchez ICI votre pipeline existant.

    TODO (à faire côté projet) :
      - Importer vos vraies classes (ScanTable, Game, Controller si nécessaire)
      - Implémenter `tick()` pour :
          1) exécuter un scan écran/table
          2) mettre à jour l'état du jeu
          3) retourner un dict `scan_table` aux clés attendues
    """

    def __init__(self) -> None:
        # Exemple (à adapter) :
        # from src.scan import ScanTable
        # from src.game import Game
        # self.scan = ScanTable()
        # self.game = Game()
        raise NotImplementedError("Branchez votre pipeline et retirez cette exception.")

    def tick(self) -> Dict[str, Dict[str, Optional[str]]]:  # pragma: no cover - intégration projet
        # Exemple d'implémentation :
        # success = self.scan.scan()
        # if success:
        #     self.game.update_from_scan(self.scan.table)
        # return self.scan.table
        raise NotImplementedError


def get_pipeline() -> BasePipeline:
    """Choisit le pipeline. Par défaut : DEMO.
    Pour passer en réel, remplacez par `return RealPipeline()`.
    """
    if os.environ.get("POKER_UI_PIPELINE", "DEMO").upper() == "REAL":
        return RealPipeline()  # lèvera NotImplemented tant que non branché
    return DemoPipeline()


# =====================
# Contrôleur UI + boucle after
# =====================
@dataclass
class UiState:
    player_cards: Tuple[str, str]
    board_cards: Tuple[str, str, str, str, str]
    message: str
    fps: float


class App:
    SCAN_INTERVAL_MS = 150  # ~6-7 FPS visés sans saturer la boucle Tk

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Poker UI — Live Scan (A: Orchestration)")
        self.pipeline = RealPipeline()

        # Flags & perf
        self.scanning = False
        self._last_tick_ts = None  # type: Optional[float]
        self._last_fps = 0.0

        # Widgets
        self._build_widgets()

        # Raccourcis
        self.root.bind("<Escape>", lambda e: self.on_close())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # -------- UI construction --------
    def _build_widgets(self) -> None:
        pad = {"padx": 10, "pady": 8}

        # Top: boutons Start/Stop + FPS
        top = tk.Frame(self.root)
        top.pack(fill=tk.X, **pad)

        self.btn_start = tk.Button(top, text="▶ Start", width=10, command=self.start_scan)
        self.btn_stop = tk.Button(top, text="■ Stop", width=10, state=tk.DISABLED, command=self.stop_scan)
        self.lbl_fps = tk.Label(top, text="FPS: 0.0")

        self.btn_start.pack(side=tk.LEFT)
        self.btn_stop.pack(side=tk.LEFT, padx=(10, 0))
        self.lbl_fps.pack(side=tk.RIGHT)

        # Player cards
        grp_player = tk.LabelFrame(self.root, text="Vos cartes")
        grp_player.pack(fill=tk.X, **pad)
        self.lbl_p1 = tk.Label(grp_player, text="?", font=("Consolas", 20))
        self.lbl_p2 = tk.Label(grp_player, text="?", font=("Consolas", 20))
        self.lbl_p1.pack(side=tk.LEFT, padx=5)
        self.lbl_p2.pack(side=tk.LEFT, padx=5)

        # Board cards
        grp_board = tk.LabelFrame(self.root, text="Board")
        grp_board.pack(fill=tk.X, **pad)
        self.lbl_b = [tk.Label(grp_board, text="?", font=("Consolas", 20)) for _ in range(5)]
        for lab in self.lbl_b:
            lab.pack(side=tk.LEFT, padx=5)

        # Message / erreurs
        self.lbl_msg = tk.Label(self.root, text="Prêt.")
        self.lbl_msg.pack(fill=tk.X, **pad)

    # -------- Start/Stop --------
    def start_scan(self) -> None:
        if self.scanning:
            return
        self.scanning = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self._last_tick_ts = time.time()
        self._schedule_next_tick()

    def stop_scan(self) -> None:
        self.scanning = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

    def on_close(self) -> None:
        self.stop_scan()
        self.root.after(50, self.root.destroy)

    # -------- Boucle after --------
    def _schedule_next_tick(self) -> None:
        if self.scanning:
            self.root.after(self.SCAN_INTERVAL_MS, self.tick)

    def tick(self) -> None:
        start = time.time()
        try:
            scan_table = self.pipeline.tick()
            ui_state = self._build_ui_state(scan_table)
            self._render(ui_state)
            # FPS
            dt = max(1e-6, time.time() - start)
            self._last_fps = 1.0 / dt
            self.lbl_fps.config(text=f"FPS: {self._last_fps:.1f}")
        except Exception as e:  # robustesse : on n'effondre pas l'UI
            self.lbl_msg.config(text=f"Erreur: {e}")
            self.stop_scan()
        finally:
            self._schedule_next_tick()

    # -------- Rendu --------
    def _build_ui_state(self, scan_table: Dict[str, Dict[str, Optional[str]]]) -> UiState:
        def gv(k: str) -> Optional[str]:
            d = scan_table.get(k) or {}
            return d.get("value")

        # Player
        p1 = format_card(gv("player_card_1_number"), gv("player_card_1_symbol"))
        p2 = format_card(gv("player_card_2_number"), gv("player_card_2_symbol"))

        # Board
        board = []
        for i in range(1, 6):
            board.append(format_card(gv(f"board_card_{i}_number"), gv(f"board_card_{i}_symbol")))
        board_tup = tuple(board)  # type: ignore

        # Message simple
        known = sum(c != "?" for c in (p1, p2, *board))
        msg = f"Cartes connues: {known}/7"

        return UiState(player_cards=(p1, p2), board_cards=board_tup, message=msg, fps=self._last_fps)

    def _render(self, s: UiState) -> None:
        self.lbl_p1.config(text=s.player_cards[0])
        self.lbl_p2.config(text=s.player_cards[1])
        for lab, txt in zip(self.lbl_b, s.board_cards):
            lab.config(text=txt)
        self.lbl_msg.config(text=s.message)


# =====================
# Entrée programme
# =====================

def main() -> None:
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
