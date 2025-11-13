#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI orchestration (Part A, non-demo, FIXED)
=========================================

- Boucle Tkinter non bloquante (`after`), Start/Stop/Snapshot, interval paramétrable.
- Aucune donnée simulée : on n'affiche *que* ce que renvoie le scanner réel.
- Import dynamique des classes via chemins "module:Class" passés en CLI (ou autodétection).
- Mapping direct `scan_table` -> affichage (cartes, board, boutons, pot/fond, bankrolls).
- Intégration `Game` optionnelle : si présente et possède `update_from_scan`,
  on calcule win_chance/EV/recommandation via `game` (sinon laissé vide).

Correctif majeur
----------------
Renommage de la méthode `_bind_keys()` (au lieu de `_bind`) pour éviter le conflit
avec la méthode interne `tkinter.Misc._bind()` qui provoquait :
`TypeError: App._bind() takes 1 positional argument but 5 were given`.

Exemples
--------
$ python ui_main.py \
    --scanner objet.scan:ScanTable \
    --game objet.game:Game

$ python ui_main.py --scanner folder_tool.scan:ScanTable

Notes
-----
- Les imports échoués n'empêchent pas le démarrage de l'UI.
- Le scanner doit fournir un `scan_table` dict avec des clés du type :
  player_card_1_number/symbol, player_card_2_*, board_card_1..5_*,
  button_1..3, pot, fond, player_money_J1..J5.
- `Game` est facultatif; s'il est absent, `win_chance/ev/reco` restent vides.
"""

from __future__ import annotations
import argparse
import importlib
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

import tkinter as tk
from tkinter import ttk

# ----------------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------------

SUIT_SYMBOLS = {
    "spades": "♠", "hearts": "♥", "diamonds": "♦", "clubs": "♣",
    "pique": "♠", "coeur": "♥", "carreau": "♦", "trefle": "♣",
}

def fmt_card(card: Optional[Tuple[Optional[str], Optional[str]]]) -> str:
    if not card:
        return "?"
    v, s = card
    if not v or not s:
        return "?"
    return f"{v}{SUIT_SYMBOLS.get(str(s).lower(), '?')}"

def safe_float_str(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.2f}"

def pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x*100:.1f}%"

def to_float_safe(txt: Optional[str]) -> Optional[float]:
    if not txt:
        return None
    clean = str(txt).replace("€", "").replace(" ", "").replace(",", ".")
    try:
        return float(clean)
    except Exception:
        return None

def getv(d: Dict[str, Any], key: str) -> Optional[str]:
    try:
        return d[key]["value"]
    except Exception:
        return None

def load_class(dotted: str):
    """
    Charge une classe depuis un chemin 'package.mod:Class'.
    Retourne None si l'import échoue.
    """
    if not dotted or ":" not in dotted:
        return None
    mod_name, cls_name = dotted.split(":", 1)
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, cls_name)
    except Exception:
        return None

# ----------------------------------------------------------------------------
# UI State
# ----------------------------------------------------------------------------

@dataclass
class ButtonInfo:
    name: Optional[str] = None
    value: Optional[float] = None

@dataclass
class PlayerInfo:
    seat: int
    bankroll: Optional[float] = None

@dataclass
class UIState:
    player_cards: List[Optional[Tuple[Optional[str], Optional[str]]]] = field(default_factory=lambda: [None, None])
    board_cards: List[Optional[Tuple[Optional[str], Optional[str]]]] = field(default_factory=lambda: [None]*5)
    pot: Optional[float] = None
    fond: Optional[float] = None
    players: List[PlayerInfo] = field(default_factory=list)
    buttons: List[ButtonInfo] = field(default_factory=lambda: [ButtonInfo(), ButtonInfo(), ButtonInfo()])
    win_chance: Optional[float] = None
    ev: Optional[float] = None
    recommended: Optional[ButtonInfo] = None
    last_scan_ms: Optional[float] = None
    fps: Optional[float] = None

# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self, scanner_cls, game_cls=None, scan_interval_ms: int = 250):
        super().__init__()
        self.title("Live Table – UI Orchestrator")
        self.geometry("880x640")
        self.minsize(780, 520)

        # Instances backend (facultatives/optionnelles)
        self.scanner = scanner_cls() if scanner_cls else None
        self.game = game_cls() if game_cls else None

        # Orchestration
        self.scanning = False
        self.scan_interval_ms = scan_interval_ms
        self._last_tick_t = None

        # State
        self.state = UIState()

        # UI
        self._build()
        self._layout()
        self._bind_keys()  # <— renommé pour éviter le conflit tkinter
        self._render()

    # ---------- UI ----------

    def _build(self):
        self.frm_top = ttk.Frame(self)
        self.btn_start = ttk.Button(self.frm_top, text="▶ Start", command=self.start_scan)
        self.btn_stop = ttk.Button(self.frm_top, text="⏸ Stop", command=self.stop_scan)
        self.btn_snap = ttk.Button(self.frm_top, text="📸 Snapshot", command=self.snapshot_once)
        self.lbl_interval = ttk.Label(self.frm_top, text="Interval (ms):")
        self.var_interval = tk.StringVar(value=str(self.scan_interval_ms))
        self.ent_interval = ttk.Entry(self.frm_top, width=6, textvariable=self.var_interval)
        self.lbl_status = ttk.Label(self.frm_top, text="Ready.", width=30, anchor="w")

        self.frm_center = ttk.Frame(self)
        self.txt = tk.Text(self.frm_center, height=26, wrap="none", font=("Consolas", 12), state="disabled")
        self.scroll = ttk.Scrollbar(self.frm_center, orient="vertical", command=self.txt.yview)
        self.txt.configure(yscrollcommand=self.scroll.set)

        self.frm_bottom = ttk.Frame(self)
        self.var_perf = tk.StringVar(value="scan: — ms | fps: —")
        self.lbl_perf = ttk.Label(self.frm_bottom, textvariable=self.var_perf)

    def _layout(self):
        self.frm_top.pack(side="top", fill="x", padx=10, pady=8)
        self.btn_start.pack(side="left", padx=(0, 6))
        self.btn_stop.pack(side="left", padx=(0, 12))
        self.btn_snap.pack(side="left", padx=(0, 18))
        self.lbl_interval.pack(side="left")
        self.ent_interval.pack(side="left", padx=(6, 18))
        self.lbl_status.pack(side="left", padx=6)

        self.frm_center.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 6))
        self.txt.pack(side="left", fill="both", expand=True)
        self.scroll.pack(side="right", fill="y")

        self.frm_bottom.pack(side="bottom", fill="x", padx=10, pady=(0, 10))
        self.lbl_perf.pack(side="left")

    def _bind_keys(self):
        # IMPORTANT : ne pas nommer cette méthode `_bind` pour éviter le conflit avec tkinter
        self.bind("<Escape>", lambda e: self.stop_scan())
        self.bind("<F5>", lambda e: self.start_scan())
        self.bind("<F6>", lambda e: self.snapshot_once())
        self.ent_interval.bind("<Return>", lambda e: self._update_interval())

    # ---------- Orchestration ----------

    def start_scan(self):
        self._update_interval()
        if not self.scanner:
            self.lbl_status.configure(text="Scanner non importé (voir --scanner).")
            return
        if not self.scanning:
            self.scanning = True
            self.lbl_status.configure(text="Scanning…")
            self._last_tick_t = time.time()
            self.after(self.scan_interval_ms, self._tick)

    def stop_scan(self):
        self.scanning = False
        self.lbl_status.configure(text="Stopped.")

    def snapshot_once(self):
        t0 = time.perf_counter()
        self._poll_once()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.state.last_scan_ms = dt_ms
        self.state.fps = None
        self._render()
        self.var_perf.set(f"scan: {dt_ms:.1f} ms | fps: —")
        self.lbl_status.configure(text="Snapshot done.")

    def _tick(self):
        if not self.scanning:
            return
        t0 = time.perf_counter()
        self._poll_once()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.state.last_scan_ms = dt_ms

        now = time.time()
        if self._last_tick_t:
            dt_s = max(now - self._last_tick_t, 1e-6)
            self.state.fps = 1.0 / dt_s
        self._last_tick_t = now

        self._render()
        fps_txt = f"{self.state.fps:.1f}" if self.state.fps else "—"
        self.var_perf.set(f"scan: {dt_ms:.1f} ms | fps: {fps_txt}")
        self.after(self.scan_interval_ms, self._tick)

    def _update_interval(self):
        try:
            v = int(self.var_interval.get())
            self.scan_interval_ms = max(25, min(2000, v))
        except Exception:
            self.scan_interval_ms = 250
            self.var_interval.set(str(self.scan_interval_ms))

    # ---------- Backend bridge ----------

    def _poll_once(self):
        """
        Récupère un scan_table réel depuis le scanner,
        met à jour l'UIState, et (si dispo) alimente Game pour win/EV/reco.
        """
        try:
            out = self.scanner.scan()  # scanner réel attendu
            scan_table = None
            if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
                scan_table = out[1]
            elif isinstance(out, dict):
                scan_table = out
            elif hasattr(self.scanner, 'table'):
                scan_table = getattr(self.scanner, 'table', None)
            if not isinstance(scan_table, dict):
                self.lbl_status.configure(text="Scan invalide (pas de dict).")
                return
        except Exception as e:
            self.lbl_status.configure(text=f"Scan error: {e}")
            return

        # ------ Mapping direct scan_table -> UIState (cartes/board/boutons/montants) ------
        pc1 = (getv(scan_table, "player_card_1_number"), getv(scan_table, "player_card_1_symbol"))
        pc2 = (getv(scan_table, "player_card_2_number"), getv(scan_table, "player_card_2_symbol"))
        board = [(getv(scan_table, f"board_card_{i}_number"), getv(scan_table, f"board_card_{i}_symbol")) for i in range(1, 6)]
        buttons = []
        for i in range(1, 4):
            raw = getv(scan_table, f"button_{i}")
            buttons.append(ButtonInfo(name=raw, value=None))

        pot = to_float_safe(getv(scan_table, "pot"))
        fond = to_float_safe(getv(scan_table, "fond"))
        players = []
        for i in range(1, 6):
            txt = getv(scan_table, f"player_money_J{i}")
            if txt is None and i > 2:
                break
            players.append(PlayerInfo(seat=i, bankroll=to_float_safe(txt)))

        # ------ Optionnel: passer par Game pour win/EV/reco ------
        win_chance = None
        ev = None
        recommended = None
        if self.game and hasattr(self.game, "update_from_scan"):
            try:
                self.game.update_from_scan(scan_table)
                if hasattr(self.game, "decision"):
                    reco = self.game.decision()
                    if isinstance(reco, str):
                        recommended = ButtonInfo(name=reco, value=None)
                    elif isinstance(reco, dict):
                        recommended = ButtonInfo(name=str(reco.get("name") or reco.get("button")), value=reco.get("value"))
                win_chance = getattr(self.game, "win_chance", None)
                ev = getattr(self.game, "ev", None)
            except Exception as e:
                self.lbl_status.configure(text=f"Game error: {e}")

        # Maj UIState
        self.state.player_cards = [pc1, pc2]
        self.state.board_cards = board
        self.state.buttons = buttons
        self.state.pot = pot
        self.state.fond = fond
        self.state.players = players
        self.state.win_chance = win_chance
        self.state.ev = ev
        self.state.recommended = recommended

    # ---------- Rendering ----------

    def _render(self):
        s = self._format_text(self.state)
        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", s)
        self.txt.configure(state="disabled")

    @staticmethod
    def _format_text(st: UIState) -> str:
        pc_line = " ".join(fmt_card(c) for c in st.player_cards)
        board_line = " ".join(fmt_card(c) for c in st.board_cards)
        buttons_line = "  |  ".join([
            f"B{i}:{(b.name if b.name else '—')}{(' '+safe_float_str(b.value)) if b.value is not None else ''}"
            for i, b in enumerate(st.buttons, start=1)
        ])
        players_line = "  |  ".join([f"J{p.seat}:{'Absent' if p.bankroll is None else safe_float_str(p.bankroll)}" for p in st.players])
        win_txt = pct(st.win_chance)
        ev_txt = safe_float_str(st.ev)
        reco_txt = "—"
        if st.recommended and st.recommended.name:
            reco_txt = f"{st.recommended.name}"
            if st.recommended.value is not None:
                reco_txt += f" {safe_float_str(st.recommended.value)}"

        perf = []
        perf.append(f"scan: {safe_float_str(st.last_scan_ms)} ms" if st.last_scan_ms is not None else "scan: — ms")
        perf.append(f"fps: {st.fps:.1f}" if st.fps is not None else "fps: —")

        lines = [
            "=== TABLE LIVE ===",
            "",
            f"Cartes joueur : {pc_line}",
            f"Board        : {board_line}",
            "",
            f"Pot : {safe_float_str(st.pot)}    |  Fond (tapis) : {safe_float_str(st.fond)}",
            f"Joueurs      : {players_line if players_line else '—'}",
            "",
            f"Chances de gain : {win_txt}",
            f"EV (attendue)   : {ev_txt}",
            f"Recommandation  : {reco_txt}",
            "",
            f"Boutons        : {buttons_line}",
            "",
            "Astuce: F5 = Start, F6 = Snapshot, Échap = Stop",
            "",
            "—",
        ]
        return "\n".join(lines)

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def autodetect_classes() -> Tuple[Optional[type], Optional[type]]:
    """
    Tente quelques chemins courants pour trouver ScanTable et Game.
    Retourne (scanner_cls, game_cls), chacun pouvant être None.
    """
    candidates_scanner = [
        "objet.scanner.scan:ScanTable",
        "objet.scan:ScanTable",
        "folder_tool.scan:ScanTable",
        "scan:ScanTable",
        "tool:ScanTable",
    ]
    candidates_game = [
        "objet.services.game:Game",
        "objet.game:Game",
        "game:Game",
        "objet.Game:Game",
    ]
    scanner_cls = None
    game_cls = None
    for dotted in candidates_scanner:
        scanner_cls = load_class(dotted)
        if scanner_cls:
            break
    for dotted in candidates_game:
        game_cls = load_class(dotted)
        if game_cls:
            break
    return scanner_cls, game_cls


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="UI orchestrator (non-demo)")
    ap.add_argument("--scanner", help="Chemin 'module:Class' du scanner (ex: objet.scan:ScanTable)", default=None)
    ap.add_argument("--game", help="Chemin 'module:Class' du Game (ex: objet.game:Game)", default=None)
    ap.add_argument("--interval", type=int, default=250, help="Intervalle de scan en ms (25..2000)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    scanner_cls = load_class(args.scanner) if args.scanner else None
    game_cls = load_class(args.game) if args.game else None
    if not scanner_cls and not game_cls:
        scanner_cls, game_cls = autodetect_classes()
    app = App(scanner_cls=scanner_cls, game_cls=game_cls, scan_interval_ms=args.interval)
    app.mainloop()


if __name__ == "__main__":
    main()
