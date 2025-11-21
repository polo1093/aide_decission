
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

import logging

from objet.services.controller import Controller   # <--- IMPORTANT : import direct


logger = logging.getLogger(__name__)


class App(tk.Tk):
    def __init__(self, controller: Controller, scan_interval_ms: int = 1000):
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
            self.scan_interval_ms = 1000
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
        except Exception as exc:
            self._handle_controller_exception(context="Snapshot", error=exc)
            raise

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
        except Exception as exc:
            self._handle_controller_exception(context="Scan continu", error=exc)
            raise
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

    def _handle_controller_exception(self, context: str, error: Exception):
        """Affiche l'erreur côté UI et loggue le détail pour la console."""
        self.last_call_ms = None
        self.fps = None
        self._set_text(f"Erreur Controller.main(): {error}")
        self.var_perf.set("scan: — ms | fps: —")
        self.lbl_status.configure(text="Erreur controller.")
        logger.exception("Erreur lors de %s", context)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="UI simple autour de Controller.main()")
    ap.add_argument(
        "--interval",
        type=int,
        default=1000,
        help="Intervalle entre deux appels à main() en ms (25..2000)",
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    controller = Controller()
    app = App(controller=controller, scan_interval_ms=args.interval)
    app.mainloop()


if __name__ == "__main__":
    main()
