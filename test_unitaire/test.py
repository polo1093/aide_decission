"""Interface de test basée sur Tkinter pour piloter le contrôleur."""

import queue
import threading
import time
import tkinter as tk
from tkinter import scrolledtext
from typing import Optional

from objet import controller


UPDATE_DELAY_MS = 100  # Intervalle de rafraîchissement pour traiter la file
SCAN_SLEEP_SECONDS = 0.5  # Pause entre deux scans successifs


class ControllerUI:
    def __init__(self) -> None:
        self.controller = controller.Controller()
        self.scan_queue: "queue.Queue[str]" = queue.Queue()
        self.scan_thread: Optional[threading.Thread] = None

        self.root = tk.Tk()
        self.root.title("Interface Graphique")
        # Positionne la fenêtre en reprenant les dimensions de l'ancienne UI
        self.root.geometry("1200x800+3245+140")

        self._build_widgets()
        self._configure_callbacks()

        # Lance le cycle de traitement de la file de messages
        self.root.after(UPDATE_DELAY_MS, self.process_queue)

    def _build_widgets(self) -> None:
        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.scan_button = tk.Button(button_frame, text="Lancer le scan", command=self.start_scan)
        self.scan_button.pack(side=tk.LEFT, padx=5)

        self.draw_button = tk.Button(button_frame, text="Draw", command=self.draw)
        self.draw_button.pack(side=tk.LEFT, padx=5)

        self.quit_button = tk.Button(button_frame, text="Quitter", command=self.quit)
        self.quit_button.pack(side=tk.RIGHT, padx=5)

        self.output = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=80, height=20)
        self.output.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.output.configure(state=tk.DISABLED)

    def _configure_callbacks(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def start_scan(self) -> None:
        if self.controller.running:
            return

        self.controller.running = True
        if not self.scan_thread or not self.scan_thread.is_alive():
            self.scan_thread = threading.Thread(target=self.scan_loop, daemon=True)
            self.scan_thread.start()

    def scan_loop(self) -> None:
        while self.controller.running:
            texte_table = self.controller.main()
            if texte_table is not None:
                self.scan_queue.put(texte_table)
            time.sleep(SCAN_SLEEP_SECONDS)

    def draw(self) -> None:
        self.controller.draw()

    def quit(self) -> None:
        self.controller.running = False
        self.root.after(UPDATE_DELAY_MS, self.root.destroy)

    def process_queue(self) -> None:
        try:
            while True:
                texte_table = self.scan_queue.get_nowait()
                self.update_output(texte_table)
        except queue.Empty:
            pass
        finally:
            self.root.after(UPDATE_DELAY_MS, self.process_queue)

    def update_output(self, texte_table: str) -> None:
        self.output.configure(state=tk.NORMAL)
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, texte_table)
        self.output.configure(state=tk.DISABLED)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    ControllerUI().run()
