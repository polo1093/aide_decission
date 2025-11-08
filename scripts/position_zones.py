# position_zones_ctk.py
# Éditeur de zones (CustomTkinter + tkinter.Canvas)
# - Clic gauche + glisser : créer une zone
# - Bouton "Annuler" : supprime la dernière zone
# - Bouton "Effacer" : supprime toutes les zones
# - Bouton "Enregistrer" : exporte zones en JSON (pixels + normalisées)
# - Bouton "Ouvrir image" : choisir une image (PNG/JPG...)
# Dépendances : customtkinter, pillow

import os, json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import customtkinter as ctk
from datetime import datetime

APP_TITLE = "Zone Editor (CustomTkinter)"
MAX_CANVAS_W = 1280
MAX_CANVAS_H = 800

class ZoneEditorCTK:
    def __init__(self):
        ctk.set_appearance_mode("light")           # "light" | "dark" | "system"
        ctk.set_default_color_theme("blue")        # "blue" | "green" | "dark-blue"

        self.root = ctk.CTk()
        self.root.title(APP_TITLE)
        self.root.geometry("1400x900")

        # ---- Etat ----
        self.img_path = None
        self.img_pil_original = None        # PIL.Image (taille d'origine)
        self.img_display = None             # PIL.Image redimensionnée pour affichage
        self.tk_img = None                  # ImageTk.PhotoImage (référence à garder)
        self.scale = 1.0                    # facteur d'échelle uniforme (display -> original)
        self.zones = []                     # liste de dicts
        self.rect_items = []                # ids tkinter Canvas des rectangles
        self.text_items = []                # ids des labels sur le canvas
        self.drag_start = None              # (x, y) départ
        self.current_rect = None            # id rectangle en cours de tracé

        # ---- Layout principal ----
        self._build_ui()
        self._bind_canvas_events()

    def _build_ui(self):
        # Top bar
        self.topbar = ctk.CTkFrame(self.root, corner_radius=0)
        self.topbar.pack(side="top", fill="x")

        self.btn_open = ctk.CTkButton(self.topbar, text="Ouvrir image", command=self.open_image)
        self.btn_open.pack(side="left", padx=8, pady=8)

        self.btn_undo = ctk.CTkButton(self.topbar, text="Annuler", command=self.undo_zone, state="disabled")
        self.btn_undo.pack(side="left", padx=8, pady=8)

        self.btn_clear = ctk.CTkButton(self.topbar, text="Effacer", command=self.clear_zones, state="disabled")
        self.btn_clear.pack(side="left", padx=8, pady=8)

        self.btn_save = ctk.CTkButton(self.topbar, text="Enregistrer", command=self.save_zones, state="disabled")
        self.btn_save.pack(side="left", padx=8, pady=8)

        self.info_label = ctk.CTkLabel(self.topbar, text="Aucune image", anchor="w")
        self.info_label.pack(side="left", padx=16, pady=8)

        # Main area (left: canvas, right: sidebar)
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(side="top", fill="both", expand=True)

        self.canvas_frame = ctk.CTkFrame(self.main_frame)
        self.canvas_frame.pack(side="left", fill="both", expand=True, padx=(10, 5), pady=10)

        # tkinter.Canvas pour dessiner (CustomTkinter n’a pas besoin d’un canvas custom)
        self.canvas = tk.Canvas(self.canvas_frame, bg="#F2F2F2", highlightthickness=0, width=MAX_CANVAS_W, height=MAX_CANVAS_H)
        self.canvas.pack(fill="both", expand=True)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self.main_frame, width=320)
        self.sidebar.pack(side="right", fill="y", padx=(5, 10), pady=10)

        self.lbl_title = ctk.CTkLabel(self.sidebar, text="Zones", font=ctk.CTkFont(size=18, weight="bold"))
        self.lbl_title.pack(anchor="w", padx=12, pady=(12, 4))

        self.zones_text = ctk.CTkTextbox(self.sidebar, width=300, height=700)
        self.zones_text.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.zones_text.insert("1.0", "Charge une image puis dessine des zones (cliquer-glisser).")
        self.zones_text.configure(state="disabled")

        # Status bar
        self.status = ctk.CTkLabel(self.root, text="Prêt", anchor="w")
        self.status.pack(side="bottom", fill="x", padx=8, pady=6)

    def _bind_canvas_events(self):
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

    # ----------------- Image -----------------
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"), ("Tous les fichiers", "*.*")]
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGBA")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'ouvrir l'image:\n{e}")
            return

        self.img_path = path
        self.img_pil_original = img
        self._prepare_display_image(img)
        self._reset_canvas_for_image()
        self._update_info()
        self.status.configure(text=f"Image chargée: {os.path.basename(path)}  ({img.width}x{img.height})")

        # Activer boutons
        self.btn_undo.configure(state="normal")
        self.btn_clear.configure(state="normal")
        self.btn_save.configure(state="normal")

    def _prepare_display_image(self, img: Image.Image):
        # Calculer l’échelle pour rentrer dans le canvas
        w, h = img.size
        scale = min(MAX_CANVAS_W / w, MAX_CANVAS_H / h, 1.0)  # jamais agrandir >1
        disp_w = int(w * scale)
        disp_h = int(h * scale)
        self.scale = scale
        self.img_display = img.resize((disp_w, disp_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(self.img_display)

    def _reset_canvas_for_image(self):
        self.canvas.delete("all")
        self.rect_items.clear()
        self.text_items.clear()
        self.zones.clear()

        # Centrer l’image dans le canvas
        self.canvas.config(width=max(self.tk_img.width(), MAX_CANVAS_W), height=max(self.tk_img.height(), MAX_CANVAS_H))
        self.canvas.create_image(10, 10, anchor="nw", image=self.tk_img, tags=("image",))

        # Actualiser panneau zones
        self._refresh_zones_panel()

    def _update_info(self):
        if not self.img_pil_original:
            self.info_label.configure(text="Aucune image")
        else:
            name = os.path.basename(self.img_path)
            w, h = self.img_pil_original.size
            self.info_label.configure(text=f"{name}  |  {w}×{h}  |  échelle {self.scale:.3f}")

    # ----------------- Dessin -----------------
    def _on_press(self, event):
        if not self.img_pil_original:
            return
        self.drag_start = (event.x, event.y)
        # créer rect temporaire
        self.current_rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y,
                                                         outline="#1f6aa5", width=2, tags=("zone_temp",))

    def _on_drag(self, event):
        if not self.current_rect or not self.drag_start:
            return
        x0, y0 = self.drag_start
        x1, y1 = event.x, event.y
        self.canvas.coords(self.current_rect, x0, y0, x1, y1)

    def _on_release(self, event):
        if not self.current_rect or not self.drag_start:
            return
        x0, y0 = self.drag_start
        x1, y1 = event.x, event.y
        self.drag_start = None

        # Normaliser orientation
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))

        # Découper dans les bornes de l’image affichée
        disp_w, disp_h = self.img_display.size
        x0 = min(max(x0, 0), disp_w)
        y0 = min(max(y0, 0), disp_h)
        x1 = min(max(x1, 0), disp_w)
        y1 = min(max(y1, 0), disp_h)

        # Trop petit ? on ignore
        if (x1 - x0) < 5 or (y1 - y0) < 5:
            self.canvas.delete(self.current_rect)
            self.current_rect = None
            return

        # Convertir coords affichage -> pixels originaux
        s = self.scale if self.scale != 0 else 1.0
        ox0, oy0 = int(x0 / s), int(y0 / s)
        ox1, oy1 = int(x1 / s), int(y1 / s)

        W, H = self.img_pil_original.size
        nx0, ny0 = ox0 / W, oy0 / H
        nx1, ny1 = ox1 / W, oy1 / H

        zone_id = len(self.zones) + 1
        zone = {
            "id": zone_id,
            "pixel": {"x0": ox0, "y0": oy0, "x1": ox1, "y1": oy1, "w": ox1 - ox0, "h": oy1 - oy0},
            "norm":  {"x0": nx0, "y0": ny0, "x1": nx1, "y1": ny1, "w": nx1 - nx0, "h": ny1 - ny0},
        }
        self.zones.append(zone)

        # Finaliser le rectangle (changer le tag/couleur)
        self.canvas.itemconfig(self.current_rect, outline="#0ea5e9")
        self.canvas.addtag_withtag("zone_final", self.current_rect)
        self.rect_items.append(self.current_rect)

        # Petit label
        label_id = self.canvas.create_text(x0 + 6, y0 + 6, anchor="nw", text=str(zone_id), fill="#0ea5e9", font=("Segoe UI", 10, "bold"))
        self.text_items.append(label_id)

        self.current_rect = None
        self._refresh_zones_panel()
        self.status.configure(text=f"Zone #{zone_id} ajoutée")

    # ----------------- Actions -----------------
    def undo_zone(self):
        if not self.zones:
            return
        self.zones.pop()
        if self.rect_items:
            self.canvas.delete(self.rect_items.pop())
        if self.text_items:
            self.canvas.delete(self.text_items.pop())
        self._refresh_zones_panel()
        self.status.configure(text="Dernière zone supprimée")

    def clear_zones(self):
        if not self.zones:
            return
        if not messagebox.askyesno("Confirmation", "Effacer toutes les zones ?"):
            return
        self.zones.clear()
        for rid in self.rect_items: self.canvas.delete(rid)
        for tid in self.text_items: self.canvas.delete(tid)
        self.rect_items.clear()
        self.text_items.clear()
        self._refresh_zones_panel()
        self.status.configure(text="Toutes les zones ont été effacées")

    def save_zones(self):
        if not self.img_pil_original or not self.zones:
            messagebox.showinfo("Info", "Charge une image et crée au moins une zone.")
            return
        default_name = f"zones_{os.path.splitext(os.path.basename(self.img_path))[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_path = filedialog.asksaveasfilename(
            title="Enregistrer les zones",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("JSON", "*.json")]
        )
        if not out_path:
            return
        payload = {
            "image": {
                "path": self.img_path,
                "width": self.img_pil_original.width,
                "height": self.img_pil_original.height,
            },
            "zones": self.zones,
        }
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            self.status.configure(text=f"Zones sauvegardées → {out_path}")
            messagebox.showinfo("Succès", f"Enregistré :\n{out_path}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'écrire le fichier:\n{e}")

    def _refresh_zones_panel(self):
        self.zones_text.configure(state="normal")
        self.zones_text.delete("1.0", "end")
        if not self.zones:
            self.zones_text.insert("1.0", "Aucune zone.")
        else:
            lines = []
            for z in self.zones:
                p = z["pixel"]
                lines.append(f"#{z['id']}  x0={p['x0']} y0={p['y0']}  x1={p['x1']} y1={p['y1']}  w={p['w']} h={p['h']}")
            self.zones_text.insert("1.0", "\n".join(lines))
        self.zones_text.configure(state="disabled")
        self._update_info()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ZoneEditorCTK()
    app.run()
