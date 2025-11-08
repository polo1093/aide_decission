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
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional, Tuple, List

import customtkinter as ctk
from PIL import ImageTk, Image

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
        self.project = ZoneProject()
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
