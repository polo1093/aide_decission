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
import sys
from pathlib import Path

# Ajout racine du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
