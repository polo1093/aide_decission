# zone_project.py
# Logique "métier" : modèle, opérations, lecture/écriture JSON, clamp
# Aucune dépendance UI. Dépend de pillow uniquement pour charger l'image.

from __future__ import annotations
import os, json
from typing import Dict, Any, Optional, Tuple, List
from collections import OrderedDict
from PIL import Image

def coerce_int(v: Any, default: int = 0) -> int:
    try:
        return int(round(float(v)))
    except Exception:
        return default

def clamp_top_left(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int]:
    if W <= 0 or H <= 0:
        return x, y
    x = max(0, min(x, max(0, W - w)))
    y = max(0, min(y, max(0, H - h)))
    return x, y

def _resolve_templates(templates: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Résout alias_of → {group: {size:[w,h], type:str, layout:dict}}
    layout peut contenir: { "lock_same_y": bool }
    """
    resolved: Dict[str, Dict[str, Any]] = {}

    def get_size(g: str, seen=None) -> Tuple[int, int]:
        if seen is None:
            seen = set()
        if g in seen:
            return 0, 0
        seen.add(g)
        t = templates.get(g, {})
        if "size" in t:
            w, h = t.get("size", [0, 0])
            return coerce_int(w), coerce_int(h)
        if "alias_of" in t:
            return get_size(str(t["alias_of"]), seen)
        return 0, 0

    def get_type(g: str, seen=None) -> str:
        if seen is None:
            seen = set()
        if g in seen:
            return ""
        seen.add(g)
        t = templates.get(g, {})
        if "type" in t and t["type"]:
            return str(t["type"])
        if "alias_of" in t:
            return get_type(str(t["alias_of"]), seen)
        return ""

    def get_layout(g: str, seen=None) -> Dict[str, Any]:
        if seen is None:
            seen = set()
        if g in seen:
            return {}
        seen.add(g)
        t = templates.get(g, {})
        if "layout" in t and isinstance(t["layout"], dict):
            return dict(t["layout"])
        if "alias_of" in t:
            return get_layout(str(t["alias_of"]), seen)
        return {}

    for g in list(templates.keys()):
        w, h = get_size(g)
        typ = get_type(g)
        layout = get_layout(g)
        resolved[g] = {"size": [w, h], "type": typ, "layout": layout}
    return resolved

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

    def __init__(self) -> None:
        self.base_dir: str = ""
        self.current_game: Optional[str] = None
        self.image_path: Optional[str] = None
        self.image: Optional[Image.Image] = None
        self.table_capture: Dict[str, Any] = {"enabled": True, "relative_bounds": [0, 0, 0, 0]}

        # Données décrites par le JSON
        self.templates: Dict[str, Any] = {}
        self._templates_resolved: Dict[str, Any] = {}
        # regions : key -> {"group": str, "top_left":[x,y], "value": Any, "label": str}
        self.regions: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    # ---------- Propriétés utiles ----------
    @property
    def image_size(self) -> Tuple[int, int]:
        if self.image is None:
            return 0, 0
        return self.image.width, self.image.height

    @property
    def templates_resolved(self) -> Dict[str, Any]:
        # recalcul léger à la demande (ou fais-le sur set)
        return _resolve_templates(self.templates)

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
        self.table_capture = {"enabled": True, "relative_bounds": [0, 0, W, H]}
        self.templates = {}
        self.regions.clear()

        data = _load_templated_json(coord_path) if os.path.isfile(coord_path) else None
        if data:
            self.table_capture = data.get("table_capture", self.table_capture)
            self.templates = data.get("templates", {})
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
            self.templates = {"action_button": {"size": [165, 70], "type": "texte"}}

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
