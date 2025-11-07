"""
Outil graphique minimal pour positionner des zones de capture.
- Ouvre une image (idéalement l'image crop de la table)
- Permet de dessiner des rectangles (cliquer-glisser)
- Chaque rectangle a un nom et un type (ex: 'carte_numero', 'carte_symbole')
- Si "Appliquer même taille aux mêmes types" est coché, le redimensionnement d'une
  zone applique la même largeur/hauteur aux autres zones du même type en conservant
  le centre.
- Sauvegarde les coordonnées dans `config/coordinates.json` en écrasant/mergeant
  les régions existantes. Les coordonnées sauvées sont des `coord_rel` calculées
  par rapport au `reference_point` fourni par l'utilisateur (champs X,Y).

Usage bref:
python scripts/position_zones.py

Dépendances: PySimpleGUI, Pillow
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import PySimpleGUI as sg
from PIL import Image, ImageTk

# Chemin du fichier de configuration
ROOT = Path(__file__).resolve().parents[1]
COORDS_PATH = Path(ROOT) / "config" / "coordinates.json"

# Types utiles
Rect = Tuple[int, int, int, int]  # (x1,y1,x2,y2)


def load_coords(path: Path) -> Dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"regions": {}, "table_capture": {}}


def save_coords(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class ZoneEditor:
    def __init__(self):
        self.coords_data = load_coords(COORDS_PATH)
        self.regions = self.coords_data.get("regions", {})

        self.image = None
        self.photo = None
        self.image_size = (800, 600)

        # rectangles: list of dict {name,type,rect,elem_id}
        self.rects: List[Dict] = []
        self.current_drawing = None  # (x0,y0)
        self.selected_index = None

        sg.theme("LightBlue")
        layout = [
            [sg.Text("Image crop:"), sg.Input(key="-FILE-"), sg.FileBrowse(file_types=(("Images","*.png;*.jpg;*.jpeg"),)), sg.Button("Charger")],
            [sg.Text("Reference point (pour coord_rel) :"), sg.Text("X"), sg.Input("0", size=(6,1), key="-REFX-"), sg.Text("Y"), sg.Input("0", size=(6,1), key="-REFY-")],
            [
                sg.Column([
                    [sg.Graph(canvas_size=(800, 600), graph_bottom_left=(0, 600), graph_top_right=(800, 0), key="-GRAPH-", enable_events=True, drag_submits=True)],
                ]),
                sg.Column([
                    [sg.Text("Rectangles")],
                    [sg.Listbox(values=[], size=(30,20), key='-LIST-', enable_events=True)],
                    [sg.Text("Nom (clé):"), sg.Input(key='-NAME-')],
                    [sg.Text("Type:"), sg.Combo(['carte_numero','carte_symbole','nombre','texte'], default_value='carte_numero', key='-TYPE-')],
                    [sg.Checkbox('Appliquer même taille aux mêmes types', key='-LINK_SIZE-', default=True)],
                    [sg.Button('Supprimer'), sg.Button('Renommer')],
                    [sg.Button('Charger coords existantes'), sg.Button('Sauver dans coordinates.json')]
                ])
            ],
            [sg.Text('Instructions: Cliquer-glisser sur l image pour créer/ajuster une zone. Sélectionner une zone dans la liste pour la modifier.')]
        ]

        self.window = sg.Window('Position Zones', layout, finalize=True)
        self.graph: sg.Graph = self.window['-GRAPH-']

        # map of graph element ids for rects
        self.graph_rects = []

    def image_to_photo(self, pil_image: Image.Image, max_size=(800,600)):
        w,h = pil_image.size
        max_w, max_h = max_size
        scale = min(max_w/w, max_h/h, 1.0)
        new_size = (int(w*scale), int(h*scale))
        img_resized = pil_image.resize(new_size, Image.ANTIALIAS)
        self.image_size = (new_size[0], new_size[1])
        return ImageTk.PhotoImage(img_resized), scale

    def load_image(self, path: str):
        self.image = Image.open(path).convert('RGB')
        self.photo, self.scale = self.image_to_photo(self.image)
        w,h = self.image_size
        self.graph.change_coordinates((0, h), (w, 0))
        self.graph.draw_image(data=self.photo, location=(0,0))
        self.redraw_all()

    def redraw_all(self):
        # clear rectangles
        for rid in list(self.graph_rects):
            try:
                self.graph.delete_figure(rid)
            except Exception:
                pass
        self.graph_rects = []

        for r in self.rects:
            x1,y1,x2,y2 = r['rect']
            # apply scale
            x1s, y1s, x2s, y2s = [int(v * self.scale) for v in (x1,y1,x2,y2)]
            rid = self.graph.draw_rectangle((x1s,y1s),(x2s,y2s), line_color='red')
            tid = self.graph.draw_text(r['name'], (x1s+3, y1s+12), color='yellow')
            self.graph_rects.append(rid)
            self.graph_rects.append(tid)

        # update listbox
        self.window['-LIST-'].update([f"{i}: {r['name']} ({r['type']})" for i,r in enumerate(self.rects)])

    def point_scaled_to_image(self, x:int,y:int) -> Tuple[int,int]:
        # graph coords already map to image scaled size
        # but events give pixel coords as ints relative to graph coordinate system
        return int(x/self.scale), int(y/self.scale)

    def add_rect(self, rect: Rect, name: str, typ: str):
        self.rects.append({'name': name, 'type': typ, 'rect': rect})
        self.redraw_all()

    def update_rect_size_by_type(self, index:int):
        # when rect at index changed, if LINK_SIZE checked, apply w/h to other rects of same type
        if not self.window['-LINK_SIZE-'].get():
            return
        src = self.rects[index]
        sx1,sy1,sx2,sy2 = src['rect']
        sw = sx2 - sx1
        sh = sy2 - sy1
        stype = src['type']
        for i,r in enumerate(self.rects):
            if i==index: continue
            if r['type'] == stype:
                # keep center
                cx = (r['rect'][0] + r['rect'][2])//2
                cy = (r['rect'][1] + r['rect'][3])//2
                nx1 = cx - sw//2
                ny1 = cy - sh//2
                nx2 = nx1 + sw
                ny2 = ny1 + sh
                self.rects[i]['rect'] = [nx1,ny1,nx2,ny2]
        self.redraw_all()

    def run(self):
        dragging = False
        start = None
        current_temp = None

        while True:
            event, values = self.window.read(timeout=50)
            if event == sg.WIN_CLOSED:
                break

            if event == 'Charger':
                f = values['-FILE-']
                if f and Path(f).exists():
                    self.load_image(f)

            if event == '-GRAPH-':
                x,y = values['-GRAPH-']
                # mouse clicked
                dragging = True
                start = (x,y)
                # start temp rectangle
                if current_temp:
                    try: self.graph.delete_figure(current_temp)
                    except: pass
                current_temp = self.graph.draw_rectangle(start, start, line_color='blue')

            if event == '-GRAPH-' + '+UP':
                # mouse released
                x,y = values['-GRAPH-']
                if dragging and start:
                    x1,y1 = start
                    x2,y2 = x,y
                    # normalize
                    left,top = min(x1,x2), min(y1,y2)
                    right,bottom = max(x1,x2), max(y1,y2)
                    # convert to image coordinates
                    ix1,iy1 = int(left / self.scale), int(top / self.scale)
                    ix2,iy2 = int(right / self.scale), int(bottom / self.scale)
                    # prompt for name/type
                    name = values['-NAME-'] or sg.popup_get_text('Nom (clé) pour cette zone', default_text='zone_'+str(len(self.rects)))
                    typ = values['-TYPE-']
                    if not name:
                        name = 'zone_'+str(len(self.rects))
                    self.add_rect([ix1,iy1,ix2,iy2], name, typ)
                    # apply size to same type
                    self.update_rect_size_by_type(len(self.rects)-1)
                dragging=False
                start=None
                if current_temp:
                    try: self.graph.delete_figure(current_temp)
                    except: pass
                    current_temp = None

            if event == '-LIST-':
                sel = values['-LIST-']
                if sel:
                    text = sel[0]
                    idx = int(text.split(':',1)[0])
                    self.selected_index = idx

            if event == 'Supprimer':
                if self.selected_index is not None and 0 <= self.selected_index < len(self.rects):
                    del self.rects[self.selected_index]
                    self.selected_index = None
                    self.redraw_all()

            if event == 'Renommer':
                if self.selected_index is not None and 0 <= self.selected_index < len(self.rects):
                    newname = values['-NAME-'] or sg.popup_get_text('Nouveau nom', default_text=self.rects[self.selected_index]['name'])
                    if newname:
                        self.rects[self.selected_index]['name'] = newname
                        self.redraw_all()

            if event == 'Charger coords existantes':
                # load regions from config and overlay them (assume coords are relative to reference point given below)
                data = load_coords(COORDS_PATH)
                regs = data.get('regions', {})
                self.rects = []
                refx = int(values.get('-REFX-',0))
                refy = int(values.get('-REFY-',0))
                for name,info in regs.items():
                    coord_rel = info.get('coord_rel')
                    if coord_rel and len(coord_rel)==4:
                        # compute absolute in image by adding reference
                        ax1 = coord_rel[0] + refx
                        ay1 = coord_rel[1] + refy
                        ax2 = coord_rel[2] + refx
                        ay2 = coord_rel[3] + refy
                        self.rects.append({'name': name, 'type': info.get('type',''), 'rect':[ax1,ay1,ax2,ay2]})
                self.redraw_all()

            if event == 'Sauver dans coordinates.json':
                # Merge current rects into existing coords and save
                data = load_coords(COORDS_PATH)
                if 'regions' not in data:
                    data['regions'] = {}
                refx = int(values.get('-REFX-',0))
                refy = int(values.get('-REFY-',0))
                for r in self.rects:
                    x1,y1,x2,y2 = r['rect']
                    coord_rel = [int(x1 - refx), int(y1 - refy), int(x2 - refx), int(y2 - refy)]
                    data['regions'][r['name']] = {
                        'coord_rel': coord_rel,
                        'coord_abs': None,
                        'type': r['type'],
                        'value': None
                    }
                save_coords(COORDS_PATH, data)
                sg.popup('Sauvé dans', str(COORDS_PATH))

        self.window.close()


if __name__ == '__main__':
    editor = ZoneEditor()
    editor.run()
