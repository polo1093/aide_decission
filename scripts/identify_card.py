#!/usr/bin/env python3
"""identify_card.py — assistant de labellisation des cartes.

Le script parcourt les captures présentes dans ``config/<jeu>/debug/crops``
et extrait, pour chaque carte connue dans ``coordinates.json``, les
fragments "number" et "suit". Lorsque la reconnaissance via les gabarits
existants échoue, une petite interface ``customtkinter`` propose de
sélectionner manuellement la valeur et la couleur de la carte. Les images
labellisées sont ensuite enregistrées dans ``config/<jeu>/cards`` avec une
bordure réduite (3 pixels par défaut) afin d'alimenter la base de modèles.

Utilisation minimale :

```
python scripts/identify_card.py --game PMU
```

Options principales :

```
  --crops-dir    Dossier à parcourir (par défaut debug/crops du jeu).
  --threshold    Score minimal pour accepter une reconnaissance automatique.
  --force-all    Inclut toutes les cartes (même reconnues) dans l'interface.
  --trim         Bordure rognée lors de la sauvegarde (3 px par défaut).
```
"""

from __future__ import annotations

import argparse
import itertools
import sys
import tkinter as tk
import tkinter.messagebox as messagebox
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Assure l'accès aux modules du dépôt lorsque le script est lancé via un chemin absolu.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from capture_cards import (
    TemplateIndex,
    extract_region_images,
    is_card_present,
    load_coordinates,
    recognize_number_and_suit,
)


DEFAULT_NUMBERS: Sequence[str] = (
    "?",
    "A",
    "K",
    "Q",
    "J",
    "10",
    "9",
    "8",
    "7",
    "6",
    "5",
    "4",
    "3",
    "2",
)
DEFAULT_SUITS: Sequence[str] = ("?", "spades", "hearts", "diamonds", "clubs")


@dataclass
class CardSample:
    """Extrait d'une carte issu d'une capture de table."""

    source_path: Path
    base_key: str
    number_patch: Image.Image
    suit_patch: Image.Image
    number_suggestion: Optional[str]
    suit_suggestion: Optional[str]
    number_score: float
    suit_score: float

    def trimmed_number(self, border: int) -> Image.Image:
        return _trim_patch(self.number_patch, border)

    def trimmed_suit(self, border: int) -> Image.Image:
        return _trim_patch(self.suit_patch, border)


def _trim_patch(img: Image.Image, border: int) -> Image.Image:
    if border <= 0:
        return img
    w, h = img.size
    if w <= border * 2 or h <= border * 2:
        return img
    return img.crop((border, border, w - border, h - border))


def _iter_capture_files(directory: Path) -> Iterable[Path]:
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        yield from sorted(directory.glob(ext))


def _unique_sorted(values: Iterable[str], defaults: Sequence[str]) -> List[str]:
    seen = list(defaults)
    for val in values:
        if val and val not in seen:
            seen.append(val)
    return seen


def collect_card_samples(
    table_paths: Iterable[Path],
    regions: Dict[str, object],
    idx: TemplateIndex,
    *,
    trim_border: int,
    accept_threshold: float,
    force_all: bool,
) -> Tuple[List[CardSample], int, int]:
    """Retourne (samples, total_cartes, reconnues_auto)."""

    samples: List[CardSample] = []
    total_cards = 0
    auto_ok = 0
    for img_path in table_paths:
        try:
            with Image.open(img_path) as im:
                table_img = im.convert("RGB")
        except FileNotFoundError:
            continue
        card_pairs = extract_region_images(table_img, regions, pad=0)
        for base_key, (num_patch, suit_patch) in card_pairs.items():
            if not is_card_present(num_patch, threshold=215, min_ratio=0.04):
                # carte absente : on ignore
                continue
            total_cards += 1
            trimmed_num = _trim_patch(num_patch, trim_border)
            trimmed_suit = _trim_patch(suit_patch, trim_border)
            suggestion_num, suggestion_suit, score_num, score_suit = recognize_number_and_suit(
                trimmed_num, trimmed_suit, idx
            )
            recognized = (
                not force_all
                and suggestion_num
                and suggestion_suit
                and score_num >= accept_threshold
                and score_suit >= accept_threshold
            )
            if recognized:
                auto_ok += 1
                continue
            samples.append(
                CardSample(
                    source_path=img_path,
                    base_key=base_key,
                    number_patch=num_patch,
                    suit_patch=suit_patch,
                    number_suggestion=suggestion_num,
                    suit_suggestion=suggestion_suit,
                    number_score=score_num,
                    suit_score=score_suit,
                )
            )
    return samples, total_cards, auto_ok


class DatasetWriter:
    def __init__(self, idx: TemplateIndex, cards_root: Path, trim_border: int) -> None:
        self.idx = idx
        self.cards_root = Path(cards_root)
        self.cards_root.mkdir(parents=True, exist_ok=True)
        (self.cards_root / "numbers").mkdir(parents=True, exist_ok=True)
        (self.cards_root / "suits").mkdir(parents=True, exist_ok=True)
        self.trim_border = trim_border
        self.counter = itertools.count(1)
        self.saved: List[Tuple[Path, Path]] = []

    def save(self, sample: CardSample, number_label: str, suit_label: str) -> Tuple[Path, Path]:
        n_img = sample.trimmed_number(self.trim_border)
        s_img = sample.trimmed_suit(self.trim_border)
        idx = next(self.counter)
        base_name = f"{sample.base_key}_{sample.source_path.stem}_{idx:04d}"
        num_path = self.cards_root / "numbers" / number_label / f"{base_name}.png"
        suit_path = self.cards_root / "suits" / suit_label / f"{base_name}.png"
        num_path.parent.mkdir(parents=True, exist_ok=True)
        suit_path.parent.mkdir(parents=True, exist_ok=True)
        n_img.save(num_path)
        s_img.save(suit_path)
        self._update_index(number_label, n_img, is_number=True)
        self._update_index(suit_label, s_img, is_number=False)
        self.saved.append((num_path, suit_path))
        return num_path, suit_path

    def _update_index(self, label: str, img: Image.Image, *, is_number: bool) -> None:
        gray = np.array(img.convert("L"))
        arr = TemplateIndex._prep(gray)
        store = self.idx.numbers if is_number else self.idx.suits
        store.setdefault(label, []).append(arr)


class LabelingApp:
    def __init__(
        self,
        samples: Sequence[CardSample],
        writer: DatasetWriter,
        *,
        number_choices: Sequence[str],
        suit_choices: Sequence[str],
    ) -> None:
        self.samples = list(samples)
        self.writer = writer
        self.index = 0
        self.number_choices = list(number_choices)
        self.suit_choices = list(suit_choices)
        self.photo: Optional[ImageTk.PhotoImage] = None

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Labellisation des cartes")
        self.root.geometry("720x540")

        self.number_var = tk.StringVar(value=self.number_choices[0] if self.number_choices else "")
        self.suit_var = tk.StringVar(value=self.suit_choices[0] if self.suit_choices else "")

        self._build_ui()
        self._show_current()

    def _build_ui(self) -> None:
        top = ctk.CTkFrame(self.root)
        top.pack(fill="x", padx=12, pady=8)

        self.status_label = ctk.CTkLabel(top, text="")
        self.status_label.pack(side="left", padx=6)

        self.suggestion_label = ctk.CTkLabel(top, text="", anchor="e")
        self.suggestion_label.pack(side="right", padx=6)

        body = ctk.CTkFrame(self.root)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.image_label = ctk.CTkLabel(body, text="", compound="top")
        self.image_label.pack(side="top", pady=12)

        form = ctk.CTkFrame(body)
        form.pack(side="top", pady=8)

        ctk.CTkLabel(form, text="Valeur").grid(row=0, column=0, padx=8, pady=6)
        self.number_menu = ctk.CTkOptionMenu(
            form,
            values=self.number_choices if self.number_choices else ["?"],
            variable=self.number_var,
        )
        self.number_menu.grid(row=0, column=1, padx=8, pady=6)

        ctk.CTkLabel(form, text="Couleur").grid(row=1, column=0, padx=8, pady=6)
        self.suit_menu = ctk.CTkOptionMenu(
            form,
            values=self.suit_choices if self.suit_choices else ["?"],
            variable=self.suit_var,
        )
        self.suit_menu.grid(row=1, column=1, padx=8, pady=6)

        btns = ctk.CTkFrame(body)
        btns.pack(side="bottom", pady=16)

        self.save_btn = ctk.CTkButton(btns, text="Enregistrer", command=self._on_save)
        self.save_btn.pack(side="left", padx=10)

        self.skip_btn = ctk.CTkButton(btns, text="Ignorer", command=self._on_skip)
        self.skip_btn.pack(side="left", padx=10)

    def run(self) -> None:
        self.root.mainloop()

    def _on_save(self) -> None:
        if self.index >= len(self.samples):
            self.root.destroy()
            return
        number_label = self.number_var.get().strip()
        suit_label = self.suit_var.get().strip()
        if number_label in {"", "?"} or suit_label in {"", "?"}:
            messagebox.showwarning("Label manquant", "Sélectionnez la valeur et la couleur de la carte.")
            return
        sample = self.samples[self.index]
        self.writer.save(sample, number_label, suit_label)
        self.index += 1
        self._show_current()

    def _on_skip(self) -> None:
        if self.index >= len(self.samples):
            self.root.destroy()
            return
        self.index += 1
        self._show_current()

    def _show_current(self) -> None:
        if self.index >= len(self.samples):
            self.status_label.configure(text="Labellisation terminée")
            self.suggestion_label.configure(text=f"{len(self.writer.saved)} cartes sauvegardées")
            self.image_label.configure(image=None, text="")
            self.save_btn.configure(text="Fermer", command=self.root.destroy)
            self.skip_btn.configure(text="Fermer", command=self.root.destroy)
            return
        sample = self.samples[self.index]
        self.status_label.configure(
            text=f"Carte {self.index + 1}/{len(self.samples)} — {sample.base_key} ({sample.source_path.name})"
        )
        self._update_menu(self.number_menu, self.number_choices, sample.number_suggestion, self.number_var)
        self._update_menu(self.suit_menu, self.suit_choices, sample.suit_suggestion, self.suit_var)
        suggestion_text = _format_suggestion(sample.number_suggestion, sample.suit_suggestion, sample.number_score, sample.suit_score)
        self.suggestion_label.configure(text=suggestion_text)

        preview = _make_preview(sample.trimmed_number(self.writer.trim_border), sample.trimmed_suit(self.writer.trim_border))
        self.photo = ImageTk.PhotoImage(preview)
        self.image_label.configure(image=self.photo)

    @staticmethod
    def _update_menu(menu: ctk.CTkOptionMenu, choices: Sequence[str], suggestion: Optional[str], var: tk.StringVar) -> None:
        values = list(choices) if choices else ["?"]
        menu.configure(values=values)
        if suggestion and suggestion in values:
            var.set(suggestion)
        else:
            var.set(values[0])


def _format_suggestion(
    num: Optional[str], suit: Optional[str], score_num: float, score_suit: float
) -> str:
    if not num and not suit:
        return ""
    parts = []
    if num:
        parts.append(f"{num} ({score_num:.2f})")
    if suit:
        parts.append(f"{suit} ({score_suit:.2f})")
    return " / ".join(parts)


def _make_preview(num_img: Image.Image, suit_img: Image.Image) -> Image.Image:
    num_w, num_h = num_img.size
    suit_w, suit_h = suit_img.size
    width = max(num_w, suit_w)
    spacer = 6
    preview = Image.new("RGB", (width, num_h + suit_h + spacer), color="#f0f0f0")
    preview.paste(num_img, ((width - num_w) // 2, 0))
    preview.paste(suit_img, ((width - suit_w) // 2, num_h + spacer))
    return preview


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interface de labellisation des cartes")
    parser.add_argument("--game", default="PMU", help="Identifiant du jeu (dossier dans config/)")
    parser.add_argument("--crops-dir", help="Dossier contenant les captures à analyser")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.92,
        help="Score minimal (0-1) pour accepter une reconnaissance automatique",
    )
    parser.add_argument("--trim", type=int, default=3, help="Bordure supprimée avant sauvegarde")
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Inclut toutes les cartes dans l'interface, même celles reconnues",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    game_dir = Path("config") / args.game
    coords_path = game_dir / "coordinates.json"
    if not coords_path.exists():
        print(f"ERREUR : fichier de coordonnées introuvable ({coords_path})")
        return 2

    crops_dir = Path(args.crops_dir) if args.crops_dir else game_dir / "debug" / "crops"
    if not crops_dir.exists():
        print(f"ERREUR : dossier de captures introuvable ({crops_dir})")
        return 2

    regions, _, _ = load_coordinates(coords_path)
    cards_root = game_dir / "cards"
    idx = TemplateIndex(cards_root)
    idx.load()

    crop_paths = list(_iter_capture_files(crops_dir))
    if not crop_paths:
        print(f"Aucune capture trouvée dans {crops_dir}")
        return 0

    samples, total_cards, auto_ok = collect_card_samples(
        crop_paths,
        regions,
        idx,
        trim_border=args.trim,
        accept_threshold=args.threshold,
        force_all=args.force_all,
    )

    print(
        f"Cartes analysées : {total_cards} — reconnues automatiquement : {auto_ok} — à labelliser : {len(samples)}"
    )
    if not samples:
        print("Rien à labelliser, base de gabarits à jour !")
        return 0

    number_choices = _unique_sorted(
        [s.number_suggestion for s in samples if s.number_suggestion], DEFAULT_NUMBERS
    )
    suit_choices = _unique_sorted(
        [s.suit_suggestion for s in samples if s.suit_suggestion], DEFAULT_SUITS
    )

    writer = DatasetWriter(idx, cards_root, args.trim)
    app = LabelingApp(
        samples,
        writer,
        number_choices=number_choices,
        suit_choices=suit_choices,
    )
    app.run()

    if writer.saved:
        print(f"{len(writer.saved)} cartes enregistrées dans {cards_root}")
    else:
        print("Aucune nouvelle carte enregistrée.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

