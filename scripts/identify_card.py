#!/usr/bin/env python3
"""identify_card.py — assistant de labellisation des cartes.

Parcourt les crops de tables dans ``config/<jeu>/debug/crops``, extrait les
patches *number* et *suit* selon `coordinates.json`, tente une reco par gabarits,
**auto-skip** des cartes déjà connues (log) avec un seuil strict, et **ne demande
que la partie inconnue** (valeur OU couleur) lorsque l’autre est déjà fiable.

Usage minimal:
    python scripts/identify_card.py --game PMU

Options utiles:
  --crops-dir     Dossier d’entrée (défaut: config/<game>/debug/crops)
  --threshold     Score min (0-1) pour considérer une reco comme fiable (def 0.92)
  --strict        Score min (0-1) pour auto-skip sans UI (def 0.985)
  --force-all     Forcer l’UI même si auto-skip serait possible
  --trim          Bordure rognée (px) autour des patches AVANT reco & sauvegarde (def 6)
"""

import argparse
import itertools
import sys
import tkinter as tk
import tkinter.messagebox as messagebox
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Accès modules du dépôt
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import customtkinter as ctk
from PIL import Image, ImageTk

from objet.scanner.cards_recognition import (
    ROOT_TEMPLATE_SET,
    TemplateIndex,
    is_card_present,
    recognize_number_and_suit,
)
from _utils import CardPatch, collect_card_patches, load_coordinates

DEFAULT_NUMBERS: Sequence[str] = (
    "?",
    "A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2",
)
DEFAULT_SUITS: Sequence[str] = ("?", "spades", "hearts", "diamonds", "clubs")


@dataclass
class CardSample:
    source_path: Path
    base_key: str
    number_patch: Image.Image
    suit_patch: Image.Image
    template_set: Optional[str]
    number_suggestion: Optional[str]
    suit_suggestion: Optional[str]
    number_score: float
    suit_score: float
    num_known: bool  # True si number est fiable (>= threshold)
    suit_known: bool # True si suit   est fiable (>= threshold)

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
    # rogne de "border" px sur TOUTES les bordures
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
    strict_threshold: float,
    force_all: bool,
) -> Tuple[List[CardSample], int, int]:
    """Retourne (samples, total_cartes, reconnues_auto).

    - Calcul des scores via `recognize_number_and_suit` sur **patches rognés** (trim_border).
    - Si score >= strict_threshold pour number/suit, on considère la partie **connue** sans UI.
    - Si les 2 parties sont connues et `force_all` False → auto-skip + log.
    - Sinon on ajoute un sample et l’UI ne demandera **que la partie inconnue**.
    """
    samples: List[CardSample] = []
    total_cards = 0
    auto_ok = 0

    for img_path in table_paths:
        try:
            with Image.open(img_path) as im:
                table_img = im.convert("RGB")
        except FileNotFoundError:
            continue

        card_pairs = collect_card_patches(table_img, regions, pad=0)
        for base_key, card_patch in card_pairs.items():
            num_patch = card_patch.number
            suit_patch = card_patch.suit
            tpl_set = card_patch.template_set
            if not is_card_present(num_patch, threshold=215, min_ratio=0.04):
                continue
            total_cards += 1

            # rognage AVANT reco
            trimmed_num = _trim_patch(num_patch, trim_border)
            trimmed_suit = _trim_patch(suit_patch, trim_border)

            suggestion_num, suggestion_suit, score_num, score_suit = recognize_number_and_suit(
                trimmed_num,
                trimmed_suit,
                idx,
                template_set=tpl_set,
            )

            # deux niveaux de confiance : acceptable vs strict autoskip
            num_known = bool(suggestion_num) and float(score_num) >= float(accept_threshold)
            suit_known = bool(suggestion_suit) and float(score_suit) >= float(accept_threshold)
            num_strict = bool(suggestion_num) and float(score_num) >= float(strict_threshold)
            suit_strict = bool(suggestion_suit) and float(score_suit) >= float(strict_threshold)

            # logs informatifs
            if num_strict:
                print(f"DISCOVERED number={suggestion_num} ({score_num:.3f}) in {img_path.name} {base_key} → autoskip nombre")
            if suit_strict:
                print(f"DISCOVERED suit={suggestion_suit} ({score_suit:.3f}) in {img_path.name} {base_key} → autoskip couleur")

            if num_strict and suit_strict and not force_all:
                auto_ok += 1
                print(
                    f"AUTO OK: {img_path.name} {base_key} → number={suggestion_num} ({score_num:.2f}), "
                    f"suit={suggestion_suit} ({score_suit:.2f})"
                )
                continue

            # On garde le sample si au moins une partie n’atteint pas le strict
            samples.append(
                CardSample(
                    source_path=img_path,
                    base_key=base_key,
                    number_patch=num_patch,
                    suit_patch=suit_patch,
                    template_set=tpl_set,
                    number_suggestion=suggestion_num,
                    suit_suggestion=suggestion_suit,
                    number_score=float(score_num),
                    suit_score=float(score_suit),
                    num_known=num_strict,   # connu = seuil strict
                    suit_known=suit_strict,
                )
            )

    return samples, total_cards, auto_ok


class DatasetWriter:
    def __init__(self, idx: TemplateIndex, cards_root: Path, trim_border: int) -> None:
        self.idx = idx
        self.cards_root = Path(cards_root)
        self.cards_root.mkdir(parents=True, exist_ok=True)
        self.trim_border = trim_border
        self.counter = itertools.count(1)
        self.saved: List[Tuple[Path, Path]] = []

    def _normalise_template_set(self, template_set: Optional[str]) -> Optional[str]:
        if template_set:
            return template_set
        default = self.idx.default_set
        if default == ROOT_TEMPLATE_SET:
            return None
        return default

    def _set_root(self, template_set: Optional[str]) -> Path:
        if not template_set:
            return self.cards_root
        return self.cards_root / template_set

    def save(
        self,
        sample: CardSample,
        number_label: str,
        suit_label: str,
        *,
        save_number: bool = True,
        save_suit: bool = True,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """Enregistre les patches rognés; possibilité de ne sauver que l’inconnu."""
        n_img = sample.trimmed_number(self.trim_border)
        s_img = sample.trimmed_suit(self.trim_border)
        idx = next(self.counter)
        base_name = f"{sample.base_key}_{sample.source_path.stem}_{idx:04d}"
        resolved_set = self._normalise_template_set(sample.template_set)
        set_root = self._set_root(resolved_set)
        num_path = set_root / "numbers" / number_label / f"{base_name}.png"
        suit_path = set_root / "suits" / suit_label / f"{base_name}.png"
        np_out: Optional[Path] = None
        sp_out: Optional[Path] = None
        if save_number:
            num_path.parent.mkdir(parents=True, exist_ok=True)
            n_img.save(num_path)
            self.idx.append_template(resolved_set, number_label, n_img, is_number=True)
            np_out = num_path
        if save_suit:
            suit_path.parent.mkdir(parents=True, exist_ok=True)
            s_img.save(suit_path)
            self.idx.append_template(resolved_set, suit_label, s_img, is_number=False)
            sp_out = suit_path
        self.saved.append((np_out or Path(), sp_out or Path()))
        return np_out, sp_out


class LabelingApp:
    def __init__(
        self,
        samples: Sequence[CardSample],
        writer: DatasetWriter,
        *,
        number_choices: Sequence[str],
        suit_choices: Sequence[str],
        accept_threshold: float,
    ) -> None:
        self.samples = list(samples)
        self.writer = writer
        self.index = 0
        self.number_choices = list(number_choices)
        self.suit_choices = list(suit_choices)
        self.accept_threshold = float(accept_threshold)
        self.photo: Optional[ImageTk.PhotoImage] = None

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Labellisation des cartes")
        self.root.geometry("720x560")

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

        # Lignes (on les cache/affiche dynamiquement)
        self.lbl_number = ctk.CTkLabel(form, text="Valeur")
        self.lbl_number.grid(row=0, column=0, padx=8, pady=6)
        self.number_menu = ctk.CTkOptionMenu(
            form,
            values=self.number_choices if self.number_choices else ["?"],
            variable=self.number_var,
        )
        self.number_menu.grid(row=0, column=1, padx=8, pady=6)

        self.lbl_suit = ctk.CTkLabel(form, text="Couleur")
        self.lbl_suit.grid(row=1, column=0, padx=8, pady=6)
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
        sample = self.samples[self.index]

        # fige/complète avec les suggestions quand c’est "connu"
        number_label = sample.number_suggestion if sample.num_known else self.number_var.get().strip()
        suit_label = sample.suit_suggestion if sample.suit_known else self.suit_var.get().strip()

        if (not number_label or number_label == "?") and (not sample.num_known):
            messagebox.showwarning("Label manquant", "Sélectionnez la valeur de la carte.")
            return
        if (not suit_label or suit_label == "?") and (not sample.suit_known):
            messagebox.showwarning("Label manquant", "Sélectionnez la couleur de la carte.")
            return

        # Sauve uniquement la partie inconnue
        save_number = not sample.num_known
        save_suit = not sample.suit_known
        np_out, sp_out = self.writer.save(
            sample, number_label, suit_label, save_number=save_number, save_suit=save_suit
        )

        if sample.num_known and sample.suit_known:
            print(f"[SAVE] (déjà connus) {sample.base_key} → rien à sauver")
        else:
            print(
                f"[SAVE] {sample.base_key} → number={number_label if save_number else '—'}; "
                f"suit={suit_label if save_suit else '—'}"
            )

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
            self.suggestion_label.configure(text=f"{len(self.writer.saved)} entrées ajoutées")
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
        suggestion_text = _format_suggestion(
            sample.number_suggestion, sample.suit_suggestion, sample.number_score, sample.suit_score
        )
        self.suggestion_label.configure(text=suggestion_text)

        # Affiche UNIQUEMENT ce qui est à renseigner
        if sample.num_known:
            self.lbl_number.grid_remove(); self.number_menu.grid_remove()
        else:
            self.lbl_number.grid(row=0, column=0, padx=8, pady=6)
            self.number_menu.grid(row=0, column=1, padx=8, pady=6)
        if sample.suit_known:
            self.lbl_suit.grid_remove(); self.suit_menu.grid_remove()
        else:
            self.lbl_suit.grid(row=1, column=0, padx=8, pady=6)
            self.suit_menu.grid(row=1, column=1, padx=8, pady=6)

        preview = _make_preview(
            sample.trimmed_number(self.writer.trim_border), sample.trimmed_suit(self.writer.trim_border)
        )
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


def _format_suggestion(num: Optional[str], suit: Optional[str], score_num: float, score_suit: float) -> str:
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
    parser.add_argument("--threshold", type=float, default=0.92, help="Score min (0-1) pour accepter une reco auto")
    parser.add_argument("--strict", type=float, default=0.985, help="Score min (0-1) pour autoskip sans UI")
    parser.add_argument("--trim", type=int, default=6, help="Bordure rognée avant sauvegarde (px)")
    parser.add_argument("--force-all", action="store_true", help="Inclut toutes les cartes même reconnues")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    game_dir = Path("config") / args.game
    coords_path = game_dir / "coordinates.json"
    if not coords_path.exists():
        print(f"ERREUR: fichier de coordonnées introuvable ({coords_path})")
        return 2

    crops_dir = Path(args.crops_dir) if args.crops_dir else game_dir / "debug" / "crops"
    if not crops_dir.exists():
        print(f"ERREUR: dossier de captures introuvable ({crops_dir})")
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
        trim_border=int(args.trim),
        accept_threshold=float(args.threshold),
        strict_threshold=float(args.strict),
        force_all=bool(args.force_all),
    )

    print(
        f"Cartes analysées: {total_cards} — auto reconnues (strict): {auto_ok} — à labelliser: {len(samples)}"
    )
    if not samples:
        print("Rien à labelliser, base à jour ✅")
        return 0

    number_choices = _unique_sorted(
        [s.number_suggestion for s in samples if s.number_suggestion], DEFAULT_NUMBERS
    )
    suit_choices = _unique_sorted(
        [s.suit_suggestion for s in samples if s.suit_suggestion], DEFAULT_SUITS
    )

    writer = DatasetWriter(idx, cards_root, int(args.trim))
    app = LabelingApp(
        samples,
        writer,
        number_choices=number_choices,
        suit_choices=suit_choices,
        accept_threshold=float(args.threshold),
    )
    app.run()

    if writer.saved:
        print(f"{len(writer.saved)} entrées ajoutées dans {cards_root}")
    else:
        print("Aucune nouvelle entrée enregistrée.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Sequence, Dict
import itertools
import time

from PIL import Image, ImageTk
import customtkinter as ctk
import tkinter as tk

# Dépend de votre module existant
# capture_cards: TemplateIndex, recognize_number_and_suit, collect_card_patches, is_card_present
from objet.scanner.cards_recognition import (
    ROOT_TEMPLATE_SET,
    TemplateIndex,
    recognize_number_and_suit,
    is_card_present,
)
from objet.utils.calibration import CardPatch, collect_card_patches

DEFAULT_NUMBERS: Sequence[str] = (
    "?",
    "A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2",
)
DEFAULT_SUITS: Sequence[str] = ("?", "spades", "hearts", "diamonds", "clubs")


def _trim(img: Image.Image, border: int) -> Image.Image:
    if border <= 0:
        return img
    w, h = img.size
    if w <= border * 2 or h <= border * 2:
        return img
    return img.crop((border, border, w - border, h - border))


@dataclass
class IdentifyResult:
    number: str
    suit: str
    meta: Dict[str, object]


class _SingleCardDialog:
    """Boîte de dialogue minimale pour compléter **uniquement** la partie inconnue.
    - Si l'une des deux (valeur/couleur) est déjà fiable, on ne montre que l'autre champ.
    - Retourne (number, suit) choisis par l'utilisateur.
    """

    def __init__(
        self,
        number_img: Image.Image,
        suit_img: Image.Image,
        *,
        missing_number: bool,
        missing_suit: bool,
        number_choices: Sequence[str],
        suit_choices: Sequence[str],
        suggested_number: Optional[str] = None,
        suggested_suit: Optional[str] = None,
    ) -> None:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Identifier la carte")
        self.root.geometry("560x480")

        self.result: Optional[Tuple[str, str]] = None

        self.number_var = tk.StringVar(value=(suggested_number or number_choices[0] if number_choices else "?"))
        self.suit_var = tk.StringVar(value=(suggested_suit or suit_choices[0] if suit_choices else "?"))

        top = ctk.CTkFrame(self.root)
        top.pack(fill="both", expand=True, padx=12, pady=12)

        # Aperçu empilé
        prev = self._make_preview(number_img, suit_img)
        self.photo = ImageTk.PhotoImage(prev)
        self.img_lbl = ctk.CTkLabel(top, image=self.photo, text="", compound="top")
        self.img_lbl.pack(pady=8)

        form = ctk.CTkFrame(top)
        form.pack(pady=8)

        row = 0
        if missing_number:
            ctk.CTkLabel(form, text="Valeur").grid(row=row, column=0, padx=8, pady=6)
            self.num_menu = ctk.CTkOptionMenu(form, values=list(number_choices) or ["?"], variable=self.number_var)
            self.num_menu.grid(row=row, column=1, padx=8, pady=6)
            row += 1
        if missing_suit:
            ctk.CTkLabel(form, text="Couleur").grid(row=row, column=0, padx=8, pady=6)
            self.suit_menu = ctk.CTkOptionMenu(form, values=list(suit_choices) or ["?"], variable=self.suit_var)
            self.suit_menu.grid(row=row, column=1, padx=8, pady=6)
            row += 1

        btns = ctk.CTkFrame(top)
        btns.pack(pady=10)
        ctk.CTkButton(btns, text="Valider", command=self._on_save).pack(side="left", padx=8)
        ctk.CTkButton(btns, text="Annuler", command=self._on_cancel).pack(side="left", padx=8)

    def run(self) -> Optional[Tuple[str, str]]:
        self.root.mainloop()
        return self.result

    def _on_save(self) -> None:
        self.result = (self.number_var.get().strip() or "?", self.suit_var.get().strip() or "?")
        self.root.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.root.destroy()

    @staticmethod
    def _make_preview(num_img: Image.Image, suit_img: Image.Image) -> Image.Image:
        num_w, num_h = num_img.size
        suit_w, suit_h = suit_img.size
        w = max(num_w, suit_w)
        spacer = 6
        canvas = Image.new("RGB", (w, num_h + suit_h + spacer), "#f0f0f0")
        canvas.paste(num_img, ((w - num_w) // 2, 0))
        canvas.paste(suit_img, ((w - suit_w) // 2, num_h + spacer))
        return canvas


class CardIdentifier:
    """Service réutilisable : identifier (valeur, couleur) à partir de 2 patches.

    - Utilise l'index `cards/` du jeu (TemplateIndex).
    - Tente la reco. Si >= strict → renvoie sans UI.
    - Sinon, si `interactive=True`, ouvre une mini-UI qui ne demande que la partie
      inconnue, **sauvegarde** cette partie et met à jour l'index.
    - Retourne toujours (number, suit, meta).
    """

    def __init__(
        self,
        game_dir: Path | str,
        *,
        trim: int = 6,
        threshold: float = 0.92,
        strict: float = 0.8,
        number_choices: Sequence[str] = DEFAULT_NUMBERS,
        suit_choices: Sequence[str] = DEFAULT_SUITS,
    ) -> None:
        self.game_dir = Path(game_dir)
        self.cards_root = self.game_dir / "cards"
        self.trim = int(trim)
        self.threshold = float(threshold)
        self.strict = float(strict)
        self.number_choices = list(number_choices)
        self.suit_choices = list(suit_choices)
        self.idx = TemplateIndex(self.cards_root)
        self.idx.load()
        self._counter = itertools.count(1)
        self._last_template_set: Optional[str] = None

    def _normalise_template_set(self, template_set: Optional[str]) -> Optional[str]:
        if template_set:
            return template_set
        if self._last_template_set:
            return self._last_template_set
        default = self.idx.default_set
        return None if default == ROOT_TEMPLATE_SET else default

    # ---------- API principale (patches) ----------
    def identify_from_patches(
        self,
        number_patch: Image.Image,
        suit_patch: Image.Image,
        *,
        base_key: str = "live",
        template_set: Optional[str] = None,
        interactive: bool = True,
        force_all: bool = False,
    ) -> IdentifyResult:
        # 1) Trim puis tentative de reco
        tnum = _trim(number_patch, self.trim)
        tsuit = _trim(suit_patch, self.trim)
        tpl_set = template_set or self._normalise_template_set(None)
        num_s, suit_s, s_num, s_suit = recognize_number_and_suit(
            tnum,
            tsuit,
            self.idx,
            template_set=tpl_set,
        )

        num_known_strict = bool(num_s) and float(s_num) >= self.strict
        suit_known_strict = bool(suit_s) and float(s_suit) >= self.strict

        if num_known_strict and suit_known_strict and not force_all:
            return IdentifyResult(num_s, suit_s, {
                "source": "auto",
                "score_number": float(s_num),
                "score_suit": float(s_suit),
            })

        # 2) Si interactif: ne demander que la partie inconnue
        if interactive:
            missing_number = not num_known_strict
            missing_suit = not suit_known_strict
            dialog = _SingleCardDialog(
                tnum,
                tsuit,
                missing_number=missing_number,
                missing_suit=missing_suit,
                number_choices=self.number_choices,
                suit_choices=self.suit_choices,
                suggested_number=num_s,
                suggested_suit=suit_s,
            )
            out = dialog.run()
            if out is None:
                # annulé → renvoyer meilleure info connue (ou "?")
                return IdentifyResult(num_s or "?", suit_s or "?", {
                    "source": "cancel",
                    "score_number": float(s_num),
                    "score_suit": float(s_suit),
                })
            lab_num, lab_suit = out
            save_number = missing_number and lab_num not in {"", "?"}
            save_suit = missing_suit and lab_suit not in {"", "?"}
            self._save_if_missing(
                tnum,
                tsuit,
                lab_num,
                lab_suit,
                save_number,
                save_suit,
                base_key,
                tpl_set,
            )
            # MAJ index pour la session courante
            if save_number:
                self._update_index(lab_num, tnum, is_number=True, template_set=tpl_set)
            if save_suit:
                self._update_index(lab_suit, tsuit, is_number=False, template_set=tpl_set)
            return IdentifyResult(lab_num or (num_s or "?"), lab_suit or (suit_s or "?"), {
                "source": "labeled",
                "score_number": float(s_num),
                "score_suit": float(s_suit),
            })

        # 3) Non interactif → renvoyer la meilleure hypothèse (ou "?")
        return IdentifyResult(num_s or "?", suit_s or "?", {
            "source": "guess",
            "score_number": float(s_num),
            "score_suit": float(s_suit),
        })

    # ---------- API pratique (image table + base_key) ----------
    def identify_from_table(
        self,
        table_img: Image.Image,
        regions: Dict[str, object],
        base_key: str,
        *,
        interactive: bool = True,
        force_all: bool = False,
    ) -> IdentifyResult:
        pairs = collect_card_patches(table_img.convert("RGB"), regions, pad=0)
        card_patch = pairs.get(base_key)
        if not card_patch:
            return IdentifyResult("?", "?", {"source": "error", "reason": "region-missing"})
        if not is_card_present(card_patch.number, threshold=215, min_ratio=0.04):
            return IdentifyResult("?", "?", {"source": "empty", "reason": "no-card"})
        return self.identify_from_patches(
            card_patch.number,
            card_patch.suit,
            base_key=base_key,
            template_set=card_patch.template_set,
            interactive=interactive,
            force_all=force_all,
        )

    # ---------- Sauvegarde minimale des nouveaux exemples ----------
    def _save_if_missing(
        self,
        num_img: Image.Image,
        suit_img: Image.Image,
        number_label: str,
        suit_label: str,
        save_number: bool,
        save_suit: bool,
        base_key: str,
        template_set: Optional[str],
    ) -> None:
        ts = int(time.time())
        idx = next(self._counter)
        base = f"{base_key}_{ts}_{idx:04d}"
        resolved_set = self._normalise_template_set(template_set)
        self._last_template_set = resolved_set
        if resolved_set:
            root = self.cards_root / resolved_set
        else:
            root = self.cards_root
        if save_number:
            p = root / "numbers" / number_label / f"{base}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            num_img.save(p)
        if save_suit:
            p = root / "suits" / suit_label / f"{base}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            suit_img.save(p)

    def _update_index(
        self,
        label: str,
        img: Image.Image,
        *,
        is_number: bool,
        template_set: Optional[str],
    ) -> None:
        resolved_set = self._normalise_template_set(template_set)
        self.idx.append_template(resolved_set, label, img, is_number=is_number)


