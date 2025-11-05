#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bundle_to_clipboard.py — Concatène tous les .py d'un projet en Markdown,
écrit le fichier (écrasement) et copie le contenu au presse-papiers.
"""

from __future__ import annotations
import argparse
import datetime as dt
import fnmatch
import os
from pathlib import Path

try:
    import pyperclip
except ImportError:
    pyperclip = None

DEFAULT_IGNORE_DIRS = {
    ".git", "__pycache__", "venv", ".venv", ".mypy_cache", ".pytest_cache",
    ".idea", ".vscode", ".ipynb_checkpoints", "build", "dist"
}

def should_ignore(rel: Path, extra_ignores: list[str]) -> bool:
    # ignore si un composant est dans DEFAULT_IGNORE_DIRS
    for part in rel.parts:
        if part in DEFAULT_IGNORE_DIRS:
            return True
    # ignore via patterns glob supplémentaires
    rel_posix = rel.as_posix()
    return any(fnmatch.fnmatch(rel_posix, pat) for pat in extra_ignores)

def collect_python_sources(base: Path, exclude_files: set[Path], extra_ignores: list[str]) -> list[Path]:
    out = []
    for p in base.rglob("*.py"):
        rel = p.relative_to(base)
        if should_ignore(rel, extra_ignores):
            continue
        if p.resolve() in exclude_files:
            continue
        out.append(p)
    # ordre déterministe
    return sorted(out, key=lambda x: x.as_posix().lower())

def build_markdown(base: Path, files: list[Path]) -> str:
    lines = [
        f"# Bundle — {base.resolve().name}",
        f"_Généré le {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "## Fichiers",
        ""
    ]
    for f in files:
        rel = f.relative_to(base).as_posix()
        lines.append(f"### {rel}\n```python")
        try:
            txt = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            txt = "# [encodage non-UTF8 — ignoré]"
        lines.append(txt)
        lines.append("```")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="Bundle .py -> Markdown + copie presse-papiers")
    ap.add_argument("base_dir", nargs="?", default=".", help="Racine du projet (défaut: .)")
    ap.add_argument("-o", "--output", default="context_bundle.md", help="Chemin du fichier de sortie")
    ap.add_argument("--exclude", action="append", default=[], help="Fichier à exclure (option répétable)")
    ap.add_argument("--add-ignore", action="append", default=[], help="Pattern glob à ignorer (répétable)")
    ap.add_argument("--no-clipboard", action="store_true", help="N'essaie pas de copier dans le presse-papiers")
    args = ap.parse_args()

    base = Path(args.base_dir).resolve()
    out_path = Path(args.output).resolve()

    exclude_files = {Path(__file__).resolve()}
    exclude_files.update(Path(p).resolve() for p in args.exclude)

    files = collect_python_sources(base, exclude_files, args.add_ignore)
    content = build_markdown(base, files)

    # écrasement du fichier de sortie
    out_path.write_text(content, encoding="utf-8")
    print(f"Écrit: {out_path} ({len(content)} caractères)")

    if not args.no_clipboard:
        if pyperclip is None:
            print("pyperclip non installé: pip install pyperclip")
        else:
            try:
                pyperclip.copy(content)
                print("Copié dans le presse-papiers ✅")
            except Exception as e:
                print(f"Impossible de copier dans le presse-papiers: {e}")
                print("Sur Linux, installez 'xclip' ou 'xsel' puis réessayez.")
                # cf. docs pyperclip

if __name__ == "__main__":
    main()
