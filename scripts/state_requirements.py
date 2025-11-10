"""Definitions des portions d'état utilisées par les scripts de workflow.

Ce module centralise la cartographie demandée afin que les outils ou
l'interface puissent instancier :class:`objet.services.game.Game` avec les éléments
pertinents pour chaque script. Les catégories sont volontairement grossières
(cartes, boutons, métriques, captures) et couvrent les besoins majeurs des
workflows existants.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet


class StatePortion(str, Enum):
    """Portions logiques de l'état de jeu consommées par les scripts."""

    CARDS = "cards"
    BUTTONS = "buttons"
    METRICS = "metrics"
    CAPTURES = "captures"


@dataclass(frozen=True)
class ScriptStateUsage:
    """Description des portions d'état nécessaires à un script."""

    name: str
    portions: FrozenSet[StatePortion]
    description: str


SCRIPT_STATE_USAGE: Dict[str, ScriptStateUsage] = {
    "capture_cards.py": ScriptStateUsage(
        name="capture_cards.py",
        portions=frozenset({StatePortion.CARDS, StatePortion.CAPTURES}),
        description="Extraction et reconnaissance des cartes depuis une capture.",
    ),
    "Crop_Video_Frames.py": ScriptStateUsage(
        name="Crop_Video_Frames.py",
        portions=frozenset({StatePortion.CAPTURES}),
        description="Découpe périodique des captures vidéo à partir des paramètres de table.",
    ),
    "crop_core.py": ScriptStateUsage(
        name="crop_core.py",
        portions=frozenset({StatePortion.CAPTURES}),
        description="Fonctions communes de capture/crop et outils de validation géométrique.",
    ),
    "position_zones.py": ScriptStateUsage(
        name="position_zones.py",
        portions=frozenset(
            {
                StatePortion.CAPTURES,
                StatePortion.CARDS,
                StatePortion.BUTTONS,
                StatePortion.METRICS,
            }
        ),
        description="Éditeur Tk classique des zones OCR (cartes, boutons, métriques).",
    ),
    "position_zones_ctk.py": ScriptStateUsage(
        name="position_zones_ctk.py",
        portions=frozenset(
            {
                StatePortion.CAPTURES,
                StatePortion.CARDS,
                StatePortion.BUTTONS,
                StatePortion.METRICS,
            }
        ),
        description="Éditeur CustomTkinter des zones OCR (cartes, boutons, métriques).",
    ),
    "zone_project.py": ScriptStateUsage(
        name="zone_project.py",
        portions=frozenset(
            {
                StatePortion.CAPTURES,
                StatePortion.CARDS,
                StatePortion.BUTTONS,
                StatePortion.METRICS,
            }
        ),
        description="Modèle et opérations associées aux projets de zones OCR.",
    ),
    "copy_python_sources.py": ScriptStateUsage(
        name="copy_python_sources.py",
        portions=frozenset(),
        description="Outil utilitaire sans dépendance sur l'état de jeu.",
    ),
}


def describe_scripts() -> Dict[str, Dict[str, str]]:
    """Retourne un dictionnaire sérialisable listant les usages déclarés."""

    return {
        name: {
            "portions": sorted(usage.portions),
            "description": usage.description,
        }
        for name, usage in SCRIPT_STATE_USAGE.items()
    }


__all__ = ["StatePortion", "ScriptStateUsage", "SCRIPT_STATE_USAGE", "describe_scripts"]
