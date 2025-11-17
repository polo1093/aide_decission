"""Entités décrivant les boutons d'action disponibles sur la table."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional


@dataclass
class Button:
    """Représentation d'un bouton unique présent sur l'interface."""

    name: str
    coordonate: Optional[tuple[int, int, int, int]] = None
    visible: bool = False
    enabled: bool = False
    score: float = 0.0

    def update(self, visible: bool, score: float, threshold: float = 0.8) -> None:
        """Met à jour l'état d'observation et déduit l'état cliquable."""
        self.visible = visible
        self.score = score
        self.enabled = visible and score >= threshold

    def reset(self) -> None:
        """Réinitialise complètement l'état du bouton."""
        self.visible = False
        self.enabled = False
        self.score = 0.0


@dataclass
class Buttons:
    """Collection utilitaire regroupant l'ensemble des boutons connus."""

    fold: Button = field(default_factory=lambda: Button(name="fold"))
    check_call: Button = field(default_factory=lambda: Button(name="check_call"))
    _raise: Button = field(default_factory=lambda: Button(name="raise"), repr=False)

    def __iter__(self) -> Iterator[Button]:
        """Itère sur tous les boutons suivis."""
        yield self.fold
        yield self.check_call
        yield self._raise

    def by_name(self) -> dict[str, Button]:
        """Retourne un mapping pratique {nom: bouton}."""
        return {
            "fold": self.fold,
            "check_call": self.check_call,
            "raise": self._raise,
        }

    def reset_all(self) -> None:
        """Réinitialise l'ensemble des boutons."""
        for button in self:
            button.reset()


Buttons.raise = property(lambda self: self._raise, doc="Alias car 'raise' est un mot-clé Python.")


__all__ = ["Button", "Buttons"]
