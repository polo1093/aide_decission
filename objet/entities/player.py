"""Entités décrivant les joueurs présents à la table."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Player:
    active_days_at_start: bool = False
    fond: Optional[float] = 0
    fond_start: Optional[float] = 0
    active_player: bool = False
    money_relance: Optional[float] = 0
    money_paid: Optional[float] = 0

    def refresh(self, fond: Optional[float]) -> None:
        self.fond = fond
        if self.fond:
            self.active_days_at_start = True


__all__ = ["Player"]
