"""Entités décrivant les joueurs présents à la table."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


list_etat = ["No_start", "fold", "play","paid"]

@dataclass
class Player_etat:
    etat: Optional[str] = list_etat[0]


@dataclass
class Player:
    coordonate_money: Optional[tuple[int, int, int, int]] = None
    coordonate_etat: Optional[tuple[int, int, int, int]] = None
    active_days_at_start: bool = False
    fond: Optional[float] = 0
    fond_start: Optional[float] = 0
    
    def __init__(self):
        self.etat = Player_etat()
    
    money_relance: Optional[float] = 0
    money_paid: Optional[float] = 0

    def refresh(self, fond: Optional[float] = 0) -> None:
        self.fond = fond
        if self.fond:
            self.active_days_at_start = True

    def is_activate(self) -> None:
        return True if self.etat in  ["play" , "paid"] else False



if __name__ == "__main__":
    p=Player()
    print(p)
    print(p.is_activate())

__all__ = ["Player"]
