"""Entités décrivant les boutons d'action disponibles sur la table."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from objet.utils.calibration import bbox_from_region, load_coordinates   
DEFAULT_COORD_PATH = Path("config/PMU/coordinates.json")

list_etat_button= ["check", "relance", "mise", "fold", "paie", "all-in"  ]

@dataclass
class Buttons:
    """Collection utilitaire regroupant l'ensemble des boutons connus."""
    coord_path: Path | str = DEFAULT_COORD_PATH
    button : list[Button] = field(default_factory=list)

    def __post_init__(self) -> None:
        regions, templates_resolved, _ = load_coordinates(self.coord_path)
        if not self.button:
            self.button = [ Button(
                coordonate=bbox_from_region(regions.get(f"button_{i}")),
                ) for i in range(1,4)]
    
    def __iter__(self) -> Iterator[Button]:
        return iter(self.button)

    def __len__(self) -> int:
        return len(self.button)

    def __getitem__(self, index: int) -> Button:
        return self.button[index]   
    
    def one_is_activate(self) -> bool:
        for b in self.button:
            if b.is_activate: return True
        return False


    def reset_all(self) -> None:
        """Réinitialise l'ensemble des boutons."""
        for button in self:
            button.reset()
            
@dataclass
class Button:
    """Représentation d'un bouton unique présent sur l'interface."""

    texte: str =""
    etat : str =""
    value: float = 0.0
    coordonate: Optional[tuple[int, int, int, int]] = None
    enabled: bool = False
    score: float = 0.0



    def reset(self) -> None:
        """Réinitialise complètement l'état du bouton."""
        self.enabled = False
        self.score = 0.0
        self.texte = ""
        
    def is_activate(self) -> bool:
        return self.enabled
        
    def apply_scan(self, texte)-> None:
        if not texte:
            self.enabled = False
            return 
        self.texte=texte
        etat = one_element_in_list_str(list_etat_button,texte)
        if etat != None :
            self.enabled = True
        # TODO value
            
        
        


def _is_target_with_two_missing(candidate: str, target: str) -> bool:
    if len(target) - len(candidate) != 2:
        return False
    it = iter(target)
    return all(ch in it for ch in candidate)

def one_element_in_list_str(list_str,str) -> str:
    for cand in list_str:
        if _is_target_with_two_missing(cand, target):
            return cand
    return None

__all__ = ["Button", "Buttons"]


if __name__ == "__main__":
    bs = Buttons()
    b=Button()
    print(b)
    print(b.is_activate())
    print(bs.button[0])