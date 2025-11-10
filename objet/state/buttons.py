"""Gestion de l'état des boutons d'action."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from objet.entities.bouton import Bouton
from objet.state.utils import extract_scan_value


@dataclass
class ButtonsState:
    """Maintient les trois boutons d'action."""

    buttons: Dict[str, Bouton] = field(
        default_factory=lambda: {f"button_{i}": Bouton() for i in range(1, 4)}
    )

    def update_from_scan(self, scan_table: Mapping[str, Any]) -> None:
        for name, btn in self.buttons.items():
            raw_value = extract_scan_value(scan_table, name)
            btn.string_to_bouton(raw_value)

    def best_button(self) -> Optional[str]:
        best_name: Optional[str] = None
        best_gain: float = float("-inf")
        for name, btn in self.buttons.items():
            if btn.gain is None:
                continue
            if btn.gain > best_gain:
                best_gain = btn.gain
                best_name = name
        return best_name


__all__ = ["ButtonsState"]
