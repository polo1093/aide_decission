"""Gestion des métriques numériques de la table."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import tool
from objet.utils.state_utils import extract_scan_value


@dataclass
class MetricsState:
    """Regroupe les métriques numériques."""

    pot: Optional[float] = None
    fond: Optional[float] = None
    chance_win_0: Optional[float] = None
    chance_win_x: Optional[float] = None
    player_money: Dict[str, Optional[float]] = field(
        default_factory=lambda: {f"J{i}": None for i in range(1, 6)}
    )
    players_count: int = 0

    def update_from_scan(self, scan_table: Mapping[str, Any]) -> None:
        self.pot = tool.convert_to_float(extract_scan_value(scan_table, "pot"))
        self.fond = tool.convert_to_float(extract_scan_value(scan_table, "fond"))
        for key in list(self.player_money.keys()):
            raw_key = f"player_money_{key}"
            self.player_money[key] = tool.convert_to_float(extract_scan_value(scan_table, raw_key))
        self.players_count = sum(
            1 for money in self.player_money.values() if money not in (None, 0)
        )


__all__ = ["MetricsState"]
