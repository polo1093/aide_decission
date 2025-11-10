"""Gestion simplifiée des informations relatives aux joueurs."""
from __future__ import annotations

from objet.entities.player import Player

PHASES = ["No_start", "preflop", "flop", "turn", "river", "end"]
NAME_OPPONENTS = ["J1", "J2", "J3", "J4", "J5"]


class Party:
    def __init__(self) -> None:
        self.opponents = {name: Player() for name in NAME_OPPONENTS}
        self.me = Player()
        self.pot = 0
        self.current_phase = "No_start"

    def refresh(self, table):
        for key in NAME_OPPONENTS:
            self.opponents[key].refresh(table[key])
        self.current_phase = self.game_phase(table)

    def active_days_at_start(self) -> int:
        return sum(player.active_days_at_start for player in self.opponents.values())

    def active_player_count(self) -> int:
        return sum(player.active_days_at_start for player in self.opponents.values())

    def game_phase(self, table):
        nbr_card_on_board = sum(1 for card in table["board"] if card)
        phase = ""

        if nbr_card_on_board == 5:
            phase = "river"
        if nbr_card_on_board == 4:
            phase = "turn"
        if nbr_card_on_board == 3:
            phase = "flop"
        if nbr_card_on_board == 0:
            phase = "preflop"

        if False:  # Gagner :
            phase = "end"
        return phase


__all__ = ["Party", "PHASES", "NAME_OPPONENTS"]
