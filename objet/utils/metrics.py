"""Snapshot immuable des métriques poker dérivées."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, TYPE_CHECKING, Literal


Street = Literal["PREFLOP", "FLOP", "TURN", "RIVER", "UNKNOWN"]

if TYPE_CHECKING:  # import tardif pour éviter un cycle au runtime
    from objet.entities.card import CardsState
    from objet.entities.player import Player, Players
    from objet.services.game import Game


@dataclass(frozen=True)
class MetricsState:
    """Instantané de métriques prêtes à alimenter la décision."""

    pot: float
    players_active: int
    players_at_start: int
    hero_stack: float
    hero_stack_start_hand: Optional[float]
    hero_stack_delta: Optional[float]
    hero_invested: Optional[float]
    board_cards: int
    street: Street
    player_stacks: Dict[str, float]
    chance_win_0: Optional[float] = None
    chance_win_x: Optional[float] = None

    @classmethod
    def from_game(
        cls,
        game: "Game",
        *,
        hero_index: int = 0,
        chance_win_0: Optional[float] = None,
        chance_win_x: Optional[float] = None,
    ) -> "MetricsState":
        """Construit un snapshot cohérent à partir de l'état courant du jeu."""

        table = getattr(game, "table", None)
        if table is None:
            raise ValueError("L'objet Game ne possède pas de table initialisée.")

        players_state = getattr(table, "players", None)
        if players_state is None:
            raise ValueError("Aucun joueur n'est attaché à la table.")
        if len(players_state) == 0:
            raise ValueError("La table ne contient aucun joueur exploitable.")

        hero = cls._resolve_hero(players_state, hero_index)
        pot_amount = cls._extract_pot_amount(table)
        board_cards = cls._count_board_cards(getattr(table, "cards", None))
        street = cls._street_from_board(board_cards)
        player_stacks = cls._build_player_stack_map(players_state)
        players_active = cls._count_active_players(players_state)
        players_at_start = cls._count_starting_players(players_state)

        hero_stack = cls._extract_stack(hero)
        hero_start_stack = cls._normalize_optional_number(getattr(hero, "fond_start_Party", None))
        hero_delta = (
            hero_stack - hero_start_stack if hero_start_stack is not None else None
        )
        hero_invested = (
            max(0.0, hero_start_stack - hero_stack) if hero_start_stack is not None else None
        )

        return cls(
            pot=pot_amount,
            players_active=players_active,
            players_at_start=players_at_start,
            hero_stack=hero_stack,
            hero_stack_start_hand=hero_start_stack,
            hero_stack_delta=hero_delta,
            hero_invested=hero_invested,
            board_cards=board_cards,
            street=street,
            player_stacks=player_stacks,
            chance_win_0=chance_win_0,
            chance_win_x=chance_win_x,
        )

    def with_chances(
        self,
        *,
        chance_win_0: Optional[float],
        chance_win_x: Optional[float],
    ) -> "MetricsState":
        """Retourne un nouveau snapshot enrichi des probabilités de gain."""

        return replace(
            self,
            chance_win_0=chance_win_0,
            chance_win_x=chance_win_x,
        )

    @staticmethod
    def _resolve_hero(players: "Players", hero_index: int) -> "Player":
        if hero_index < 0:
            raise ValueError("L'index du héros ne peut pas être négatif.")
        try:
            return players[hero_index]
        except IndexError as exc:
            raise ValueError(
                f"Impossible de récupérer le joueur héros à l'index {hero_index}."
            ) from exc

    @staticmethod
    def _extract_pot_amount(table: object) -> float:
        pot = getattr(table, "pot", None)
        if pot is None:
            raise ValueError("La table ne fournit pas de structure de pot.")
        amount = getattr(pot, "amount", None)
        if amount is None:
            raise ValueError("Le montant du pot est indisponible.")
        return float(amount)

    @staticmethod
    def _count_board_cards(cards_state: Optional["CardsState"]) -> int:
        if cards_state is None:
            raise ValueError("Aucune information de cartes n'est disponible.")
        board_cards = getattr(cards_state, "board", None)
        if board_cards is None:
            raise ValueError("Les cartes du board ne sont pas accessibles.")
        known = [card for card in board_cards if getattr(card, "formatted", None)]
        return len(known)

    @staticmethod
    def _street_from_board(board_cards: int) -> Street:
        if board_cards == 0:
            return "PREFLOP"
        if board_cards == 3:
            return "FLOP"
        if board_cards == 4:
            return "TURN"
        if board_cards == 5:
            return "RIVER"
        return "UNKNOWN"

    @staticmethod
    def _build_player_stack_map(players: "Players") -> Dict[str, float]:
        stacks: Dict[str, float] = {}
        for idx, player in enumerate(players, start=1):
            fond = getattr(player, "fond", None)
            if fond is None:
                raise ValueError(f"Le joueur J{idx} ne possède pas de fond attaché.")
            amount = getattr(fond, "amount", None)
            if amount is None:
                raise ValueError(f"Le stack du joueur J{idx} est indisponible.")
            stacks[f"J{idx}"] = float(amount)
        return stacks

    @staticmethod
    def _count_active_players(players: "Players") -> int:
        active = sum(1 for player in players if player.is_activate())
        if active <= 0:
            raise ValueError("Aucun joueur actif détecté sur la table.")
        return active

    @staticmethod
    def _count_starting_players(players: "Players") -> int:
        starting = sum(1 for player in players if getattr(player, "active_at_start", False))
        if starting <= 0:
            raise ValueError(
                "Impossible de déterminer le nombre de joueurs présents au début de la main."
            )
        return starting

    @staticmethod
    def _extract_stack(player: "Player") -> float:
        fond = getattr(player, "fond", None)
        if fond is None:
            raise ValueError("Le joueur héros ne possède pas de fond associé.")
        amount = getattr(fond, "amount", None)
        if amount is None:
            raise ValueError("Le stack courant du héros est indisponible.")
        return float(amount)

    @staticmethod
    def _normalize_optional_number(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return float(value)


__all__ = ["MetricsState", "Street"]
