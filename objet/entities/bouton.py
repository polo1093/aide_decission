import logging
from dataclasses import dataclass, field
from typing import ClassVar, Optional
import unicodedata
import re
import tool


@dataclass
class Action:
    """Représente une action détectée sur un bouton."""

    POSSIBLE_ACTIONS_BOUTON: ClassVar[list[str]] = [
        "parole",
        "suivre",
        "relancer a",
        "se coucher",
        "miser",
    ]
    liste_actions: ClassVar[list[str]] = POSSIBLE_ACTIONS_BOUTON + [
        "pas en jeu",
        "relance à fois 4",
        "close",
    ]  # À améliorer

    name: Optional[str] = None
    value: Optional[float] = None

    @classmethod
    def _normalize_string(cls, s):
        """
        Convertit une chaîne en minuscules et enlève les accents.
        
        Args:
            s (str): La chaîne à normaliser.
        
        Returns:
            str: La chaîne normalisée.
        """
        s = s.lower()
        # Enlever les accents
        s = ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
        return s

    
    # def get_possible_actions(cls):
    #     """
    #     Retourne la liste complète des actions possibles.
        
    #     Returns:
    #         list: Liste des actions possibles.
    #     """
    #     return cls.liste_actions

    @classmethod
    def create_action_from_string(cls, action_str):
        """
        Crée une instance d'Action à partir d'une chaîne de caractères.
        
        Args:
            action_str (str): La chaîne de caractères représentant l'action (e.g., "Mise 0,02 €").
        
        Returns:
            Action or None: Une instance d'Action si valide, sinon None.
        """
        if not action_str:
            return None

        normalized_str = cls._normalize_string(action_str)

        # Trouver l'action correspondante
        action_found = None
        for action in cls.POSSIBLE_ACTIONS_BOUTON:
            if action[:3] in normalized_str:
                action_found = action
                break

        if not action_found:
            logging.warning(f"Action non reconnue dans la chaîne : '{action_str}'")
            return None

        # Extraire la valeur numérique avec la virgule, si présente
        value = None
        match = re.search(r'(\d+,\d+)', normalized_str)
        if match:
            value_str = match.group(1).replace(',', '.')
            try:
                value = tool.convert_to_float(value_str)
            except ValueError:
                logging.warning(f"Impossible de convertir la valeur '{value_str}' en float.")
                value = None

        return cls(name=action_found.capitalize(), value=value)



@dataclass
class Bouton:
    """Représente un bouton d'action détecté à l'écran."""

    POSSIBLE_ACTIONS: ClassVar[list[str]] = Action.POSSIBLE_ACTIONS_BOUTON

    name: Optional[str] = None
    value: Optional[float] = None
    gain: Optional[float] = None

    def string_to_bouton(self, button_string):
        """
        Args:
            button_string (str): La chaîne de caractères extraite par l'OCR (e.g., "Mise 0,02 €").
        
        Returns:
            True or None.
        """
        # Utiliser la méthode de la classe Action pour créer une Action
        action_instance = Action.create_action_from_string(button_string)
        self.gain=None
        if action_instance:
            self.name = action_instance.name
            self.value = action_instance.value
            return True
        else:
            self.name = None
            self.value = None
            return None

