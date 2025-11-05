import tool
import pyautogui
import sys
import time
from dataclasses import dataclass
from typing import Tuple
import pytweening
import random

@dataclass
class Cliqueur:
    wait_default: float = 0.05
    wait_press_default: float = 0.15

    def __post_init__(self):
        pass

    def click(self, coords: Tuple[int, int], button: str = 'left', wait: float = None, wait_press: float = None):
        """
        Effectue un clic de souris aux coordonnées spécifiées.

        Args:
            coords (Tuple[int, int]): Les coordonnées (x, y) pour le clic.
            button (str, optional): Le bouton de la souris à utiliser ('left', 'right'). Defaults to 'left'.
            wait (float, optional): Temps d'attente avant le clic. Defaults to wait_default.
            wait_press (float, optional): Durée du clic. Defaults to wait_press_default.
        """
        if wait is None:
            wait = self.wait_default
        if wait_press is None:
            wait_press = self.wait_press_default

        # Déplace la souris vers les coordonnées spécifiées avec une durée et un tweening aléatoires
        pyautogui.moveTo(
            coords[0],
            coords[1],
            duration=random.uniform(0.2, 0.6),
            tween=pytweening.easeInOutBounce
        )
        time.sleep(wait)
        pyautogui.mouseDown(button=button)
        time.sleep(wait_press + random.uniform(0.2, 0.6))
        pyautogui.mouseUp(button=button)
        time.sleep(wait)

    def click_button(self, button_rect: Tuple[int, int, int, int], shrink_factor: float = 0.6):
        """
        Clique sur un bouton en utilisant un point aléatoire dans un rectangle réduit.

        Args:
            button_rect (Tuple[int, int, int, int]): Coordonnées du rectangle du bouton (left, top, right, bottom).
        """
        # Décompose les coordonnées
        left, top, right, bottom = button_rect
        width = right - left
        height = bottom - top

        # Réduit le rectangle de 40% pour obtenir un rectangle centré plus petit
        new_width = width * shrink_factor
        new_height = height * shrink_factor
        left += (width - new_width) / 2
        top += (height - new_height) / 2

        # Sélectionne un point aléatoire dans le nouveau rectangle
        x = random.uniform(left, left + new_width)
        y = random.uniform(top, top + new_height)
        coords = (x, y)

        # Effectue le clic aux coordonnées choisies
        self.click(coords, 'left')
