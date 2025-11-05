import matplotlib.pyplot as plt

import cv2
import numpy as np
import PIL
import logging
from PIL import ImageGrab, Image, ImageDraw, ImageFont
import pyautogui

from folder_tool import ocr
import constants
from constants import TIMER_SCAN_REFRESH
from folder_tool import timer
from Type_game import Type_game
from pokereval.card import Card

import tool

class ScanTable():
    def __init__(self):
        self.reader = ocr.OCR()
        self.screen_old = Image.open('screen/debug/DXFo5zrJK5.jpg')
        self.screen_array = np.array(ImageGrab.grab())
        self.timer_screen = timer.Timer(TIMER_SCAN_REFRESH)
        self.reference_point=None
        self.table = constants.load_coordinates()
        self.scan_string = "don t find"
        self.screen_reference = cv2.imread('screen/launch/me.png', 0)
        self.screen_pre_tour = cv2.imread('screen/launch/test_pre_tour.png', 0)
        if self.screen_reference is None:
            logging.error("Impossible de charger l'image de référence de la table.")

    def scan(self,debug=False):
        self.screen_refresh()
        self.reset_table()
        if self.find_table() and self.test_pre_tour:
            self.calculate_absolute_coordinates()
            self.table = self.reader.ocr_table(self.table, self.screen_old,debug=debug)
            
            return True
        self.scan_string = "don't find"
        return False
    

    def reset_table(self):
        """Réinitialise uniquement les "value" du tableau sans affecter les autres informations."""
        for key, info in self.table.items():
            if 'value' in info:
                info['value'] = None
        
    def screen_refresh(self):
        if self.timer_screen.is_expire():
            self.timer_screen.refresh()
            self.screen_old = ImageGrab.grab()
            self.screen_array = np.array(self.screen_old)
        return self.screen_old
    
    def find_table(self):
        """find_table return the top right position of the table"""
        screen_gray = np.array(self.screen_old.convert('L'))  # convert image to grayscale
        res = cv2.matchTemplate(screen_gray, self.screen_reference, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > 0.92:
            self.reference_point = max_loc
        else:
            self.reference_point = None
            logging.debug("Table non trouvée dans l'image.")
        return self.reference_point
    
    def test_pre_tour(self):
        screen_gray = np.array(self.screen_old.convert('L'))  # convert image to grayscale
        res = cv2.matchTemplate(screen_gray, self.screen_pre_tour, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > 0.92:
            return True
        return False
    
    def get_image_chunk(self, coord_relatif):
        """
        Prend une liste de coordonnées relatives [x1, y1, x2, y2].
        Utilise coord_abs pour obtenir des coordonnées absolues.
        Renvoie un morceau de l'image de self.screen_array basé sur ces coordonnées.
        """
        abs_coords = self.coord_abs(coord_relatif)
        return self.screen_array[abs_coords[1]:abs_coords[3], abs_coords[0]:abs_coords[2]]

    def scan_point_white(self, coordonnees=[0,0]):
        """Scan a point and return True if it's close to white, False otherwise."""
        try:
            point_color = self.screen_array[coordonnees[1], coordonnees[0]]  # Get color at x, y
        except IndexError:
            logging.warning(f"Coordonnées hors limites: {coordonnees}")
            return False
        white = np.array([255, 255, 255])
        # Compute the Euclidean distance between the point color and white
        distance = np.linalg.norm(point_color - white)
        return distance < 30        

    def coord_abs(self, coord_relatif):
        """ 
        Prend une liste de coordonnées relatives [x1, y1, x2, y2].
        Renvoie la liste des coordonnées absolues basée sur self.reference_point.
        """
        return [self.reference_point[0] + coord_relatif[0], self.reference_point[1] + coord_relatif[1],
                self.reference_point[0] + coord_relatif[2], self.reference_point[1] + coord_relatif[3]]

    def calculate_absolute_coordinates(self):
        for region_name, region_info in self.table.items():
            coord_rel = region_info['coord_rel']
            self.table[region_name]['coord_abs'] = self.coord_abs(coord_rel)

    def show_debug_image(self):
        """Affiche l'image de débogage avec les rectangles rouges et les noms des régions."""
        # Ajouter une image de débogage (copie de l'écran)
        img= self.screen_old.copy()
        self.draw = ImageDraw.Draw(img)
        
        for region_name, region_info in self.table.items():
            coord_abs = region_info.get('coord_abs')
            if coord_abs:
                left, upper, right, lower = coord_abs
                # Dessiner le rectangle rouge
                self.draw.rectangle([left, upper, right, lower], outline='red', width=2)
                
                # Charger une police de caractères
                try:
                    font = ImageFont.truetype("arial.ttf", 15)
                except IOError:
                    font = ImageFont.load_default()
                
                # Calculer la position du texte (nom de la région)
                text_position = (left, upper - 15 if upper - 15 > 0 else upper + 5)
                
                # Dessiner le nom de la région au-dessus du rectangle
                self.draw.text(text_position, region_name, fill='red', font=font)
        
        # Afficher l'image avec les rectangles et les noms
        img.show()
    


import time
import logging

def main():
    # Configurer le logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialiser l'objet ScanTable
    scan_table = ScanTable()
    
    # Enregistrer le temps de début
    start_time = time.time()
    
    # Effectuer le scan avec le mode débogage activé
    success = scan_table.scan(debug=True)
    
    # Enregistrer le temps de fin
    end_time = time.time()
    
    # Calculer le temps total
    total_time = end_time - start_time
    
    # Afficher les résultats
    if success:
        print("Scan réussi. Résultats:")
        for key, value in scan_table.table.items():
            print(f"{key}: {value}")
        
        # Afficher l'image avec les rectangles rouges et les noms
        scan_table.show_debug_image()
    else:
        print(f"Scan échoué: {scan_table.scan_string}")
    
    # Afficher le temps total
    print(f"Temps de traitement total : {total_time:.2f} secondes")

if __name__ == "__main__":
    main()

