import cv2
import numpy as np
import logging
from typing import Dict, Tuple

from folder_tool import timer



# Import des fonctions utilitaires depuis le dossier scripts
try:
    from scripts.crop_core import find_ref_point,crop_from_size_and_offset
except ImportError:
    # Fallback au cas où le script ne serait pas dans le path
    find_ref_point = None

class ScanTable():
    def __init__(self):

        self.screen_array = np.array(ImageGrab.grab())
        self.screen_reference = cv2.imread('screen/launch/me.png', 0) #paht dans PMU par défaut mais  en passer paramètre
        self.screen_crop = None
        
        
        self.TIMER_SCAN_REFRESH = 0.5 
        self.timer_screen = timer.Timer(self.TIMER_SCAN_REFRESH)
        self.table_origin = None                # (x0, y0, x1, y1) absolu sur le screen global
        self.scan_string = "don t find"


    def test_scan(self,debug=False):
        while(self.screen_refresh==False):
            wait(0.1)
        if self.screen_refresh():
            if  self.find_table():
                 return True
            self.scan_string = "don't find"
            return False
       
            

        
    def screen_refresh(self):
        if self.timer_screen.is_expire():
            self.timer_screen.refresh()
            self.screen_array = np.array(ImageGrab.grab())
            return True
        return False
    
    def find_table(self):
        """Trouve la table en utilisant la fonction de template matching du module crop_core."""
        

        self.screen_crop,self.table_origin=  crop_from_size_and_offset()
        if self.table_origin is  None:
            self.scan_string = "don't find"
            return False
        return True
   
    
      




