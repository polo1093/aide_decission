# launch_controller.py à la racine du projet

import cv2
import numpy as np
import PIL
from PIL import ImageGrab, Image
# from objet.services.cliqueur import Cliqueur
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class Controller():
    def __init__(self):
        self.count = 0
        self.running = False
        self.cpt = 0
        self.game_stat = {}
        #
        from objet.services.game import Game
        self.game = Game()
        # self.click = Cliqueur()
        
    def main(self):
        
             # machine à état de la partie et save 
            # Todo
        
            
            
        if self.game.scan_to_data_table():
        
            self.game.update_from_scan()
            

        
        
        
        
            return self.game_stat_to_string()
        self.cpt += 1
        return "don t find"+f"     Scan n°{self.cpt}"
        

    
    def game_stat_to_string(self):
        """
        Formate les informations du jeu pour l'utilisateur.

        Returns:
            str: Une chaîne de caractères contenant les informations formatées.
        """
        # Récupération des informations de base
        # metrics = self.game.metrics
        # nbr_player = metrics.players_count
        # pot = metrics.pot
        # fond = metrics.fond
        # chance_win_0 = metrics.chance_win_0
        # chance_win_x = metrics.chance_win_x

        # Fonction pour arrondir à 4 chiffres significatifs
        def round_sig(x, sig=4):
            if isinstance(x, (int, float)):
                return float(f"{x:.{sig}g}")
            else:
                return x

        # Arrondi des valeurs numériques
        # pot = round_sig(pot)
        # fond = round_sig(fond)
        # chance_win_0 = round_sig(chance_win_0)
        # chance_win_x = round_sig(chance_win_x)

        # Informations sur les cartes du joueur
        me_cards = [card.formatted for card in self.game.cards.me_cards()]
        me_cards_str = ', '.join(me_cards)

        # Informations sur le board
        board_cards = [card.formatted for card in self.game.cards.board_cards()]
        board_cards_str = ', '.join(board_cards)

        # Informations sur les boutons
        # buttons_info = []
        # # Ajout d'une ligne d'en-tête avec des largeurs de colonnes fixes
        # buttons_info.append(f"{'Bouton':<10} {'Action':<15} {'Valeur':<10} {'Gain':<10}")
        # buttons_info.append('-' * 50)  # Ligne de séparation

        # for i in range(1, 4):
        #     button = self.game.table.buttons.buttons.get(f'button_{i}')
        #     if button:
        #         name = button.name if button.name is not None else ''
        #         value = round_sig(button.value) if button.value is not None else ''
        #         gain = round_sig(button.gain) if button.gain is not None else ''
        #         buttons_info.append(f"{f'Button {i}':<10} {name:<15} {str(value):<10} {str(gain):<10}")
        #     else:
        #         buttons_info.append(f"{f'Button {i}':<10} {'':<15} {'':<10} {'':<10}")

        # buttons_str = '\n'.join(buttons_info)

        # Informations sur l'argent des joueurs
        # player_money = metrics.player_money
        # player_money_info = []
        # for player, money in player_money.items():
        #     money_str = str(round_sig(money)) if money is not None else 'Absent'
        #     player_money_info.append(f"{player}: {money_str}")

        # player_money_str = '\n'.join(player_money_info)

        return (
        #     f"Nombre de joueurs: {nbr_player}   Pot: {pot} €   Fond: {fond} €\n"
             f"Mes cartes: {me_cards_str}\n"
             f"Cartes sur le board: {board_cards_str}\n"
        #     f"Chance de gagner (1 joueur): {chance_win_0}\n"
        #     f"Chance de gagner ({nbr_player} joueurs): {chance_win_x}\n\n"
        #     f"Informations sur les boutons:\n{buttons_str}\n\n"
        #     f"Argent des joueurs:\n{player_money_str}"
         )

        return "Statistiques du jeu désactivées pour les tests de scan."



if __name__ == "__main__":
    controller = Controller()
    result = controller.main()
    print(result)
    
  # Sécurisation : on vérifie que table/scan/screen_array existent
    scan = getattr(controller.game.table, "scan", None)
    img = getattr(scan, "screen_array", None) if scan is not None else None
  

    import cv2
    import numpy as np
    from PIL import Image

    img = controller.game.table.scan.screen_array  # BGR

    if isinstance(img, np.ndarray):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).show()
    elif isinstance(img, Image.Image):
        img.show()
    else:
        print("Type d'image inattendu:", type(img))


    


    img = controller.game.table.scan.screen_crop  # BGR
    if img is None:
        print("Aucun crop de table disponible.")
    else:
        if isinstance(img, np.ndarray):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).show()
        elif isinstance(img, Image.Image):
            img.show()
        else:
            print("Type d'image inattendu:", type(img))

