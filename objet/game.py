from pokereval.card import Card
from pokereval.hand_evaluator import HandEvaluator
import warnings
import logging
import tool

from dataclasses import dataclass, field
from typing import Optional
from objet import bouton


@dataclass
class Game:
    """Stocke l'état courant de la table et calcule les décisions."""

    data: Optional[dict] = None
    etat: dict = field(
        default_factory=lambda: {
            "Nbr_player": 5,
            "pot": 0,
            "button_1": bouton.Bouton(),
            "button_2": bouton.Bouton(),
            "button_3": bouton.Bouton(),
            "board": [None, None, None, None, None],
            "me_card": [None, None],
            "chance_win_0": None,
            "chance_win_x": None,
        }
    )
    resultat_calcul: dict = field(default_factory=dict)
    
    def scan_to_data_table(self,scan_table):
        """
        Transforme le dictionnaire du résultat du scan en données structurées et actualise self.etat.
        
        Args:
            scan_table (dict): Dictionnaire contenant l'état actuel de la table de poker.
        """

        # Convertir les cartes du board
        for i in range(1, 6):
            number_key = f'board_card_{i}_number'
            symbol_key = f'board_card_{i}_symbol'
            if number_key in scan_table and symbol_key in scan_table:
                number = scan_table[number_key]['value']
                symbol = scan_table[symbol_key]['value']
                self.etat["board"][i-1] = convert_card(f"{number}{symbol}") if number and symbol else None
        
        # Convertir les cartes du joueur
        for i in range(1, 3):
            number_key = f'player_card_{i}_number'
            symbol_key = f'player_card_{i}_symbol'
            if number_key in scan_table and symbol_key in scan_table:
                number = scan_table[number_key]['value']
                symbol = scan_table[symbol_key]['value']
                self.etat["me_card"][i-1] = convert_card(f"{number}{symbol}") if number and symbol else None
        
        # Mettre à jour pot et fond
        self.etat["pot"] = tool.convert_to_float(scan_table["pot"]['value'])
        self.etat["fond"] = tool.convert_to_float(scan_table["fond"]['value']) 
        
        # Traiter les boutons
        for i in range(1, 4):
            button_key = f'button_{i}'
            self.etat[button_key].string_to_bouton(scan_table[button_key]['value'])
        
        # Mettre à jour l'argent des joueurs
        self.etat["player_money"] = {
            "J1": tool.convert_to_float(scan_table["player_money_J1"]['value']) ,
            "J2": tool.convert_to_float(scan_table["player_money_J2"]['value']) ,
            "J3": tool.convert_to_float(scan_table["player_money_J3"]['value']) ,
            "J4": tool.convert_to_float(scan_table["player_money_J4"]['value']) ,
            "J5": tool.convert_to_float(scan_table["player_money_J5"]['value']) 
        }
         # Mettre à jour le nombre de joueurs si nécessaire
        self.etat["Nbr_player"] = self._nbr_player()
        
      
    def decision(self,scan_table):
        self.scan_to_data_table(scan_table)
        if self.etat['me_card'] == ['_', '_'] or self.etat['me_card'] == [None, None]:
            return None
        
        self._calcul_chance_win()
        max_gain = -1
        best_bouton = None
        for bouton in ['button_1','button_2','button_3']:
            if self.etat[bouton].value != None and self.etat[bouton].name != None:
               gain=self._calucul_EV(self.etat["chance_win_x"],self.etat[bouton].value)
            elif self.etat[bouton].name=="se coucher":
               gain=0    
            elif self.etat[bouton].name != None:
                gain=self._calucul_EV(self.etat["chance_win_x"],0)
            else:
                gain = None
            self.etat[bouton].gain=gain
            if gain is not None and gain > max_gain:
                max_gain = gain
                best_bouton = bouton
        return best_bouton
                
    def _calcul_chance_win(self):
        # Filtrer les cartes valides
        me_cards = [card for card in self.etat['me_card'] if card is not None]
        board_cards = [card for card in self.etat['board'] if card is not None]
        
        # Vérifier que le joueur a bien deux cartes
        if len(me_cards) != 2:
            raise ValueError("Les cartes du joueur ne sont pas complètes ou invalides.")
        
        # Vérifier que le nombre de cartes sur le board est correct (0, 3, 4 ou 5)
        if len(board_cards) not in [0, 3, 4, 5]:
            raise ValueError("Le nombre de cartes sur le board est incorrect.")
        
        # Évaluer la main du joueur
        self.etat["chance_win_0"] = HandEvaluator.evaluate_hand(me_cards, board_cards)
        self.etat["chance_win_x"] = self.etat["chance_win_0"] ** self.etat['Nbr_player']
        pass

        
        

    def _calucul_EV(self,chance_win,mise):
        r = chance_win * (self.etat["pot"] + (mise*(self.etat["Nbr_player"]+1))) - (1 - chance_win) * mise #ne prends pas en compte si les joueurs ont déjà payer
        return r

    def _nbr_player(self):
        """Compte le nombre de joueurs présents en se basant sur self.etat['player_money']. Les joueurs avec une valeur None ou zéro sont considérés comme absents."""
        return sum(1 for money in self.etat["player_money"].values() if money not in [None, 0])









def convert_card(string_carte: str):
    """
    Convertit une chaîne de caractères représentant une carte de poker en un objet Card de la bibliothèque pokereval.

    Args:
        string_carte (str): une chaîne de caractères représentant une carte de poker.

    Returns:
        Card: un objet Card représentant la carte de poker, ou None si la carte n'est pas reconnue.
    """
    suit_dict  = {"♦": 1, "♥": 2, "♠": 3, "♣": 4}
    value_dict  = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                   '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

    if string_carte in [None, "", "_"]:
        return None

    # Correction des erreurs d'OCR
    if string_carte[0] == '0':
        message = f"Debug : La carte spécifiée '{string_carte}' est modifiée en '10{string_carte[1:]}' pour correction."
        string_carte = 'Q' + string_carte[1:]
        logging.debug(message)

    # Gérer les cartes à deux ou plusieurs caractères
    if len(string_carte) >= 2:
        value_part = string_carte[:-1]
        suit_part = string_carte[-1]
        if value_part in value_dict and suit_part in suit_dict:
            value = value_dict[value_part]
            suit = suit_dict[suit_part]
        else:
            message = f"Debug : La carte spécifiée '{string_carte}' n'est pas reconnue."
            logging.debug(message)
            return None
    else:
        message = f"Debug : La carte spécifiée '{string_carte}' est trop courte."
        logging.debug(message)
        return None

    return Card(value, suit)




