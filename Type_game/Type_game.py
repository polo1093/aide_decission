
class GameBase:
    def __init__(self):
        # Définissez ici les constantes communes à tous les jeux
        pass
        
        # ... d'autres constantes communes ...
        
        
    def get_all_coordinates(self):
        coordinates = []
        # Parcourir self.card_scans
        for key, card_list in self.card_scans.items():
            for card in card_list:
                coordinates.append(card.number_position)
                coordinates.append(card.color_position)
        # Parcourir self.money_scans
        for key, money in self.money_scans.items():
            coordinates.append(money.position)
        return coordinates
    
    def _init_scans_money(self, positions_dict, scan_class):
        scans = {key: scan_class(positions)  for key, positions in positions_dict.items()}
        return scans
    
    def _init_scans_card(self, positions_dict, scan_class):
        scans = {}
        for key, positions in positions_dict.items():
            scans[key] = [scan_class(pos["number"], pos["color"]) for pos in positions]
        return scans
    
    def position_card(self):
        return{
            'board': [
                {
                    "number": [x, y, x+self.SIZE_CARD_NUMBER[0], y+self.SIZE_CARD_NUMBER[1]],
                    "color": [x+self.SIZE_CARD_COLOR[1], y+self.SIZE_CARD_COLOR[2], x+self.SIZE_CARD_COLOR[1]+self.SIZE_CARD_COLOR[0], y+self.SIZE_CARD_COLOR[2]+self.SIZE_CARD_COLOR[3]]
                }
                for x, y in self._POSITION_CARD_RELATIF['board']
            ],
            'me_card': [
                {
                    "number": [x, y, x+self.SIZE_CARD_NUMBER[0], y+self.SIZE_CARD_NUMBER[1]],
                    "color": [x+self.SIZE_CARD_COLOR[1], y+self.SIZE_CARD_COLOR[2], x+self.SIZE_CARD_COLOR[1]+self.SIZE_CARD_COLOR[0], y+self.SIZE_CARD_COLOR[2]+self.SIZE_CARD_COLOR[3]]
                }
                for x, y in self._POSITION_CARD_RELATIF['me_card']
            ],}
        
        
        
class Card_scan():
    def __init__(self, number_position, color_position):
        self.number_position = number_position  # [x1,y1,x2,y2]
        self.color_position = color_position  # [x1,y1,x2,y2]

class Money_Scan():
    def __init__(self, position):
        self.position = position  # [x1,y1,x2,y2]



class PMU_Poker(GameBase):
    def __init__(self):
        super().__init__()  # Appel du constructeur de la classe de base pour récupérer les constantes communes
        # Définissez ici les constantes spécifiques au PMU Poker
        self.POSITION_POINT_WHITE = [50, 20] # Pour savoir s'il y une card ou pas
        self.SIZE_CARD_COLOR = [50, 0, 51, 54]  # [largeur, décalage largeur,décalage hauteur, hauteur] 40 65
        self.SIZE_CARD_NUMBER = [50, 60]  # [largeur, hauteur] 65
        self._POSITION_CARD_RELATIF = {'board': [(-274, -607), (-131, -607), (13, -607), (158, -607), (303, -607)],
                        'me_card': [(-54, -186), (80, -186)]}

        self._POSITION_MONEY = {
            'pot': [102, -657, 177, -617],
            'fond': [17, 52, 102, 92],
            'bouton_1': [284, 65, 449, 135],
            'bouton_2': [521, 65, 686, 135],
            'bouton_3': [764, 65, 929, 135],
            'J1': [-684, -211, -599, -171],
            'J2': [-617, -669, -532, -629],
            'J3': [21, -866, 106, -826],
            'J4': [660, -669, 745, -629],
            'J5': [726, -211, 811, -171]
        }
        self._POSITION_CARD = self.position_card()
        
        self.card_scans = self._init_scans_card(self._POSITION_CARD, Card_scan)
        self.money_scans = self._init_scans_money(self._POSITION_MONEY, Money_Scan)
        







class Jeu2(GameBase):
    def __init__(self):
        super().__init__()  # Appel du constructeur de la classe de base pour récupérer les constantes communes
        # Définissez ici les constantes spécifiques au jeu 2
        self.SOME_GAME2_CONSTANT = "Bonjour"
        # ... d'autres constantes spécifiques au jeu 2 ...


