# aide de décision



Plan (reformulé)



Charger config/<game>/coordinates.json.


To do

faire un script qui unifie les mains de position_zones_ctk puis Crop_Video_Frames puis identify_cards puis capture_cards




Étape 4 : Moteur d’aide à la décision

Interpréter l’état du jeu : à partir des cartes reconnues (ex: cartes communes et main du joueur dans une partie de poker), construire la représentation de l’état de la partie. Intégrer d’autres informations si nécessaire (mises, positions des joueurs, etc. – éventuel pour d’autres jeux).

Calculer les probabilités ou scores : implémenter la logique de calcul (par ex. pour le poker, évaluer la force de la main, calculer les probabilités de gagner, etc., ou toute métrique utile à la décision pour le jeu en question).

Générer une recommandation : en se basant sur ces calculs, formuler une suggestion d’action optimale ou une aide à la décision (par ex. « relancer », « se coucher » pour le poker, ou toute recommandation spécifique au jeu).

Interface de sortie : Prévoir comment afficher ou retourner cette aide à la décision à l’utilisateur (console, interface graphique, overlay sur le jeu, etc.).

Tests: Valider le moteur de décision avec des scénarios connus (par ex. mains de poker prédéfinies où l’issue est connue, pour vérifier que l’évaluation et la recommandation correspondent aux attentes).



## Installation

1. Cloner le dépôt.
2. Installer Python 3.9 ou version supérieure.
3. Installer les dépendances :

```bash
pip install -r requirements.txt
```

Assurez‑vous que Tesseract est installé sur votre système pour que l'OCR fonctionne correctement.
Si l'exécutable n'est pas détecté automatiquement, définissez la variable
d'environnement `TESSERACT_CMD` avec le chemin complet vers `tesseract` :

```bash
export TESSERACT_CMD=/usr/bin/tesseract
```

## Utilisation

Lancer l'interface de démonstration :

```bash
python test.py
```

Une fenêtre permet de lancer le scan de la table. Les informations détectées s'affichent en temps réel et le bot peut cliquer sur l'action proposée.

## Modifier les coordonnées

Toutes les positions à l'écran sont définies dans le fichier `coordinates.json` à la racine du projet. Chaque entrée contient les coordonnées relatives d'une région sous la forme `[x1, y1, x2, y2]`. Modifiez ces valeurs pour adapter le bot à une nouvelle résolution ou interface, puis relancez le programme pour appliquer les changements.

## Capture de nouvelles cartes

Le script `scripts/capture_cards.py` aide à constituer un jeu d'images pour l'OCR.
Spécifiez les noms de régions définis dans `coordinates.json` ou des coordonnées
absolues pour découper la valeur et le symbole d'une carte :

```bash
python scripts/capture_cards.py \
    --number player_card_1_number \
    --symbol player_card_1_symbol \
    --ref 1000,200
```

Si le test automatique détecte au moins dix pixels blancs en bas à droite, le
programme demande la valeur et la couleur de la carte. Les images sont alors
enregistrées dans `screen/debug/Carte/<valeur>/` et `screen/symbole/<couleur>/`.

## Copier les sources Python

Le script `scripts/copy_python_sources.py` regroupe tout le code `.py`
du projet (à l'exception de ce script) et le place dans le presse-papiers.
Installez `pyperclip` si nécessaire, puis exécutez :

```bash
python scripts/copy_python_sources.py
```

Vous pouvez ignorer d'autres fichiers avec `--exclude chemin/vers/fichier.py`.

## Utilitaires de calibration partagés

Les scripts de calibration (`capture_cards.py`, `identify_card.py`, `position_zones*.py`,
`zone_project.py`) s'appuient désormais sur un module commun `scripts/_utils.py`.
Ce module centralise :

- le chargement de `coordinates.json` (résolution des `templates`, accès à
  `table_capture`, conversion robuste des entiers) ;
- les fonctions de clamp et d'extraction d'images utilisées par les différents
  CLI/UI.

Les interfaces en ligne de commande existantes ne changent pas : les mêmes
options et arguments continuent de fonctionner, avec un comportement aligné
entre tous les outils.

## Avertissement

Ce projet est fourni à titre expérimental. L'utilisation d'un bot sur des plateformes de poker en ligne peut être interdite par leurs conditions d'utilisation. L'auteur décline toute responsabilité en cas d'usage inapproprié.

