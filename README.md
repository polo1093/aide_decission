# aide de décision

PUML manuelle 

## Architecture globale

L’application est structurée autour d’un noyau `Game` qui orchestre l’état de la table, des joueurs, des boutons et des cartes, ainsi que la logique de partie et de décision.

### 1. Game

1.1. **Table**  
&nbsp;&nbsp;&nbsp;&nbsp;1.1.1. `scan_table`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.1.1.1. (ajout du scan *fond* et *pot* en OCR)  
&nbsp;&nbsp;&nbsp;&nbsp;1.1.2. `Card_state`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.1.2.1. `Card` (regrouper toutes les classes d’entities carte en une seule)  
&nbsp;&nbsp;&nbsp;&nbsp;1.1.3. `Buttons_state`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.1.3.1. `Button`  
&nbsp;&nbsp;&nbsp;&nbsp;1.1.4. `Player_state`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.1.4.1. `Player`

1.2. **Party**  
&nbsp;&nbsp;&nbsp;&nbsp;1.2.1. Gestion des états de la partie (phase de jeu, street, main en cours, historique, etc.)

1.3. **Décission**  
&nbsp;&nbsp;&nbsp;&nbsp;Moteur de décision basé sur l’état courant du `Game` / `Table` / `Player_state` (rules, heuristiques, modèle ML, etc.)

### 2. Contrôleur

Bloc à challenger :  
- Question ouverte : **le contrôleur est-il toujours utile ?**  
- Si conservé, il joue le rôle d’orchestrateur haut niveau :  
  - création et cycle de vie de `Game`,  
  - coordination entre `scan_table`, `Party` et `Décission`,  
  - gestion des événements externes (UI, hotkeys, logs, etc.).

### 3. Afficheur / `launch.py` + Thinker

- `launch.py` sert de point d’entrée applicatif.  
- Rôle principal :  
  - initialiser le “thinker” (boucle principale d’analyse/decision),  
  - câbler l’affichage (console, UI, overlay…) avec l’état de `Game` / `Table`,  
  - piloter la fréquence des scans (`scan_table`) et des décisions.

Ce schéma sert de référence pour l’implémentation et pour organiser les modules Python (fichiers et packages) selon cette hiérarchie logique.


flowchart LR
  A[config/coordinates.json] --> B[Capture/Screen Grab] fait
  B --> C[Crop & Pré-traitement] fait
  C --> D[OCR / Matching] en cours
  D --> E[État du jeu] à faire
  E --> F[Moteur d'aide à la décision] à faire
  F --> G[Sorties: console/UI/overlay] en cours





To do list 




1.faire un script qui unifie les mains de position_zones_ctk puis Crop_Video_Frames puis identify_cards puis capture_cards, pr une configuratiion rapides

2.






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

## Quickstart
```bash
git clone https://github.com/polo1093/aide_decission.git
cd aide_decission
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Installez Tesseract puis définissez TESSERACT_CMD si nécessaire
# Windows: setx TESSERACT_CMD "C:\Program Files\Tesseract-OCR\tesseract.exe"
python launch.py --profile demo                 # Exemple d’exécution
## Modifier les coordonnées

Toutes les positions à l'écran sont définies dans le fichier `coordinates.json` à la racine du projet. Chaque entrée contient les coordonnées relatives d'une région sous la forme `[x1, y1, x2, y2]`. Modifiez ces valeurs pour adapter le bot à une nouvelle résolution ou interface, puis relancez le programme pour appliquer les changements.

## Capture de nouvelles cartes

Le script `scripts/capture_cards.py` aide à constituer un jeu d'images pour l'OCR.
Spécifiez les noms de régions définis dans `coordinates.json` ou des coordonnées
absolues pour découper la valeur et le symbole d'une carte :

```bash
python scripts/capture_cards.py 
    --number player_card_1_number 
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

Ce projet est fourni à titre expérimental.