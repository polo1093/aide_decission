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
- joue le rôle d’orchestrateur haut niveau :  
  - création et cycle de vie de `Game`,  
  - coordination entre  `Game` et `Décission`,  
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

Réfaire une séance de capture avec OBS . 
Refaire les paramétrages et le crop de l'écran, et tout ça, les screen au bon endroit . 
Ajoutez les boutons et le fond. 
Tester l'OCR. 

Ajouter une fonction pour détecter un truc bizarre qui est affiché à l'écran. 
Pour ajouter une fonction pour détecter si c'est à nous de jouer ou pas. 

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

Toutes les positions à l'écran sont définies dans le fichier `coordinates.json` à la racine du projet. Chaque région expose désormais directement la position absolue de son coin haut-gauche via `top_left` (en pixels écran) et s'appuie sur un gabarit (`templates`) pour déterminer sa taille. Mettez à jour ces coordonnées pour adapter le bot à une nouvelle résolution ou interface, puis relancez le programme pour appliquer les changements.

## Calibration absolue et point de référence

Toutes les captures sont désormais traitées en coordonnées **absolues**. Les
scripts recherchent automatiquement le point de référence défini par
`anchor.png` (ou `anchor.jpg`) afin de compenser un éventuel décalage de la
table sur l'écran. Ce mécanisme permet de travailler indifféremment avec des
captures plein écran ou des extraits cadrés sur la table :

- `table_capture.bounds` décrit les bornes absolues de la table et est calculé
  automatiquement si absent du JSON.
- `table_capture.ref_offset` indique la position attendue de l'ancre à l'intérieur
  de la table. Les scripts s'en servent pour corriger l'origine lorsque
  l'ancre est détectée.

## Capture de nouvelles cartes

1. **Extraire des captures plein écran** depuis une vidéo de calibration :

    ```bash
    python scripts/Crop_Video_Frames.py \
        --game-dir config/PMU \
        --video debug/cards_video/cards_video.mp4 \
        --out config/PMU/debug/screens
    ```

    Les images sont enregistrées dans `config/<game>/debug/screens` et servent
    de base aux étapes suivantes.

2. **Identifier les cartes manquantes** avec `identify_card.py` :

    ```bash
    python scripts/identify_card.py --game PMU --screens-dir config/PMU/debug/screens
    ```

    Le script applique automatiquement la correspondance sur l'ancre pour
    extraire chaque patch (valeur et symbole) avant de proposer une
    labellisation assistée.

3. **Valider la détection** à l'aide de `capture_cards.py` ou via
   `quick_setup.py`, qui enchaîne édition des zones, captures vidéo, matching
   et validation dynamique.

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