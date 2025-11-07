# aide de décision

To DO list: 

Étape 1 : Recadrage du plateau de jeu et configuration initiale

Définir une structure de configuration (fichier JSON par jeu, ex: jeu1.json) pour stocker les coordonnées des zones du jeu (plateau, emplacements des cartes, etc.).

Capturer un screenshot de la fenêtre du jeu (nommé par exemple jeu1.png pour le jeu en cours).

Déterminer manuellement la zone du plateau sur le screenshot (la portion de l’image correspondant à la table de jeu, sans les bordures/fenêtres du système).

Enregistrer les coordonnées de recadrage de la table dans le JSON de configuration du jeu (pour pouvoir cropper l’image du jeu à l’avenir).

Prévoir l’extensibilité multi-jeux : s’assurer que la structure JSON et le code pourront gérer facilement d’autres jeux (noms de fichiers/config différents).

    Tests: Écrire des tests unitaires pour valider le recadrage : par exemple, vérifier qu’en fournissant une image d’écran et des coordonnées de découpe, le programme obtient bien l’image recadrée attendue.

Étape 2 : Outil interactif d’ajustement des zones de cartes

Développer un script Python d’étalonnage qui affiche le screenshot recadré du plateau et permet de placer/déplacer des rectangles représentant les zones importantes (emplacements de cartes, etc.).

Positionner les zones des cartes : ajouter des rectangles ajustables pour chaque carte sur la table (ex: les 5 cartes communes au centre pour le poker, les cartes des joueurs, etc.).

Configurer les sous-zones des cartes : pour chaque emplacement de carte, définir également les zones du numéro (valeur) et du symbole (suit) sur la carte. Ces sous-zones pourront être définies pour une carte type et reproduites sur les autres si la disposition est identique.

Gestion des groupes de zones : implémenter une logique pour lier certains rectangles entre eux. Par exemple, les 5 zones de cartes du centre doivent garder la même taille et un espacement constant – ajuster un rectangle doit redimensionner/déplacer les autres en conséquence (afin de conserver un alignement régulier).

Ajustement de l’espacement : permettre de modifier l’écart entre les cartes groupées (p. ex. en déplaçant une carte tout en maintenant une touche pour ajuster uniformément l’espace entre toutes).

Interface utilisateur : permettre de valider une fois le placement terminé (par ex. bouton ou touche pour sauvegarder). Le programme doit alors enregistrer toutes les coordonnées calibrées dans le fichier JSON du jeu correspondant.

Tests: Vérifier la logique de groupement par des tests unitaires (par ex. simuler le redimensionnement d’une carte et vérifier que les 4 autres cartes centrales ont bien adopté la même taille et que l’espacement est cohérent).

Étape 3 : Reconnaissance des cartes sur l’image

Implémenter l’extraction des cartes : à l’aide des coordonnées définies dans le JSON, extraire automatiquement chaque zone de carte du screenshot de la table (découper l’image de chaque carte à partir de l’écran recadré).

Identifier la valeur et la couleur : analyser chaque extrait de carte pour reconnaître le numéro (ou figure) et le symbole (♥♣♦♠ par ex. pour un jeu de cartes). Cela peut se faire via de la reconnaissance d’image (template matching) ou OCR pour les caractères, selon le cas.

Base de références : Préparer une collection d’images de référence pour chaque numéro/figure et chaque symbole, ou entraîner un modèle, afin de comparer et déterminer la carte affichée.

Lecture des cartes du jeu : Combiner les résultats (numéro + symbole) pour déterminer la carte complète (par ex. "AS de cœur"). Répéter pour toutes les cartes détectées sur la table et en main des joueurs le cas échéant.

    Tests: Pour chaque étape de reconnaissance, écrire des tests unitaires avec des images exemples (par ex. vérifier que l’extraction d’une carte à partir d’une image connue donne bien la bonne sous-image, ou que la reconnaissance identifie correctement une carte donnée à partir d’un échantillon d’image).

Étape 4 : Moteur d’aide à la décision

Interpréter l’état du jeu : à partir des cartes reconnues (ex: cartes communes et main du joueur dans une partie de poker), construire la représentation de l’état de la partie. Intégrer d’autres informations si nécessaire (mises, positions des joueurs, etc. – éventuel pour d’autres jeux).

Calculer les probabilités ou scores : implémenter la logique de calcul (par ex. pour le poker, évaluer la force de la main, calculer les probabilités de gagner, etc., ou toute métrique utile à la décision pour le jeu en question).

Générer une recommandation : en se basant sur ces calculs, formuler une suggestion d’action optimale ou une aide à la décision (par ex. « relancer », « se coucher » pour le poker, ou toute recommandation spécifique au jeu).

Interface de sortie : Prévoir comment afficher ou retourner cette aide à la décision à l’utilisateur (console, interface graphique, overlay sur le jeu, etc.).

Tests: Valider le moteur de décision avec des scénarios connus (par ex. mains de poker prédéfinies où l’issue est connue, pour vérifier que l’évaluation et la recommandation correspondent aux attentes).


## Principales fonctionnalités

- **Capture et analyse de l'écran** : le module `objet.scan` identifie les cartes du board et les mises via `pytesseract`.
- **Modélisation de la partie** : `objet.game` calcule les probabilités de gain avec `pokereval` et détermine l'action optimale.
- **Automatisation des clics** : `objet.cliqueur` simule les clics sur l'interface pour miser ou se coucher.
- **Interface de test** : `test.py` lance une petite fenêtre PySimpleGUI permettant d'activer le scan et d'afficher les informations récupérées.

Les positions des différents éléments à l'écran sont maintenant stockées dans `coordinates.json` et les images de référence sont dans le dossier `screen`.

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

## Avertissement

Ce projet est fourni à titre expérimental. L'utilisation d'un bot sur des plateformes de poker en ligne peut être interdite par leurs conditions d'utilisation. L'auteur décline toute responsabilité en cas d'usage inapproprié.

