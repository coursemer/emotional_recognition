# 🎀 Application de Reconnaissance d'Émotions 🎀

Une application de reconnaissance d'émotions en temps réel qui utilise votre webcam pour détecter et afficher vos émotions avec une interface mignonne et girly !

## 📋 Table des Matières
- [Qu'est-ce que c'est ?](#quest-ce-que-cest-)
- [Prérequis](#prérequis)
- [Installation Étape par Étape](#installation-étape-par-étape)
- [Comment Lancer l'Application](#comment-lancer-lapplication)
- [Comment Utiliser l'Application](#comment-utiliser-lapplication)
- [Résolution des Problèmes](#résolution-des-problèmes)
- [Technologies Utilisées](#technologies-utilisées)

## Qu'est-ce que c'est ? 🤔

Cette application utilise l'intelligence artificielle pour :
1. **Détecter votre visage** via votre webcam
2. **Analyser votre expression faciale**
3. **Identifier votre émotion** parmi 7 catégories :
   - 😠 Angry (En colère)
   - 🤢 Disgust (Dégoût)
   - 😨 Fear (Peur)
   - 😊 Happy (Heureux)
   - 😢 Sad (Triste)
   - 😲 Surprise (Surpris)
   - 😐 Neutral (Neutre)

L'interface est stylée avec des couleurs roses et la police Times New Roman pour un look mignon !

## Prérequis 📝

Avant de commencer, vous devez avoir installé sur votre ordinateur :

### 1. Python 3.9
**Comment vérifier si vous l'avez déjà :**
```bash
python3 --version
```

Si vous voyez quelque chose comme `Python 3.9.x`, c'est bon ! ✅

**Si vous ne l'avez pas :**
- **Mac** : Téléchargez depuis [python.org](https://www.python.org/downloads/)
- **Windows** : Téléchargez depuis [python.org](https://www.python.org/downloads/)
- **Linux** : 
  ```bash
  sudo apt-get update
  sudo apt-get install python3.9
  ```

### 2. Git
**Comment vérifier si vous l'avez déjà :**
```bash
git --version
```

**Si vous ne l'avez pas :**
- **Mac** : Installez Xcode Command Line Tools
  ```bash
  xcode-select --install
  ```
- **Windows** : Téléchargez depuis [git-scm.com](https://git-scm.com/download/win)
- **Linux** :
  ```bash
  sudo apt-get install git
  ```

### 3. Une Webcam
Votre ordinateur doit avoir une webcam qui fonctionne ! 📷

## Installation Étape par Étape 🚀

### Étape 1 : Ouvrir le Terminal

**Sur Mac :**
1. Appuyez sur `Cmd + Espace`
2. Tapez "Terminal"
3. Appuyez sur Entrée

**Sur Windows :**
1. Appuyez sur `Windows + R`
2. Tapez "cmd"
3. Appuyez sur Entrée

**Sur Linux :**
1. Appuyez sur `Ctrl + Alt + T`

### Étape 2 : Choisir un Dossier

Décidez où vous voulez mettre le projet. Par exemple, sur le Bureau :

```bash
cd Desktop
```

💡 **Astuce** : `cd` signifie "change directory" (changer de dossier)

### Étape 3 : Télécharger le Projet

Copiez-collez cette commande dans votre terminal :

```bash
git clone https://github.com/coursemer/emotional_recognition.git
```

Appuyez sur Entrée et attendez. Vous verrez des messages défiler. C'est normal ! ✅

### Étape 4 : Entrer dans le Dossier du Projet

```bash
cd emotional_recognition
```

💡 **Vérification** : Tapez `ls` (Mac/Linux) ou `dir` (Windows) pour voir les fichiers. Vous devriez voir :
- `main.py`
- `requirements.txt`
- `download_model.py`
- etc.

### Étape 5 : Créer un Environnement Virtuel

Un environnement virtuel, c'est comme une bulle isolée pour votre projet. Ça évite les conflits avec d'autres programmes Python.

**Sur Mac/Linux :**
```bash
python3 -m venv venv
```

**Sur Windows :**
```bash
python -m venv venv
```

Attendez quelques secondes. Quand le terminal vous redonne la main, c'est bon ! ✅

### Étape 6 : Activer l'Environnement Virtuel

**Sur Mac/Linux :**
```bash
source venv/bin/activate
```

**Sur Windows :**
```bash
venv\Scripts\activate
```

💡 **Comment savoir si ça a marché ?** Vous devriez voir `(venv)` au début de votre ligne de commande, comme ça :
```
(venv) votre-nom@ordinateur:~/Desktop/emotional_recognition$
```

### Étape 7 : Installer les Dépendances

Les dépendances sont les bibliothèques dont le projet a besoin pour fonctionner.

```bash
pip install -r requirements.txt
```

⏰ **ATTENTION** : Cette étape peut prendre **5 à 10 minutes** ! C'est normal, il télécharge beaucoup de choses (TensorFlow est très gros).

Vous verrez plein de texte défiler. Attendez jusqu'à ce que vous voyiez :
```
Successfully installed ...
```

## Comment Lancer l'Application 🎬

### Étape 1 : Assurez-vous que l'Environnement Virtuel est Activé

Vous devez voir `(venv)` au début de votre ligne de commande.

**Si vous ne le voyez pas**, retournez à l'Étape 6 de l'installation.

### Étape 2 : Lancer le Programme

```bash
python main.py
```

**Sur certains systèmes, vous devrez peut-être utiliser :**
```bash
python3 main.py
```

### Étape 3 : Attendre le Chargement

⏰ **Le premier lancement prend 30-60 secondes** car il charge le modèle d'IA.

Vous verrez des messages comme :
```
[2025-12-04 23:55:00] Démarrage de l'application de reconnaissance d'émotions
[2025-12-04 23:55:01] Chargement du modèle et du détecteur de visage...
[2025-12-04 23:55:05] Modèle chargé avec succès
[2025-12-04 23:55:05] Webcam initialisée avec succès
[2025-12-04 23:55:05] Démarrage de la détection d'émotions. Appuyez sur 'q' pour quitter.
```

### Étape 4 : Une Fenêtre s'Ouvre !

Une fenêtre appelée **"Reconnaissance des Émotions"** devrait s'ouvrir avec :
- L'image de votre webcam
- Un rectangle rose autour de votre visage
- Le nom de votre émotion en rose au-dessus de votre tête
- Le pourcentage de confiance

## Comment Utiliser l'Application 🎯

### Positionnement

Pour que l'application détecte bien votre visage :

1. **Placez-vous face à la caméra** (pas de profil)
2. **Assurez-vous d'avoir assez de lumière** 💡
3. **Restez dans le cadre central** de la webcam
4. **Gardez votre visage visible** (pas de mains devant, pas de masque)

### Faire des Expressions

Essayez différentes expressions faciales :
- 😊 **Souriez** → devrait détecter "happy"
- 😐 **Visage neutre** → devrait détecter "neutral"
- 😢 **Faites une tête triste** → devrait détecter "sad"
- 😲 **Ouvrez grand la bouche** → devrait détecter "surprise"

### Quitter l'Application

**Appuyez sur la touche `q`** de votre clavier (comme "quitter").

La fenêtre se fermera et vous verrez des statistiques dans le terminal :
```
[2025-12-04 23:58:00] Arrêt de l'application. Temps d'exécution: 180.00 secondes
[2025-12-04 23:58:00] Nombre total de frames traitées: 1200
[2025-12-04 23:58:00] FPS moyen: 6.67
```

## Résolution des Problèmes 🔧

### Problème : "command not found: python3"

**Solution :** Essayez avec `python` au lieu de `python3`
```bash
python --version
python main.py
```

### Problème : "No module named 'cv2'"

**Solution :** L'environnement virtuel n'est pas activé ou les dépendances ne sont pas installées.

1. Activez l'environnement virtuel (voir Étape 6)
2. Réinstallez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

### Problème : "Cannot open camera"

**Solutions possibles :**

1. **Une autre application utilise la webcam** (Zoom, Skype, etc.)
   - Fermez toutes les applications qui utilisent la webcam
   
2. **Permissions de la webcam**
   - **Mac** : Allez dans Préférences Système → Sécurité et confidentialité → Caméra
   - Autorisez Terminal ou votre application à accéder à la caméra

3. **Webcam externe déconnectée**
   - Vérifiez que votre webcam USB est bien branchée

### Problème : "L'application ne détecte pas mon visage"

**Solutions :**

1. **Ajoutez plus de lumière** 💡
2. **Rapprochez-vous de la caméra**
3. **Centrez votre visage dans l'image**
4. **Enlevez les lunettes de soleil, chapeaux, masques**
5. **Assurez-vous que votre visage est de face** (pas de profil)

### Problème : L'application est très lente

**C'est normal !** L'analyse d'émotions en temps réel demande beaucoup de calculs.

**FPS attendus :** 5-10 images par seconde sur un ordinateur normal.

Si c'est vraiment trop lent (moins de 3 FPS), fermez les autres applications.

### Problème : "ImportError: numpy"

**Solution :** Réinstallez numpy
```bash
pip uninstall -y numpy
pip install numpy==1.23.5
```

## Technologies Utilisées 🛠️

| Technologie | Version | Rôle |
|------------|---------|------|
| **Python** | 3.9 | Langage de programmation principal |
| **OpenCV** | 4.8.0.74 | Détection de visages et traitement d'images |
| **TensorFlow** | 2.12.0 | Intelligence artificielle pour la reconnaissance d'émotions |
| **Pillow** | 11.3.0 | Rendu de texte avec police personnalisée |
| **NumPy** | 1.23.5 | Calculs mathématiques |

### Le Modèle d'IA

Le modèle utilisé s'appelle **fer2013_mini_XCEPTION** :
- Entraîné sur le dataset FER2013 (35,000 images de visages)
- Architecture : Mini-Xception (version légère de Xception)
- Précision : ~66% sur le dataset de test

## Structure du Projet 📁

```
emotional_recognition/
│
├── main.py                              # Programme principal
├── download_model.py                    # Script pour télécharger le modèle
├── requirements.txt                     # Liste des dépendances
├── .gitignore                          # Fichiers à ignorer par Git
│
├── fer2013_mini_XCEPTION.102-0.66.hdf5 # Modèle d'IA pré-entraîné
├── haarcascade_frontalface_default.xml # Détecteur de visages
│
└── venv/                               # Environnement virtuel (créé par vous)
    └── ...
```

## Commandes Utiles 📝

### Désactiver l'Environnement Virtuel
```bash
deactivate
```

### Réactiver l'Environnement Virtuel
**Mac/Linux :**
```bash
source venv/bin/activate
```

**Windows :**
```bash
venv\Scripts\activate
```

### Mettre à Jour les Dépendances
```bash
pip install --upgrade -r requirements.txt
```

### Voir les Dépendances Installées
```bash
pip list
```

## FAQ ❓

**Q : Mes données sont-elles envoyées quelque part ?**  
R : Non ! Tout se passe localement sur votre ordinateur. Rien n'est envoyé sur Internet.

**Q : Puis-je utiliser cette application sans Internet ?**  
R : Oui, une fois installée ! L'installation nécessite Internet pour télécharger les dépendances, mais ensuite l'application fonctionne hors ligne.

**Q : L'application enregistre-t-elle des vidéos ou photos de moi ?**  
R : Non, rien n'est enregistré. L'application analyse les images en temps réel et les oublie immédiatement.

**Q : Pourquoi la détection n'est pas toujours précise ?**  
R : L'IA n'est pas parfaite ! La précision dépend de :
- La qualité de la lumière
- L'angle de votre visage
- La clarté de votre expression
- Les limites du modèle (66% de précision)

**Q : Puis-je changer les couleurs de l'interface ?**  
R : Oui ! Modifiez les lignes 214-215 dans `main.py` :
```python
pink_color = (255, 105, 180)  # Changez ces valeurs RGB
pastel_pink = (255, 182, 193)
```

**Q : Puis-je utiliser une autre police ?**  
R : Oui ! Modifiez la ligne 202 dans `main.py` avec le chemin vers votre police :
```python
font = ImageFont.truetype("/chemin/vers/votre/police.ttf", 32)
```

## Auteurs 👥

Développé avec ❤️ pour le projet de reconnaissance d'émotions.

## Licence 📄

Ce projet utilise des modèles et bibliothèques open-source. Voir les licences individuelles de chaque dépendance.

---

**Besoin d'aide ?** Ouvrez une issue sur GitHub ou contactez votre binôme ! 😊