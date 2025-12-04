import cv2
import numpy as np
import os
import time
import requests
from datetime import datetime
from pathlib import Path
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw, ImageFont

def log_message(message):
    """Affiche un message de log avec un timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def download_file(url, filename):
    """Télécharge un fichier depuis une URL et le sauvegarde localement"""
    try:
        log_message(f"Téléchargement de {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Créer le répertoire s'il n'existe pas
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        
        # Télécharger le fichier par blocs
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar_size = 50
        
        with open(filename, 'wb') as f:
            downloaded_size = 0
            for data in response.iter_content(block_size):
                f.write(data)
                downloaded_size += len(data)
                if total_size > 0:
                    progress = min(100, int((downloaded_size / total_size) * 100))
                    filled = int(progress_bar_size * downloaded_size // total_size)
                    bar = '█' * filled + '-' * (progress_bar_size - filled)
                    # Afficher la taille téléchargée sur la taille totale avec unités
                    size_units = ['B', 'KB', 'MB', 'GB']
                    size_index = 0
                    display_size = total_size
                    while display_size > 1024 and size_index < len(size_units) - 1:
                        display_size /= 1024.0
                        size_index += 1
                    total_size_str = f"{display_size:.1f} {size_units[size_index]}"
                    
                    current_display_size = downloaded_size
                    current_size_index = 0
                    while current_display_size > 1024 and current_size_index < len(size_units) - 1:
                        current_display_size /= 1024.0
                        current_size_index += 1
                    current_size_str = f"{current_display_size:.1f} {size_units[current_size_index]}"
                    
                    print(f'\r[{bar}] {progress:3d}% ({current_size_str} / {total_size_str})', end='')
            print()  # Nouvelle ligne après la barre de progression
            
        log_message(f"{filename} téléchargé avec succès")
        return True
    except Exception as e:
        log_message(f"Erreur lors du téléchargement de {filename}: {e}")
        return False

def check_and_download_files():
    """Vérifie et télécharge les fichiers nécessaires s'ils ne sont pas présents"""
    files_to_download = [
        (MODEL_URL, MODEL_FILE),
        (CASCADE_URL, FACE_DETECTOR_PATH)
    ]
    
    all_files_available = True
    
    for url, filename in files_to_download:
        if not os.path.exists(filename):
            log_message(f"Fichier manquant: {filename}")
            if not download_file(url, filename):
                all_files_available = False
    
    return all_files_available

# --- Configuration ---
MODEL_URL = 'https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
CASCADE_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'

MODEL_FILE = 'fer2013_mini_XCEPTION.102-0.66.hdf5'
FACE_DETECTOR_PATH = 'haarcascade_frontalface_default.xml'
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
WEBCAM_INDEX = 0

log_message("Démarrage de l'application de reconnaissance d'émotions")

# Vérification et téléchargement des fichiers nécessaires
if not check_and_download_files():
    log_message("Erreur: Impossible de télécharger tous les fichiers nécessaires.")
    exit(1)

# --- Chargement du modèle et du détecteur de visage ---
log_message("Chargement du modèle et du détecteur de visage...")
try:
    log_message(f"Chargement du modèle depuis : {MODEL_FILE}")
    classifier = load_model(MODEL_FILE)
    log_message("Modèle chargé avec succès")
    
    log_message(f"Chargement du classificateur de visage depuis : {FACE_DETECTOR_PATH}")
    face_detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
    log_message("Détecteur de visage chargé avec succès")
except Exception as e:
    log_message(f"ERREUR lors du chargement des fichiers: {e}")
    log_message("Veuillez vous assurer que 'fer2013_mini_XCEPTION.102-0.66.hdf5' et 'haarcascade_frontalface_default.xml' sont dans le même répertoire.")
    exit()

# --- Initialisation de la webcam ---
log_message(f"Initialisation de la webcam (index: {WEBCAM_INDEX})...")
camera = cv2.VideoCapture(WEBCAM_INDEX)

if not camera.isOpened():
    log_message(f"ERREUR: Impossible d'ouvrir la webcam à l'index {WEBCAM_INDEX}")
    exit()

log_message("Webcam initialisée avec succès")
log_message("Démarrage de la détection d'émotions. Appuyez sur 'q' pour quitter.")
log_message("--------------------------------------------------")

frame_count = 0
start_time = time.time()

while True:
    frame_count += 1
    # Capture de l'image
    ret, frame = camera.read()
    if not ret:
        log_message("ERREUR: Impossible de capturer l'image. Arrêt...")
        break
        
    if frame_count % 30 == 0:  # Toutes les 30 frames
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        log_message(f"Traitement de la frame #{frame_count} - FPS: {fps:.2f}")

    # Flip horizontalement l'image de la caméra pour un effet miroir
    frame = cv2.flip(frame, 1)
    
    # Convertir en niveaux de gris pour la détection des visages
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détection des visages dans l'image
    faces = face_detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,  # Réduit pour détecter plus facilement
        minSize=(60, 60),  # Taille minimale réduite
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Filtrer les visages détectés
    filtered_faces = []
    frame_height, frame_width = frame.shape[:2]
    
    for (x, y, w, h) in faces:
        # Calculer le centre du visage
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Vérifier si le visage est dans une zone acceptable (80% de l'image)
        center_margin_x = frame_width * 0.1
        center_margin_y = frame_height * 0.1
        
        is_in_frame = (center_x > center_margin_x and 
                      center_x < (frame_width - center_margin_x) and
                      center_y > center_margin_y and 
                      center_y < (frame_height - center_margin_y))
        
        # Vérifier la luminosité moyenne du visage (seuil plus bas)
        face_roi = gray_frame[y:y+h, x:x+w]
        brightness = np.mean(face_roi)
        is_bright_enough = brightness > 50  # Seuil réduit
        
        # Vérifier le ratio hauteur/largeur (plus permissif)
        aspect_ratio = w / float(h)
        is_good_aspect = 0.6 < aspect_ratio < 1.6
        
        if is_in_frame and is_bright_enough and is_good_aspect:
            filtered_faces.append((x, y, w, h))
    
    # Mettre à jour la liste des visages avec ceux qui passent les filtres
    faces = np.array(filtered_faces)
    
    if len(faces) > 0:
        log_message(f"{len(faces)} visage(s) clair(s) et centré(s) détecté(s)")
    
    # Trier les visages par taille (du plus grand au plus petit)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    
    # Ne garder que les 2 visages les plus grands (s'il y en a plusieurs)
    max_faces = 2
    faces = faces[:max_faces]
    
    # Convertir en image PIL pour le dessin avancé
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman.ttf", 32)
        small_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Traitement de chaque visage détecté
    for i, (x, y, w, h) in enumerate(faces, 1):
        log_message(f"Traitement du visage #{i} - Position: (x:{x}, y:{y}), Taille: {w}x{h}")
        
        # Couleurs "Girly"
        pink_color = (255, 105, 180)  # Hot Pink
        pastel_pink = (255, 182, 193) # Light Pink
        
        # Dessiner le rectangle (style plus doux)
        draw.rectangle([x, y, x+w, y+h], outline=pink_color, width=3)

        # Extract the Region of Interest (ROI) for the face
        roi_gray = gray_frame[y:y+h, x:x+w]
        
        try:
            # Preprocess the ROI for the emotion classifier
            roi_gray = cv2.resize(roi_gray, (64, 64))
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Prédiction de l'émotion
            log_message("Prédiction de l'émotion en cours...")
            prediction = classifier.predict(roi, verbose=0)[0]
            emotion_probability = np.max(prediction)
            emotion_label = EMOTIONS[prediction.argmax()]
            
            log_message(f"Émotion détectée: {emotion_label} (Confiance: {emotion_probability*100:.1f}%)")
            
            # Afficher seulement l'émotion principale avec le style "cute"
            text = f"{emotion_label}: {emotion_probability*100:.1f}%"
            
            # Fond pour le texte (optionnel, pour lisibilité)
            # text_bbox = draw.textbbox((x, y - 40), text, font=font)
            # draw.rectangle(text_bbox, fill=pastel_pink)
            
            draw.text((x, y - 40), text, font=font, fill=pink_color)

        except Exception as e:
            print(f"Error during prediction: {e}")

    # Reconvertir en format OpenCV
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


    # Afficher le frame avec les détections
    cv2.imshow('Reconnaissance des Émotions', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Nettoyage ---
elapsed_time = time.time() - start_time
log_message("--------------------------------------------------")
log_message(f"Arrêt de l'application. Temps d'exécution: {elapsed_time:.2f} secondes")
log_message(f"Nombre total de frames traitées: {frame_count}")
log_message(f"FPS moyen: {frame_count/elapsed_time:.2f}")
log_message("Libération des ressources...")

# Libération des ressources
camera.release()
cv2.destroyAllWindows()
log_message("Application arrêtée avec succès")
