import cv2
import numpy as np
import os
import time
import requests
from datetime import datetime
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app)

# --- Configuration ---
MODEL_URL = 'https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
CASCADE_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'

MODEL_FILE = 'fer2013_mini_XCEPTION.102-0.66.hdf5'
FACE_DETECTOR_PATH = 'haarcascade_frontalface_default.xml'
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
WEBCAM_INDEX = 0

# Global variables
camera = None
classifier = None
face_detector = None
current_emotion = "neutral"
current_confidence = 0.0
current_probabilities = {emotion: 0.0 for emotion in EMOTIONS}

# Temporal smoothing for better accuracy
prediction_history = []
HISTORY_SIZE = 5  # Nombre de prédictions à moyenner
CONFIDENCE_THRESHOLD = 0.4  # Seuil minimum de confiance (40%)

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
        
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(filename, 'wb') as f:
            downloaded_size = 0
            for data in response.iter_content(block_size):
                f.write(data)
                downloaded_size += len(data)
        
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

def initialize_models():
    """Initialise le modèle et le détecteur de visage"""
    global classifier, face_detector
    
    log_message("Vérification des fichiers nécessaires...")
    if not check_and_download_files():
        log_message("Erreur: Impossible de télécharger tous les fichiers nécessaires.")
        return False
    
    log_message("Chargement du modèle et du détecteur de visage...")
    try:
        log_message(f"Chargement du modèle depuis : {MODEL_FILE}")
        classifier = load_model(MODEL_FILE)
        log_message("Modèle chargé avec succès")
        
        log_message(f"Chargement du classificateur de visage depuis : {FACE_DETECTOR_PATH}")
        face_detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
        log_message("Détecteur de visage chargé avec succès")
        return True
    except Exception as e:
        log_message(f"ERREUR lors du chargement des fichiers: {e}")
        return False

def initialize_camera():
    """Initialise la webcam"""
    global camera
    
    log_message(f"Initialisation de la webcam (index: {WEBCAM_INDEX})...")
    camera = cv2.VideoCapture(WEBCAM_INDEX)
    
    if not camera.isOpened():
        log_message(f"ERREUR: Impossible d'ouvrir la webcam à l'index {WEBCAM_INDEX}")
        return False
    
    log_message("Webcam initialisée avec succès")
    return True

def generate_frames():
    """Génère les frames pour le streaming vidéo avec optimisations FPS et précision améliorée"""
    global camera, classifier, face_detector, current_emotion, current_confidence, current_probabilities, prediction_history
    
    frame_count = 0
    last_faces = []
    
    while True:
        if camera is None or not camera.isOpened():
            break
            
        ret, frame = camera.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Flip horizontalement pour effet miroir
        frame = cv2.flip(frame, 1)
        
        # Convertir en niveaux de gris
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Détection des visages seulement toutes les 3 frames pour améliorer FPS
        if frame_count % 3 == 0:
            faces = face_detector.detectMultiScale(
                gray_frame,
                scaleFactor=1.2,  # Augmenté pour plus de vitesse
                minNeighbors=4,   # Réduit pour plus de vitesse
                minSize=(80, 80), # Augmenté pour ignorer petits visages
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Filtrage simplifié des visages
            if len(faces) > 0:
                # Trier par taille et garder les 2 plus grands
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:2]
                last_faces = faces
        else:
            # Réutiliser les visages de la frame précédente
            faces = last_faces
        
        # Couleur rose girly
        pink_color = (255, 105, 180)
        
        # Traitement de chaque visage - prédiction seulement toutes les 2 frames
        for (x, y, w, h) in faces:
            # Dessiner le rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), pink_color, 3)
            
            # Prédiction d'émotion seulement toutes les 2 frames
            if frame_count % 2 == 0:
                roi_gray = gray_frame[y:y+h, x:x+w]
                
                try:
                    # Prétraitement amélioré pour meilleure précision
                    roi_gray = cv2.resize(roi_gray, (64, 64))
                    
                    # Égalisation d'histogramme pour améliorer le contraste
                    roi_gray = cv2.equalizeHist(roi_gray)
                    
                    # Normalisation
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    
                    # Prédiction
                    prediction = classifier.predict(roi, verbose=0)[0]
                    
                    # Ajouter à l'historique pour lissage temporel
                    prediction_history.append(prediction)
                    if len(prediction_history) > HISTORY_SIZE:
                        prediction_history.pop(0)
                    
                    # Moyenner les prédictions sur l'historique
                    if len(prediction_history) >= 3:  # Au moins 3 prédictions
                        smoothed_prediction = np.mean(prediction_history, axis=0)
                    else:
                        smoothed_prediction = prediction
                    
                    emotion_probability = np.max(smoothed_prediction)
                    emotion_label = EMOTIONS[smoothed_prediction.argmax()]
                    
                    # Appliquer seuil de confiance
                    if emotion_probability >= CONFIDENCE_THRESHOLD:
                        # Mettre à jour les variables globales seulement si confiance suffisante
                        current_emotion = emotion_label
                        current_confidence = emotion_probability * 100
                        
                        # Stocker toutes les probabilités pour comparaison
                        for i, emotion in enumerate(EMOTIONS):
                            current_probabilities[emotion] = float(smoothed_prediction[i] * 100)
                    # Sinon, garder l'émotion précédente
                    
                except Exception as e:
                    pass  # Ignorer les erreurs silencieusement pour ne pas ralentir
            
            # Afficher le texte avec l'émotion actuelle
            text = f"{current_emotion}: {current_confidence:.1f}%"
            cv2.putText(frame, text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, pink_color, 2)
        
        # Encoder le frame en JPEG avec qualité réduite pour plus de vitesse
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]  # Qualité 75 au lieu de 95 par défaut
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route pour le streaming vidéo"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion')
def get_emotion():
    """API pour obtenir l'émotion actuelle et toutes les probabilités"""
    return jsonify({
        'emotion': current_emotion,
        'confidence': round(current_confidence, 1),
        'probabilities': {k: round(v, 1) for k, v in current_probabilities.items()}
    })

@app.route('/health')
def health():
    """Vérification de l'état de l'application"""
    return jsonify({
        'status': 'ok',
        'camera': camera is not None and camera.isOpened(),
        'model': classifier is not None
    })

if __name__ == '__main__':
    log_message("🎀 Démarrage de l'application Flask de reconnaissance d'émotions 🎀")
    
    # Initialiser les modèles
    if not initialize_models():
        log_message("Impossible d'initialiser les modèles. Arrêt.")
        exit(1)
    
    # Initialiser la caméra
    if not initialize_camera():
        log_message("Impossible d'initialiser la caméra. Arrêt.")
        exit(1)
    
    log_message("✨ Application prête! Ouvrez http://localhost:5000 dans votre navigateur ✨")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    finally:
        if camera is not None:
            camera.release()
        log_message("Application arrêtée")
