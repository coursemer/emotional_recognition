import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import time
import threading
from datetime import datetime
from deepface_trainer_simple import trainer

app = Flask(__name__)
CORS(app)

# Configuration
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
WEBCAM_INDEX = 0

# Global variables
camera = None
classifier = None
face_detector = None
current_emotion = "neutral"
current_confidence = 0.0
current_probabilities = {emotion: 0.0 for emotion in EMOTIONS}

# Performance optimization
frame_skip = 2  # Process every 2nd frame for better FPS
last_analysis_time = 0
analysis_interval = 0.1  # Analyze every 100ms max
emotion_cache = {}
cache_duration = 0.5  # Cache results for 500ms
face_cascade = None

# Emotion simulation (since we can't use TensorFlow/DeepFace easily)
def simulate_emotion_detection(face_img):
    """Simple emotion simulation for demo purposes"""
    # In a real implementation, this would use a trained model
    # For now, we'll simulate with some basic logic
    
    # Generate random probabilities that sum to 1
    probs = np.random.dirichlet(np.ones(len(EMOTIONS)))
    
    # Bias towards certain emotions based on image characteristics
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
    
    # Simple heuristics (very basic)
    brightness = np.mean(gray)
    
    # Adjust probabilities based on brightness (just for simulation)
    if brightness > 180:  # Bright image - maybe happy
        probs[EMOTIONS.index('happy')] *= 1.5
    elif brightness < 100:  # Dark image - maybe sad
        probs[EMOTIONS.index('sad')] *= 1.5
    
    # Normalize
    probs = probs / probs.sum()
    
    # Get dominant emotion
    dominant_idx = np.argmax(probs)
    dominant_emotion = EMOTIONS[dominant_idx]
    confidence = probs[dominant_idx]
    
    return {
        'dominant_emotion': dominant_emotion,
        'emotions': dict(zip(EMOTIONS, probs.tolist())),
        'confidence': float(confidence),
        'success': True
    }

def detect_faces(frame):
    """Detect faces in the frame with improved accuracy and performance"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization for better contrast
    gray = cv2.equalizeHist(gray)
    
    # Use a smaller image for faster detection
    small_gray = cv2.resize(gray, (0,0), fx=0.5, fy=0.5)
    
    # Detect faces with optimized parameters
    faces = face_cascade.detectMultiScale(
        small_gray, 
        scaleFactor=1.08,  # Finer scale for better accuracy
        minNeighbors=6,    # More neighbors for better quality
        minSize=(20, 20),  # Smaller minimum size
        maxSize=(150, 150), # Smaller maximum size for performance
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Scale coordinates back to original size
    faces = [(int(x*2), int(y*2), int(w*2), int(h*2)) for (x, y, w, h) in faces]
    
    # If no faces detected, try with more lenient parameters
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(30, 30),
            maxSize=(300, 300),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    
    return faces

def analyze_emotions_from_frame(frame):
    """Analyze emotions from all detected faces with caching"""
    global current_emotion, current_confidence, current_probabilities
    
    current_time = time.time()
    
    # Check cache first
    cache_key = hash(frame.tobytes()) if hasattr(frame, 'tobytes') else hash(str(frame.shape))
    if cache_key in emotion_cache:
        cached_result = emotion_cache[cache_key]
        if current_time - cached_result['timestamp'] < cache_duration:
            # Update global variables from cache
            if cached_result['results']:
                first_result = cached_result['results'][0]
                current_emotion = first_result['emotion']
                current_confidence = first_result['confidence']
                current_probabilities = first_result['probabilities']
            return cached_result['results']
    
    # Rate limiting
    if current_time - last_analysis_time < analysis_interval:
        return []
    
    faces = detect_faces(frame)
    results = []
    
    if len(faces) > 0:
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face with some padding
            padding = 20
            x_padded = max(0, x - padding)
            y_padded = max(0, y - padding)
            w_padded = min(frame.shape[1] - x_padded, w + 2 * padding)
            h_padded = min(frame.shape[0] - y_padded, h + 2 * padding)
            
            face_img = frame[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
            
            # Analyze emotion
            emotion_result = simulate_emotion_detection(face_img)
            
            # Store result with face coordinates
            results.append({
                'face_id': i,
                'emotion': emotion_result['dominant_emotion'],
                'confidence': emotion_result['confidence'],
                'probabilities': emotion_result['emotions'],
                'coordinates': (x, y, w, h),
                'success': True
            })
        
        # Update global variables with the first face (for compatibility)
        if results:
            first_result = results[0]
            current_emotion = first_result['emotion']
            current_confidence = first_result['confidence']
            current_probabilities = first_result['probabilities']
        
        # Cache the results
        emotion_cache[cache_key] = {
            'results': results,
            'timestamp': current_time
        }
        
        # Clean old cache entries
        emotion_cache.clear() if len(emotion_cache) > 50 else None
    
    return results

def generate_frames():
    """Generate video frames for streaming with optimized FPS"""
    global camera, last_analysis_time
    
    if camera is None:
        camera = cv2.VideoCapture(WEBCAM_INDEX)
        # Set camera parameters for better performance
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        if not camera.isOpened():
            print("Error: Could not open webcam")
            # Create a black frame as fallback
            while True:
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, "Webcam not available", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', error_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    print("Webcam opened successfully")
    frame_count = 0
    fps_counter = 0
    fps_start_time = time.time()
    
    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Could not read frame from webcam")
            break
        
        # Flip frame horizontally to prevent mirror effect
        frame = cv2.flip(frame, 1)
        
        frame_count += 1
        fps_counter += 1
        
        # Calculate and display FPS
        current_time = time.time()
        if current_time - fps_start_time >= 1.0:
            fps = fps_counter / (current_time - fps_start_time)
            fps_counter = 0
            fps_start_time = current_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Analyze emotion every frame_skip frames for better performance
        if frame_count % frame_skip == 0:
            try:
                emotion_results = analyze_emotions_from_frame(frame)
                last_analysis_time = current_time
                
                # Draw face rectangles and emotion text for all detected faces
                for result in emotion_results:
                    x, y, w, h = result['coordinates']
                    
                    # Draw rectangle with different colors for different faces
                    color = (255, 105, 180) if result['face_id'] == 0 else (255, 165, 0)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Add emotion text
                    emotion_text = f"Face {result['face_id']+1}: {result['emotion']} ({result['confidence']:.2f})"
                    cv2.putText(frame, emotion_text, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Add confidence bar
                    bar_width = int(w * result['confidence'])
                    cv2.rectangle(frame, (x, y+h+5), (x+bar_width, y+h+10), color, -1)
                    
                    # Add training data if confidence is high
                    if result['confidence'] > 0.7:
                        # Extract face for training
                        padding = 20
                        x_padded = max(0, x - padding)
                        y_padded = max(0, y - padding)
                        w_padded = min(frame.shape[1] - x_padded, w + 2 * padding)
                        h_padded = min(frame.shape[0] - y_padded, h + 2 * padding)
                        
                        face_img = frame[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
                        
                        # Add to training data
                        trainer.add_training_sample(face_img, result['emotion'], result['confidence'])
                        
                        # Indicate training sample collected
                        cv2.circle(frame, (x + w - 10, y + 10), 5, (0, 255, 0), -1)
                    
            except Exception as e:
                print(f"Error in emotion analysis: {e}")
        
        # Add frame counter for debugging
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Convert frame to JPEG with lower quality for better performance
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
        if not ret:
            print("Error: Could not encode frame")
            continue
            
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion_data')
def emotion_data():
    """Get current emotion data as JSON"""
    return jsonify({
        'emotion': current_emotion,
        'confidence': current_confidence,
        'probabilities': current_probabilities,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/multi_face_data')
def multi_face_data():
    """Get multi-face emotion data as JSON"""
    return jsonify({
        'faces': [],
        'timestamp': datetime.now().isoformat(),
        'message': 'Multi-face data available in video stream'
    })

@app.route('/training/start')
def start_training():
    """Start continuous training"""
    try:
        trainer.start_continuous_training()
        return jsonify({
            'status': 'success',
            'message': 'Continuous training started',
            'training_status': trainer.get_training_status()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/training/stop')
def stop_training():
    """Stop continuous training"""
    try:
        trainer.stop_training()
        return jsonify({
            'status': 'success',
            'message': 'Training stopped',
            'training_status': trainer.get_training_status()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/training/status')
def training_status():
    """Get training status"""
    return jsonify(trainer.get_training_status())

@app.route('/training/train_now')
def train_now():
    """Trigger immediate training"""
    try:
        success = trainer.simulate_training()
        return jsonify({
            'status': 'success' if success else 'failed',
            'message': 'Training completed' if success else 'Training failed',
            'training_status': trainer.get_training_status()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

def init_resources():
    """Initialize resources"""
    global face_cascade
    
    print("Initializing resources...")
    
    # Load face cascade
    face_cascade_path = 'haarcascade_frontalface_default.xml'
    try:
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        if face_cascade.empty():
            print("Warning: Could not load face cascade classifier")
            return False
        else:
            print("Face cascade classifier loaded successfully")
            return True
    except Exception as e:
        print(f"Error loading face cascade: {e}")
        return False

if __name__ == '__main__':
    init_resources()
    
    # Start continuous training automatically
    print("Starting 24/7 continuous training system...")
    trainer.start_continuous_training()
    
    app.run(debug=True, host='0.0.0.0', port=5002)
