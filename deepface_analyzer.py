import cv2
import numpy as np
from deepface import DeepFace
import time
from typing import Dict, Tuple, Optional

class DeepFaceAnalyzer:
    def __init__(self):
        self.models = ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'ArcFace', 'Dlib']
        self.backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
        self.last_analysis_time = 0
        self.min_interval = 0.1  # Minimum interval between analyses (seconds)
        self.cache_duration = 0.5  # Cache duration for results
        self.cached_result = None
        self.cache_timestamp = 0
        
    def analyze_emotion(self, face_img: np.ndarray, enforce_detection: bool = False) -> Dict:
        """
        Analyze emotion from face image using DeepFace
        
        Args:
            face_img: Face image as numpy array (BGR format from OpenCV)
            enforce_detection: Whether to enforce face detection
            
        Returns:
            Dictionary containing emotion analysis results
        """
        current_time = time.time()
        
        # Check if we can use cached result
        if (self.cached_result and 
            current_time - self.cache_timestamp < self.cache_duration):
            return self.cached_result
        
        # Rate limiting
        if current_time - self.last_analysis_time < self.min_interval:
            return self.cached_result or self._get_default_result()
        
        try:
            # Convert BGR to RGB for DeepFace
            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            else:
                rgb_img = face_img
            
            # Analyze with DeepFace
            analysis = DeepFace.analyze(
                img_path=rgb_img,
                actions=['emotion'],
                enforce_detection=enforce_detection,
                detector_backend='opencv',
                silent=True
            )
            
            # Handle both single face and multiple faces results
            if isinstance(analysis, list):
                if len(analysis) > 0:
                    result = analysis[0]
                else:
                    return self._get_default_result()
            else:
                result = analysis
            
            # Extract emotion data
            emotions = result.get('emotion', {})
            dominant_emotion = result.get('dominant_emotion', 'neutral')
            region = result.get('region', {})
            
            # Format result to match existing structure
            formatted_result = {
                'dominant_emotion': dominant_emotion,
                'emotions': emotions,
                'confidence': emotions.get(dominant_emotion, 0.0),
                'region': region,
                'success': True
            }
            
            # Cache the result
            self.cached_result = formatted_result
            self.cache_timestamp = current_time
            self.last_analysis_time = current_time
            
            return formatted_result
            
        except Exception as e:
            print(f"DeepFace analysis error: {str(e)}")
            return self._get_default_result()
    
    def analyze_frame(self, frame: np.ndarray, face_coords: Optional[Tuple[int, int, int, int]] = None) -> Dict:
        """
        Analyze emotion from video frame
        
        Args:
            frame: Video frame as numpy array (BGR format)
            face_coords: Optional face coordinates (x, y, w, h)
            
        Returns:
            Dictionary containing emotion analysis results
        """
        if face_coords:
            x, y, w, h = face_coords
            face_img = frame[y:y+h, x:x+w]
        else:
            face_img = frame
        
        return self.analyze_emotion(face_img, enforce_detection=False)
    
    def _get_default_result(self) -> Dict:
        """Return default result when analysis fails"""
        return {
            'dominant_emotion': 'neutral',
            'emotions': {
                'angry': 0.0,
                'disgust': 0.0,
                'fear': 0.0,
                'happy': 0.0,
                'sad': 0.0,
                'surprise': 0.0,
                'neutral': 1.0
            },
            'confidence': 0.0,
            'region': {},
            'success': False
        }
    
    def get_available_models(self):
        """Get list of available DeepFace models"""
        return self.models
    
    def get_available_backends(self):
        """Get list of available detector backends"""
        return self.backends

# Global instance
deepface_analyzer = DeepFaceAnalyzer()
