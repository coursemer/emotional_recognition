import cv2
import numpy as np
import os
import time
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
import tensorflow as tf
from deepface import DeepFace
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from collections import deque
import pickle

class DeepFaceContinuousTrainer:
    def __init__(self):
        self.training_data_dir = "training_data"
        self.models_dir = "trained_models"
        self.logs_dir = "training_logs"
        
        # Create directories
        for dir_path in [self.training_data_dir, self.models_dir, self.logs_dir]:
            Path(dir_path).mkdir(exist_ok=True)
        
        # Training configuration
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.target_image_size = (224, 224)
        self.batch_size = 32
        self.epochs_per_cycle = 5
        self.min_samples_per_emotion = 50
        
        # Training state
        self.is_training = False
        self.training_thread = None
        self.training_history = deque(maxlen=100)
        self.model_accuracy = 0.0
        self.last_training_time = None
        
        # Data collection
        self.collected_samples = {emotion: [] for emotion in self.emotions}
        self.data_buffer = deque(maxlen=1000)
        
        # Load existing model and data
        self.load_existing_data()
        
    def load_existing_data(self):
        """Load existing training data and models"""
        try:
            # Load collected samples
            data_file = Path(self.training_data_dir) / "collected_samples.pkl"
            if data_file.exists():
                with open(data_file, 'rb') as f:
                    self.collected_samples = pickle.load(f)
                print(f"Loaded {sum(len(v) for v in self.collected_samples.values())} training samples")
            
            # Load training history
            history_file = Path(self.logs_dir) / "training_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                    self.training_history.extend(history_data.get('history', []))
                    self.model_accuracy = history_data.get('latest_accuracy', 0.0)
                print(f"Loaded training history with accuracy: {self.model_accuracy:.2f}")
                
        except Exception as e:
            print(f"Error loading existing data: {e}")
    
    def save_training_data(self):
        """Save training data and history"""
        try:
            # Save collected samples
            data_file = Path(self.training_data_dir) / "collected_samples.pkl"
            with open(data_file, 'wb') as f:
                pickle.dump(self.collected_samples, f)
            
            # Save training history
            history_file = Path(self.logs_dir) / "training_history.json"
            history_data = {
                'history': list(self.training_history),
                'latest_accuracy': self.model_accuracy,
                'last_update': datetime.now().isoformat()
            }
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving training data: {e}")
    
    def add_training_sample(self, face_image, emotion_label, confidence=0.0):
        """Add a new training sample"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(face_image)
            
            # Add to appropriate emotion category
            if emotion_label in self.emotions and confidence > 0.3:  # Only use high-confidence samples
                self.collected_samples[emotion_label].append({
                    'image': processed_image,
                    'timestamp': datetime.now().isoformat(),
                    'confidence': confidence
                })
                
                # Keep only recent samples to prevent overfitting
                if len(self.collected_samples[emotion_label]) > 500:
                    self.collected_samples[emotion_label] = self.collected_samples[emotion_label][-400:]
                
                return True
        except Exception as e:
            print(f"Error adding training sample: {e}")
        
        return False
    
    def preprocess_image(self, image):
        """Preprocess image for training"""
        # Resize to target size
        image = cv2.resize(image, self.target_image_size)
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def prepare_training_data(self):
        """Prepare data for training"""
        X = []
        y = []
        
        for emotion_idx, emotion in enumerate(self.emotions):
            samples = self.collected_samples[emotion]
            
            # Only train if we have enough samples
            if len(samples) >= self.min_samples_per_emotion:
                for sample in samples:
                    X.append(sample['image'])
                    y.append(emotion_idx)
        
        if len(X) == 0:
            return None, None
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def create_model(self):
        """Create a custom emotion recognition model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.emotions), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self):
        """Train the model with collected data"""
        try:
            print("Starting model training...")
            
            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_training_data()
            
            if X_train is None:
                print("Not enough training data")
                return False
            
            print(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples")
            
            # Create and train model
            model = self.create_model()
            
            # Training with callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
            ]
            
            history = model.fit(
                X_train, y_train,
                batch_size=self.batch_size,
                epochs=self.epochs_per_cycle,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            self.model_accuracy = accuracy
            
            # Save model
            model_path = Path(self.models_dir) / f"emotion_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
            model.save(model_path)
            
            # Also save as latest
            latest_path = Path(self.models_dir) / "latest_model.h5"
            model.save(latest_path)
            
            # Log training
            training_log = {
                'timestamp': datetime.now().isoformat(),
                'accuracy': float(accuracy),
                'loss': float(loss),
                'samples_trained': len(X_train),
                'samples_tested': len(X_test),
                'epochs': len(history.history['loss'])
            }
            
            self.training_history.append(training_log)
            self.last_training_time = datetime.now()
            
            # Save training data
            self.save_training_data()
            
            print(f"Training completed. Accuracy: {accuracy:.4f}")
            return True
            
        except Exception as e:
            print(f"Error during training: {e}")
            return False
    
    def start_continuous_training(self):
        """Start continuous training in background"""
        if self.is_training:
            print("Training already in progress")
            return
        
        def training_loop():
            while True:
                try:
                    print("Checking if training should run...")
                    
                    # Check if we have enough data
                    total_samples = sum(len(samples) for samples in self.collected_samples.values())
                    min_required = len(self.emotions) * self.min_samples_per_emotion
                    
                    if total_samples >= min_required:
                        print(f"Starting training cycle with {total_samples} samples")
                        success = self.train_model()
                        
                        if success:
                            print("Training cycle completed successfully")
                        else:
                            print("Training cycle failed")
                    else:
                        print(f"Not enough data for training. Have {total_samples}, need {min_required}")
                    
                    # Wait before next cycle (1 hour)
                    time.sleep(3600)
                    
                except Exception as e:
                    print(f"Error in training loop: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        self.is_training = True
        self.training_thread = threading.Thread(target=training_loop, daemon=True)
        self.training_thread.start()
        print("Continuous training started")
    
    def get_training_status(self):
        """Get current training status"""
        return {
            'is_training': self.is_training,
            'model_accuracy': self.model_accuracy,
            'last_training': self.last_training_time.isoformat() if self.last_training_time else None,
            'samples_per_emotion': {emotion: len(samples) for emotion, samples in self.collected_samples.items()},
            'total_samples': sum(len(samples) for samples in self.collected_samples.values()),
            'training_history_count': len(self.training_history)
        }
    
    def stop_training(self):
        """Stop continuous training"""
        self.is_training = False
        if self.training_thread:
            self.training_thread.join(timeout=5)
        print("Training stopped")

# Global trainer instance
trainer = DeepFaceContinuousTrainer()
