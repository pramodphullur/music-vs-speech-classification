"""
3D CNN Audio Classification: Music vs Speech

A deep learning approach to classify audio files using 3D Convolutional Neural Networks
with spectrogram image representations.

Author: Pramod P Hullur
Date: 16-07-2025
"""

import os
import librosa
import matplotlib.cm
import numpy as np
import matplotlib
import librosa.display
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
TARGET_SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
MAX_DURATION = 5.0  # seconds
FIXED_LENGTH = int(TARGET_SR * MAX_DURATION / HOP_LENGTH)  # ~216 frames

class AudioPreprocessor:
    """Handle audio preprocessing and feature extraction"""
    
    def __init__(self, target_sr=TARGET_SR, n_mels=N_MELS, max_duration=MAX_DURATION):
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.max_duration = max_duration
        self.fixed_length = int(target_sr * max_duration / HOP_LENGTH)
        
    def normalize_spectrogram(self, spectrogram, axis=(0, 1)):
        """Normalize spectrogram to [0, 1] range"""
        min_val = np.min(spectrogram, axis=axis, keepdims=True)
        max_val = np.max(spectrogram, axis=axis, keepdims=True)
        
        # Avoid division by zero
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)
        
        normalized = (spectrogram - min_val) / range_val
        return np.clip(normalized, 0, 1)
    
    def pad_or_truncate(self, spectrogram, target_length):
        """Pad or truncate spectrogram to fixed length"""
        current_length = spectrogram.shape[1]
        
        if current_length > target_length:
            # Truncate from the middle
            start = (current_length - target_length) // 2
            return spectrogram[:, start:start + target_length]
        elif current_length < target_length:
            # Pad with zeros
            pad_width = target_length - current_length
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            return np.pad(spectrogram, ((0, 0), (pad_left, pad_right)), mode='constant')
        else:
            return spectrogram
    
    def create_depth_layers(self, spectrogram_2d, n_layers=8):
        """Create depth layers from 2D spectrogram using intensity binning"""
        # Apply colormap
        colormap = matplotlib.colormaps.get_cmap('viridis')  # Changed to viridis for better results
        rgba_img = colormap(spectrogram_2d)
        
        # Convert to RGB
        rgb_img = rgba_img[:, :, :3]
        
        # Calculate intensity (grayscale)
        intensity = np.dot(rgb_img, [0.299, 0.587, 0.114])
        
        # Create bins for intensity
        bins = np.linspace(0, 1, n_layers + 1)
        
        # Create depth layers
        H, W = spectrogram_2d.shape
        depth_layers = np.zeros((n_layers, H, W, 1), dtype=np.float32)
        
        for i in range(n_layers):
            # Create mask for current intensity bin
            mask = (intensity >= bins[i]) & (intensity < bins[i + 1])
            
            # Fill layer with normalized intensity values
            depth_layers[i, :, :, 0] = np.where(mask, intensity, 0)
        
        return depth_layers
    
    def process_audio_file(self, filepath):
        """Process single audio file into 3D representation"""
        try:
            # Load audio
            y, sr = librosa.load(filepath, sr=self.target_sr)
            
            # Truncate or pad to fixed duration
            max_samples = int(self.target_sr * self.max_duration)
            if len(y) > max_samples:
                y = y[:max_samples]
            else:
                y = np.pad(y, (0, max_samples - len(y)), mode='constant')
            
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=self.target_sr, 
                n_mels=self.n_mels,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH
            )
            
            # Convert to log scale
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            normalized_spec = self.normalize_spectrogram(log_mel_spec)
            
            # Ensure fixed length
            normalized_spec = self.pad_or_truncate(normalized_spec, self.fixed_length)
            
            # Create depth layers
            depth_representation = self.create_depth_layers(normalized_spec)
            
            return depth_representation
            
        except Exception as e:
            logger.error(f"Error processing {filepath}: {str(e)}")
            return None

class MusicSpeechDataset:
    """Handle dataset loading and preparation"""
    
    def __init__(self, data_dir, preprocessor):
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor
        
    def load_data_from_directory(self, directory_path, label, class_name):
        """Load all audio files from a directory"""
        spectrograms = []
        labels = []
        
        directory = Path(directory_path)
        if not directory.exists():
            logger.warning(f"Directory {directory} does not exist")
            return spectrograms, labels
            
        audio_files = list(directory.glob("*.wav"))
        logger.info(f"Loading {len(audio_files)} {class_name} files from {directory}")
        
        for filepath in audio_files:
            processed_audio = self.preprocessor.process_audio_file(filepath)
            if processed_audio is not None:
                spectrograms.append(processed_audio)
                labels.append(label)
                
        logger.info(f"Successfully loaded {len(spectrograms)} {class_name} samples")
        return spectrograms, labels
    
    def prepare_dataset(self):
        """Load and prepare the complete dataset"""
        all_spectrograms = []
        all_labels = []
        
        # Load training data
        train_speech_specs, train_speech_labels = self.load_data_from_directory(
            self.data_dir / "train" / "speech", 0, "speech"
        )
        train_music_specs, train_music_labels = self.load_data_from_directory(
            self.data_dir / "train" / "music", 1, "music"
        )
        
        # Load test data
        test_speech_specs, test_speech_labels = self.load_data_from_directory(
            self.data_dir / "test" / "speech", 0, "speech"
        )
        test_music_specs, test_music_labels = self.load_data_from_directory(
            self.data_dir / "test" / "music", 1, "music"
        )
        
        # Combine all data
        all_spectrograms = (train_speech_specs + train_music_specs + 
                          test_speech_specs + test_music_specs)
        all_labels = (train_speech_labels + train_music_labels + 
                     test_speech_labels + test_music_labels)
        
        if len(all_spectrograms) == 0:
            raise ValueError("No audio files found in the dataset")
        
        # Convert to numpy arrays
        X = np.array(all_spectrograms, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int32)
        
        # Log dataset statistics
        logger.info(f"Dataset shape: {X.shape}")
        logger.info(f"Labels shape: {y.shape}")
        logger.info(f"Speech samples: {np.sum(y == 0)}")
        logger.info(f"Music samples: {np.sum(y == 1)}")
        
        return X, y

class CNN3DModel:
    """3D CNN model for music-speech classification"""
    
    def __init__(self, input_shape, num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        """Build the 3D CNN architecture"""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First 3D Conv block
            layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling3D((2, 2, 2)),
            layers.Dropout(0.2),
            
            # Second 3D Conv block
            layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling3D((2, 2, 2)),
            layers.Dropout(0.3),
            
            # Third 3D Conv block
            layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling3D(),
            layers.Dropout(0.4),
            
            # Classification head
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
            
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Model compiled successfully")
        self.model.summary()
        
    def get_callbacks(self, patience=15):
        """Get training callbacks"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=8):
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
            
        callbacks = self.get_callbacks()
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not built yet.")
            
        # Make predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        test_accuracy = np.mean(y_pred == y_test)
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Classification report
        class_names = ['Speech', 'Music']
        report = classification_report(y_test, y_pred, target_names=class_names)
        logger.info(f"Classification Report:\n{report}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        return y_pred, y_pred_proba, test_accuracy

class MusicSpeechClassifier:
    """Main classifier class that orchestrates the entire pipeline"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.preprocessor = AudioPreprocessor()
        self.dataset = MusicSpeechDataset(data_dir, self.preprocessor)
        self.model = None
        
    def prepare_data(self, test_size=0.2, val_size=0.2, random_state=42):
        """Prepare the dataset"""
        logger.info("Preparing dataset...")
        
        X, y = self.dataset.prepare_dataset()
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            random_state=random_state, stratify=y_temp
        )
        
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Validation set: {X_val.shape}")
        logger.info(f"Test set: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def build_and_train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=8):
        """Build and train the model"""
        input_shape = X_train.shape[1:]
        logger.info(f"Input shape: {input_shape}")
        
        self.model = CNN3DModel(input_shape)
        self.model.build_model()
        self.model.compile_model()
        
        history = self.model.train(X_train, y_train, X_val, y_val, epochs, batch_size)
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
            
        return self.model.evaluate(X_test, y_test)
    
    def predict_audio_file(self, audio_file_path):
        """Predict class for a new audio file"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
            
        # Process the audio file
        processed_audio = self.preprocessor.process_audio_file(audio_file_path)
        
        if processed_audio is None:
            raise ValueError(f"Could not process audio file: {audio_file_path}")
            
        # Add batch dimension
        input_data = np.expand_dims(processed_audio, axis=0)
        
        # Make prediction
        prediction = self.model.model.predict(input_data)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction, axis=1)[0]
        
        class_names = ['Speech', 'Music']
        result = {
            'class': class_names[predicted_class],
            'confidence': confidence,
            'probabilities': {
                'speech': prediction[0][0],
                'music': prediction[0][1]
            }
        }
        
        return result
    
    def save_model(self, filepath='music_speech_classifier.keras'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
            
        self.model.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        self.model = CNN3DModel((8, 128, 216, 1))  # You'll need to adjust this based on your actual input shape
        self.model.model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', marker='o')
    ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution pipeline"""
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Initialize classifier
    data_dir = "/dataset"  # Update this path
    classifier = MusicSpeechClassifier(data_dir)
    
    try:
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = classifier.prepare_data()
        
        # Build and train model
        history = classifier.build_and_train_model(
            X_train, y_train, X_val, y_val, 
            epochs=10, batch_size=8
        )
        
        # Plot training history
        plot_training_history(history)
        
        # Evaluate model
        y_pred, y_pred_proba, test_accuracy = classifier.evaluate_model(X_test, y_test)
        
        # Save model
        classifier.save_model()
        
        logger.info("Training completed successfully!")
        
        return classifier, history, test_accuracy
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    # Execute the main pipeline 
    classifier, history, test_accuracy = main()
    
    # Example usage for prediction
    # result = classifier.predict_audio_file("path/to/new/audio/file.wav")
    # print(f"Prediction: {result['class']} (confidence: {result['confidence']:.4f})")
