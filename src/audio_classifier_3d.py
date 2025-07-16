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
import cv2


# Configuration
CONFIG = {
    'data_dir': "/kaggle/input/music-speech/dataset",
    'target_sr': 22050,
    'batch_size': 4,
    'epochs': 50,
    'learning_rate': 0.001,
    'model_save_path': 'my_model.keras',
    'checkpoint_path': 'best_3d_cnn_model.h5'
}


def normalize(spectrogram_3d, min_amp=None, max_amp=None, axis=(0, 1)):
    """
    Normalize spectrogram to [0, 1] range.
    
    Args:
        spectrogram_3d: Input spectrogram array
        min_amp: Minimum amplitude (computed if None)
        max_amp: Maximum amplitude (computed if None)
        axis: Axes along which to compute min/max
    
    Returns:
        Normalized spectrogram clipped to [0, 1]
    """
    if min_amp is None:
        min_amp = np.min(spectrogram_3d, axis=axis, keepdims=True)
    if max_amp is None:
        max_amp = np.max(spectrogram_3d, axis=axis, keepdims=True)
    
    # Avoid division by zero
    denominator = max_amp - min_amp
    denominator = np.where(denominator == 0, 1, denominator)
    
    normalized = (spectrogram_3d - min_amp) / denominator
    return np.clip(normalized, 0, 1)


def process_audio_file(filepath, target_sr=22050):
    """
    Process a single audio file into 4D tensor representation.
    
    Args:
        filepath: Path to the audio file
        target_sr: Target sampling rate
    
    Returns:
        4D numpy array representing the processed audio
    """
    try:
        # Load audio
        y, sr = librosa.load(filepath, sr=target_sr)
        
        # Generate mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=target_sr)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Normalize
        spectrogram_2d = normalize(log_mel_spectrogram, axis=(0, 1))
        
        # Apply colormap
        colormap = matplotlib.colormaps.get_cmap('magma')
        rgba_img = colormap(spectrogram_2d)
        rgba_uint8 = (rgba_img * 255).astype(np.uint8)
        
        # Convert to RGB
        rgb_img = np.delete(rgba_img, 3, axis=2)
        rgb_uint8 = (rgb_img * 255).astype(np.uint8)
        
        # Calculate average RGB for binning
        avg_rgb = np.mean(rgb_uint8, axis=2)
        
        # Create intensity bins
        bins = np.linspace(0, 255, 11)
        H, W, _ = rgba_uint8.shape
        output_4d = np.zeros((10, H, W, 4), dtype=np.uint8)
        
        # Fill 4D tensor based on intensity bins
        for i in range(10):
            mask = ((avg_rgb >= bins[i]) & (avg_rgb < bins[i+1]))
            temp_img = np.zeros_like(rgba_uint8)
            temp_img[mask, :3] = rgba_uint8[mask, :3]
            temp_img[mask, 3] = 255
            temp_img[~mask, 3] = 0
            output_4d[i] = temp_img
        
        return output_4d
        
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None


def save_waveform_info(audio_dir, target_sr):
    """
    Process all audio files in a directory.
    
    Args:
        audio_dir: Directory containing audio files
        target_sr: Target sampling rate
    
    Returns:
        Tuple of (spectrograms, labels)
    """
    spectrograms, labels = [], []
    
    if not os.path.exists(audio_dir):
        print(f"Directory {audio_dir} does not exist!")
        return spectrograms, labels
    
    for filename in os.listdir(audio_dir):
        filepath = os.path.join(audio_dir, filename)
        if filepath.endswith('.wav'):
            processed_audio = process_audio_file(filepath, target_sr)
            if processed_audio is not None:
                spectrograms.append(processed_audio)
                # Label assignment: 0 for speech, 1 for music
                labels.append(0 if "speech" in filename else 1)
    
    return spectrograms, labels


def prepare_dataset(data_dir, target_sr):
    """
    Prepare the complete dataset from directory structure.
    
    Args:
        data_dir: Root directory containing train/test splits
        target_sr: Target sampling rate
    
    Returns:
        Tuple of (X, y) where X is features and y is labels
    """
    print("Loading training data...")
    
    # Load training data
    train_speech, train_speech_labels = save_waveform_info(
        os.path.join(data_dir, "train", "speech"), target_sr
    )
    train_music, train_music_labels = save_waveform_info(
        os.path.join(data_dir, "train", "music"), target_sr
    )
    
    # Load test data
    test_speech, test_speech_labels = save_waveform_info(
        os.path.join(data_dir, "test", "speech"), target_sr
    )
    test_music, test_music_labels = save_waveform_info(
        os.path.join(data_dir, "test", "music"), target_sr
    )
    
    # Combine all data
    all_spectrograms = train_speech + train_music + test_speech + test_music
    all_labels = train_speech_labels + train_music_labels + test_speech_labels + test_music_labels
    
    # Convert to numpy arrays
    X = np.array(all_spectrograms, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    
    # Normalize to [0, 1] range
    X = X / 255.0
    
    # Print dataset information
    print(f"Dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of speech samples: {np.sum(y == 0)}")
    print(f"Number of music samples: {np.sum(y == 1)}")
    
    return X, y


def create_3d_cnn_model(input_shape, num_classes=2):
    """
    Create a 3D CNN model for audio classification.
    
    Args:
        input_shape: Shape of input data (depth, height, width, channels)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First 3D Conv block
        layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Dropout(0.25),
        
        # Second 3D Conv block
        layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Dropout(0.25),
        
        # Third 3D Conv block
        layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Dropout(0.3),
        
        # Global pooling and classification head
        layers.GlobalAveragePooling3D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_and_train_model(model, X_train, y_train, X_val, y_val, 
                           epochs=50, batch_size=4, learning_rate=0.001):
    """
    Compile and train the 3D CNN model.
    
    Args:
        model: Keras model to train
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    
    Returns:
        Training history
    """
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            CONFIG['checkpoint_path'],
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def plot_training_history(history):
    """
    Plot training and validation metrics.
    
    Args:
        history: Training history from model.fit()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained Keras model
        X_test, y_test: Test data and labels
    
    Returns:
        Tuple of (predictions, prediction_probabilities)
    """
    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate accuracy
    test_accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Classification report
    class_names = ['Speech', 'Music']
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return y_pred, y_pred_proba


def predict_new_audio(model, audio_file_path, target_sr=22050):
    """
    Predict class for a new audio file.
    
    Args:
        model: Trained Keras model
        audio_file_path: Path to the audio file
        target_sr: Target sampling rate
    
    Returns:
        Tuple of (predicted_class, confidence)
    """
    # Process the audio file
    processed_audio = process_audio_file(audio_file_path, target_sr)
    
    if processed_audio is None:
        print(f"Error: Could not process audio file {audio_file_path}")
        return None, None
    
    # Normalize and reshape for prediction
    input_data = processed_audio.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction, axis=1)[0]
    
    class_names = ['Speech', 'Music']
    print(f"Predicted class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.4f}")
    
    return predicted_class, confidence


def main_pipeline():
    """
    Execute the complete 3D CNN pipeline.
    
    Returns:
        Tuple of (model, history, predictions, prediction_probabilities)
    """
    print("Starting 3D CNN Audio Classification Pipeline...")
    
    # Prepare dataset
    print("Preparing dataset...")
    X, y = prepare_dataset(CONFIG['data_dir'], CONFIG['target_sr'])
    
    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Create model
    input_shape = X_train.shape[1:]  # (depth, height, width, channels)
    print(f"Input shape: {input_shape}")
    
    model = create_3d_cnn_model(input_shape)
    
    # Train model
    print("Training model...")
    history = compile_and_train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=CONFIG['epochs'], 
        batch_size=CONFIG['batch_size'],
        learning_rate=CONFIG['learning_rate']
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("Evaluating model...")
    y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # Save final model
    model.save(CONFIG['model_save_path'])
    print(f"Model saved as '{CONFIG['model_save_path']}'")
    
    return model, history, y_pred, y_pred_proba


if __name__ == "__main__":
    # Execute the pipeline
    model, history, y_pred, y_pred_proba = main_pipeline()
    
    print("\nPipeline completed successfully!")
    print(f"Model saved to: {CONFIG['model_save_path']}")
    print(f"Training plots saved to: training_history.png, confusion_matrix.png")
