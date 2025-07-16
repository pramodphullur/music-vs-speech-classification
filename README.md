# music-vs-speech-classification
# 3D CNN Audio Classification: Music vs Speech

A deep learning project that classifies audio files as either music or speech using 3D Convolutional Neural Networks with spectrogram image representations.

## Project Overview

This project implements a novel approach to audio classification by:
1. Converting audio spectrograms into multi-layered RGB images using colormaps
2. Binning pixels by intensity to create 3D representations
3. Using 3D CNNs to classify the resulting 4D tensor data

## Dataset

The project uses a music-speech dataset with the following structure:
```
dataset/
├── train/
│   ├── music/
│   └── speech/
└── test/
    ├── music/
    └── speech/
```

- **Total samples**: 128 (64 speech, 64 music)
- **Training set**: 89 samples
- **Validation set**: 19 samples  
- **Test set**: 20 samples

## Architecture

### Model Structure
- **Input Shape**: (10, 128, 1292, 4) - 4D tensor representing binned spectrogram layers
- **Architecture**: 3D CNN with three convolutional blocks
- **Parameters**: 347,234 total parameters (1.32 MB)

### Layer Details
1. **Conv3D Block 1**: 32 filters, 3x3x3 kernel, BatchNorm, MaxPooling3D, Dropout(0.25)
2. **Conv3D Block 2**: 64 filters, 3x3x3 kernel, BatchNorm, MaxPooling3D, Dropout(0.25)
3. **Conv3D Block 3**: 128 filters, 3x3x3 kernel, BatchNorm, MaxPooling3D, Dropout(0.3)
4. **Classification Head**: GlobalAveragePooling3D → Dense(256) → Dense(128) → Dense(2)

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Dependencies
- Python 3.7+
- TensorFlow 2.x
- librosa
- scikit-learn
- matplotlib
- numpy
- opencv-python

## Usage

### Training
```python
# Run the complete pipeline
python audio_classifier_3d.py
```

### Inference
```python
from audio_classifier_3d import predict_new_audio

# Load trained model
model = tf.keras.models.load_model('my_model.keras')

# Predict on new audio file
predicted_class, confidence = predict_new_audio(model, 'path/to/audio.wav')
```

## Results

### Current Performance
- **Test Accuracy**: 50%
- **Training Issues**: Model shows overfitting with validation accuracy stuck at 47.37%

## Data Processing Pipeline

1. **Audio Loading**: Load WAV files with librosa at 22050 Hz
2. **Mel Spectrogram**: Generate mel-scale spectrograms
3. **Log Transformation**: Convert to log scale (dB)
4. **Normalization**: Normalize to [0,1] range
5. **Colormap Application**: Apply 'magma' colormap for RGB conversion
6. **Intensity Binning**: Create 10 layers based on pixel intensity bins
7. **4D Tensor Creation**: Generate final (10, H, W, 4) representation

## Known Issues

1. **Training Performance**: Model overfits quickly, validation accuracy remains constant
2. **Class Imbalance**: Model predicts only music class during inference
3. **Data Preprocessing**: Complex 4D tensor creation may be overcomplicated

## Potential Improvements

### Model Architecture
- Add regularization techniques (L2, weight decay)
- Implement data augmentation
- Try different pooling strategies
- Experiment with attention mechanisms

### Data Processing
- Simplify to 2D CNN with standard spectrogram inputs
- Add temporal augmentation (time stretching, pitch shifting)
- Implement proper feature scaling

### Training Strategy
- Use class weights for balanced training
- Implement curriculum learning
- Add early stopping with better patience
- Try different optimizers (AdamW, RMSprop)

## Project Structure

```
audio-classification-3d/
├── src/
│   ├── audio_classifier_3d.py      # Main training script
│   ├── data_preprocessing.py       # Data loading and preprocessing
│   ├── model_architecture.py       # 3D CNN model definition
│   └── evaluation.py               # Model evaluation utilities
├── models/
│   └── saved_models/               # Trained model checkpoints
├── dataset/
│   ├── train/                      # .wav formate
│   └── test/                       # .wav formate
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── LICENSE                         # MIT License
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: Music-Speech Classification Dataset
- Inspired by research in audio signal processing and 3D deep learning
- Built with TensorFlow and librosa

## 📧 Contact

For questions or suggestions, please open an issue or contact [ram.hullur.backup@gmail.com](mailto:ram.hullur.backup@gmail.com).

---

**Note**: This project is experimental and the current results show room for improvement. Contributions and suggestions are welcome!
