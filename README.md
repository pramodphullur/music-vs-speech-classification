# Music vs Speech Classification using 3D CNNs

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10+](https://img.shields.io/badge/tensorflow-2.10+-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research Paper](https://img.shields.io/badge/Research-Published-green.svg)](https://link.springer.com/chapter/10.1007/978-981-97-8861-3_24)

**An improved implementation of our published research on audio classification using 3D CNNs with multi-layered spectrogram representations.**

> **📄 Published Research**: This project is based on our paper published in Springer's proceedings: *"[Music Versus Speech Classification Using 3D CNN]"* - [DOI: 10.1007/978-981-97-8861-3_24](https://link.springer.com/chapter/10.1007/978-981-97-8861-3_24)

A professional-grade deep learning system that implements and improves upon our novel approach to audio classification using 3D Convolutional Neural Networks applied to multi-layered spectrogram representations.

## Project Overview

This project implements and significantly improves upon our published research methodology for audio classification:

1. **Converting audio to mel-spectrograms** with enhanced preprocessing pipeline
2. **Creating depth layers** through intensity-based binning of spectrogram visualizations
3. **Applying optimized 3D CNNs** to the resulting 4D tensor representations
4. **Professional production pipeline** with robust error handling and scalability

### Research Innovation

**Based on Our Published Work**: Our original research introduced a novel approach to audio classification by treating spectrograms as multi-layered images. This implementation includes several key improvements over the original paper:

- **Enhanced Architecture**: Optimized 3D CNN design with better regularization
- **Improved Data Pipeline**: Professional-grade preprocessing with error handling
- **Standard Datasets**: Integration with GTZAN and LibriSpeech benchmarks
- **Production Ready**: Model persistence, batch processing, and deployment features
- **Better Performance**: Addresses overfitting issues from the original implementation

### Key Innovation from Our Research
Unlike traditional 2D approaches, our method creates pseudo-3D representations of audio spectrograms by binning pixel intensities into depth layers, allowing 3D CNNs to capture both spectral-temporal and intensity-based patterns simultaneously.

## Datasets

**Note**: Datasets are not included in this repository to keep it lightweight and avoid licensing issues.

### Music Data: GTZAN Dataset
- **Source**: [Kaggle - GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- **Description**: 1,000 music tracks across 10 genres (30 seconds each)
- **Format**: WAV files, 22050 Hz
- **Usage**: Industry standard for music genre classification research

### Speech Data: LibriSpeech ASR Corpus  
- **Source**: [OpenSLR - LibriSpeech](https://www.openslr.org/12)
- **Description**: Large-scale corpus of read English speech (~1000 hours)
- **Format**: FLAC files (auto-converted to WAV)
- **License**: CC BY 4.0

## 🚀 Quick Start

### Step 1: Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/music-vs-speech-classification.git
cd music-vs-speech-classification

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Datasets

#### Option A: Using Kaggle CLI (GTZAN)
```bash
# Install Kaggle CLI and configure API key
pip install kaggle
kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification
unzip gtzan-dataset-music-genre-classification.zip
```

#### Option B: Manual Download
1. **GTZAN**: Download from [Kaggle link](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
2. **LibriSpeech**: Download from [OpenSLR](https://www.openslr.org/12) (recommend `dev-clean` for testing)

### Step 3: Prepare Dataset

```python
from music_speech_classifier import prepare_datasets

# One-time dataset preparation
prepare_datasets(
    gtzan_path="path/to/gtzan/Data/genres_original",
    librispeech_path="path/to/LibriSpeech"
)
```

### Step 4: Train Model

```python
from music_speech_classifier import main

# Execute complete training pipeline
classifier, history, test_accuracy = main()
print(f"Final test accuracy: {test_accuracy:.4f}")
```

### Step 5: Use Trained Model

```python
# Predict on new audio files
result = classifier.predict_audio_file("your_audio.wav")
print(f"Prediction: {result['class']} (confidence: {result['confidence']:.4f})")
```

## Architecture

### Data Processing Pipeline

```
Audio File (.wav/.flac)
    ↓
Load & Normalize (22050 Hz, 5 seconds)
    ↓
Mel-Spectrogram (128 mel bands, 2048 FFT)
    ↓
Log-Scale Transformation (dB)
    ↓
Min-Max Normalization [0,1]
    ↓
Colormap Application (Viridis)
    ↓
Intensity Binning (8 depth layers)
    ↓
4D Tensor: (8, 128, 216, 1)
```

### Model Architecture

```
Input: (8, 128, 216, 1)  # (depth, mel_bands, time_frames, channels)
    ↓
Conv3D(16) + BatchNorm + MaxPool3D(2,2,2) + Dropout(0.2)
    ↓
Conv3D(32) + BatchNorm + MaxPool3D(2,2,2) + Dropout(0.3)
    ↓
Conv3D(64) + BatchNorm + GlobalAvgPool3D + Dropout(0.4)
    ↓
Dense(128) + Dropout(0.5)
    ↓
Dense(64) + Dropout(0.5)
    ↓
Dense(2, softmax)  # [Speech, Music]
```

## Performance

### Expected Results
- **Test Accuracy**: 85-95%
- **Precision**: 90%+ for both classes
- **Recall**: 85%+ for both classes
- **F1-Score**: 87%+ average

### Training Configuration
- **Optimizer**: Adam (lr=0.001, decay)
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 8 (memory optimized)
- **Epochs**: 100 (early stopping enabled)
- **Data Split**: 70% train, 15% validation, 15% test

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
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── LICENSE                         # MIT License
```

## Advanced Usage

### Custom Training Parameters

```python
from music_speech_classifier import MusicSpeechClassifier

classifier = MusicSpeechClassifier("./dataset")
X_train, X_val, X_test, y_train, y_val, y_test = classifier.prepare_data()

# Custom training
history = classifier.build_and_train_model(
    X_train, y_train, X_val, y_val,
    epochs=150,
    batch_size=4  # Reduce if memory issues
)
```

### Model Evaluation

```python
# Detailed evaluation
y_pred, y_pred_proba, accuracy = classifier.evaluate_model(X_test, y_test)

# Load saved model
classifier.load_model("path/to/saved/model.keras")
```

### Batch Prediction

```python
import os
from pathlib import Path

# Predict on multiple files
audio_dir = Path("path/to/audio/files")
results = []

for audio_file in audio_dir.glob("*.wav"):
    result = classifier.predict_audio_file(audio_file)
    results.append({
        'file': audio_file.name,
        'prediction': result['class'],
        'confidence': result['confidence']
    })
```

## Troubleshooting

### Common Issues & Solutions

#### 1. Out of Memory Errors
```python
# Reduce batch size in main()
batch_size = 4  # or even 2 for limited GPU memory
```

#### 2. Dataset Not Found
```bash
# Verify directory structure
ls dataset/train/music/    # Should show .wav files and
ls dataset/train/speech/   # Should show .wav files

# Re-run dataset preparation if empty
python -c "from music_speech_classifier import prepare_datasets; prepare_datasets('gtzan_path', 'librispeech_path')"
```

#### 3. Audio Loading Issues
```bash
# Install additional audio codecs
pip install soundfile librosa[display] audioread
```

#### 4. CUDA/GPU Problems
```python
# Check GPU availability
import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))

# Force CPU if needed
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

#### 5. Low Accuracy/Overfitting
- Ensure balanced dataset (equal music/speech samples)
- Increase dropout rates
- Add data augmentation
- Reduce model complexity

## Performance Optimization

### For Better Results:
1. **Data Augmentation**: Add time/pitch shifting
2. **Ensemble Methods**: Combine multiple models
3. **Hyperparameter Tuning**: Use Optuna or similar
4. **Advanced Architectures**: Try attention mechanisms

### For Production Deployment:
1. **Model Quantization**: Reduce model size
2. **TensorFlow Lite**: Mobile deployment
3. **TensorFlow Serving**: Server deployment
4. **ONNX Conversion**: Cross-platform compatibility

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/music-vs-speech-classification.git

# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pytest  # Code formatting and testing

# Run tests
pytest tests/
```

## 📚 Research & References

### 📄 Primary Research (Our Work)
**Main Publication**: 
- **Authors**: [Pramod Hullur, Dhanya Kulkarni, Vinayak Jainapur, Satish Chikkamath, S. R. Nirmala & Suneeta V. Budihal ]
- **Title**: [Music Versus Speech Classification Using 3D CNN]
- **Conference/Journal**: Springer Proceedings
- **DOI**: [10.1007/978-981-97-8861-3_24](https://link.springer.com/chapter/10.1007/978-981-97-8861-3_24)
- **Year**: 2024

**Key Contributions from Our Research**:
- Novel 3D representation of audio spectrograms through intensity binning
- Application of 3D CNNs to multi-layered spectrogram data
- Comparative analysis of 2D vs 3D approaches for audio classification
- Experimental validation on music-speech classification tasks

### 🔗 Related Academic Work
- **GTZAN**: Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. *IEEE Transactions on Speech and Audio Processing*.
- **LibriSpeech**: Panayotov, V., et al. (2015). Librispeech: an ASR corpus based on public domain audio books. *ICASSP*.
- **3D CNNs**: Tran, D., et al. (2015). Learning spatiotemporal features with 3D convolutional networks. *ICCV*.

### 📊 Implementation Improvements Over Original Paper
This codebase includes several enhancements beyond our published work:

1. **Architecture Optimization**: Reduced overfitting through better regularization
2. **Standard Benchmarks**: Integration with established datasets (GTZAN, LibriSpeech)
3. **Production Pipeline**: Professional-grade code with error handling and logging
4. **Scalability**: Batch processing and model persistence features
5. **Performance**: Improved accuracy through better preprocessing and training strategies

### 🎓 Citation
If you use this work or build upon our research, please cite our paper:

```bibtex
@InProceedings{Music vs Speech Classification Using 3D CNN,
    author={[Hullur, Pramod
    and Kulkarni, Dhanya
    and Jainapur, Vinayak
    and Chikkamath, Satish
    and Nirmala, S. R.
    and Budihal, Suneeta V.]},
    title={[Music Versus Speech Classification Using 3D CNN]},
    booktitle={[Proceedings of 5th International Conference on Recent Trends in Machine Learning, IoT, Smart Cities and Applications]},
    year={2025},
    publisher={Springer Nature Singapore},
    address={Singapore},
    pages={273--282},
    isbn={978-981-97-8861-3},
    doi={10.1007/978-981-97-8861-3_24},
    url={https://link.springer.com/chapter/10.1007/978-981-97-8861-3_24}
}

```

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Springer**: For publishing our research and supporting academic contribution
- **Research Community**: Colleagues and reviewers who provided valuable feedback on our original paper
- **OpenSLR**: For providing the LibriSpeech corpus for speech research
- **GTZAN**: Dataset creators for enabling music classification research
- **Open Source Community**: TensorFlow, librosa, and other library maintainers
- **Academic Institutions**: For supporting research and development

### 🌟 Research Impact
Our published work contributes to the growing field of deep learning applications in audio processing, specifically:
- Novel approaches to spectrogram representation
- 3D CNN applications beyond video processing
- Multi-modal audio feature extraction techniques

## 📧 Support & Contact

### For Research Collaboration
- **Primary Author**: [ram.hullur.backup@gmail.com](mailto:ram.hullur.backup@gmail.com)
- **Research Inquiries**: Please reference our published paper [DOI: 10.1007/978-981-97-8861-3_24](https://link.springer.com/chapter/10.1007/978-981-97-8861-3_24)
- **Academic Discussions**: [GitHub Discussions](https://github.com/pramodphullur/music-vs-speech-classification/discussions)

### For Technical Support
- **Issues**: [GitHub Issues](https://github.com/pramodphullur/music-vs-speech-classification/issues)
- **Code Questions**: Create an issue with the "question" label
- **Feature Requests**: Create an issue with the "enhancement" label

---

### 🌟 Star this repository if our research helped you!

**Research-backed implementation with production-ready improvements**

*This project implements our published research with significant enhancements for practical deployment and improved performance.*

---

> **Academic Note**: This implementation extends our peer-reviewed research published in Springer proceedings. The code includes optimizations and improvements developed post-publication to address scalability and performance issues identified in the original work.

> **Research Collaboration**: We welcome academic collaborations and research extensions. Please cite our work if you build upon this research.
