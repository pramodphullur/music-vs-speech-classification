import os
import librosa
import numpy as np
import matplotlib
import logging
from pathlib import Path

# Constants
TARGET_SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
MAX_DURATION = 5.0
FIXED_LENGTH = int(TARGET_SR * MAX_DURATION / HOP_LENGTH)

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioPreprocessor:
    def __init__(self, target_sr=TARGET_SR, n_mels=N_MELS, max_duration=MAX_DURATION):
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.max_duration = max_duration
        self.fixed_length = int(target_sr * max_duration / HOP_LENGTH)

    def normalize_spectrogram(self, spec, axis=(0, 1)):
        min_val = np.min(spec, axis=axis, keepdims=True)
        max_val = np.max(spec, axis=axis, keepdims=True)
        range_val = np.where(max_val - min_val == 0, 1, max_val - min_val)
        return np.clip((spec - min_val) / range_val, 0, 1)

    def pad_or_truncate(self, spec, target_length):
        cur_len = spec.shape[1]
        if cur_len > target_length:
            start = (cur_len - target_length) // 2
            return spec[:, start:start + target_length]
        elif cur_len < target_length:
            pad_left = (target_length - cur_len) // 2
            pad_right = target_length - cur_len - pad_left
            return np.pad(spec, ((0, 0), (pad_left, pad_right)), mode='constant')
        return spec

    def create_depth_layers(self, spec_2d, n_layers=8):
        cmap = matplotlib.colormaps.get_cmap('viridis')
        rgba = cmap(spec_2d)
        rgb = rgba[:, :, :3]
        intensity = np.dot(rgb, [0.299, 0.587, 0.114])
        bins = np.linspace(0, 1, n_layers + 1)
        H, W = spec_2d.shape
        layers = np.zeros((n_layers, H, W, 1), dtype=np.float32)
        for i in range(n_layers):
            mask = (intensity >= bins[i]) & (intensity < bins[i + 1])
            layers[i, :, :, 0] = np.where(mask, intensity, 0)
        return layers

    def process_audio_file(self, path):
        try:
            y, _ = librosa.load(path, sr=self.target_sr)
            max_samples = int(self.target_sr * self.max_duration)
            if len(y) > max_samples:
                y = y[:max_samples]
            else:
                y = np.pad(y, (0, max_samples - len(y)), mode='constant')
            mel = librosa.feature.melspectrogram(y, sr=self.target_sr, n_mels=self.n_mels, n_fft=N_FFT, hop_length=HOP_LENGTH)
            log_mel = librosa.power_to_db(mel, ref=np.max)
            norm = self.normalize_spectrogram(log_mel)
            norm = self.pad_or_truncate(norm, self.fixed_length)
            depth = self.create_depth_layers(norm)
            return depth
        except Exception as e:
            logger.error(f"Error processing {path}: {str(e)}")
            return None

class MusicSpeechDataset:
    def __init__(self, data_dir, preprocessor):
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor

    def load_data_from_directory(self, dir_path, label, class_name):
        specs, labels = [], []
        dir_path = Path(dir_path)
        if not dir_path.exists():
            logger.warning(f"Directory {dir_path} not found")
            return specs, labels
        files = list(dir_path.glob("*.wav"))
        logger.info(f"Loading {len(files)} {class_name} files from {dir_path}")
        for f in files:
            proc = self.preprocessor.process_audio_file(f)
            if proc is not None:
                specs.append(proc)
                labels.append(label)
        logger.info(f"Loaded {len(specs)} {class_name} samples")
        return specs, labels

    def prepare_dataset(self):
        X, y = [], []
        speech_train, label_speech_train = self.load_data_from_directory(self.data_dir / "train/speech", 0, "speech")
        music_train, label_music_train = self.load_data_from_directory(self.data_dir / "train/music", 1, "music")
        speech_test, label_speech_test = self.load_data_from_directory(self.data_dir / "test/speech", 0, "speech")
        music_test, label_music_test = self.load_data_from_directory(self.data_dir / "test/music", 1, "music")
        X = speech_train + music_train + speech_test + music_test
        y = label_speech_train + label_music_train + label_speech_test + label_music_test
        if not X:
            raise ValueError("No data loaded.")
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        logger.info(f"Dataset shape: {X.shape}, Labels shape: {y.shape}")
        logger.info(f"Speech samples: {np.sum(y == 0)}, Music samples: {np.sum(y == 1)}")
        return X, y

