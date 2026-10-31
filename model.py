"""Spectrogram CNN for three-class water-pipe leak audio."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

ARCH = "spectrogram_cnn"
CLASSES = ["leak", "no_leak", "environmental_noise"]
SAMPLE_RATE = 16000
DURATION = 1.0
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
N_SAMPLES = int(SAMPLE_RATE * DURATION)
N_FRAMES = 1 + (N_SAMPLES - N_FFT) // HOP_LENGTH
TOP_DB = 80.0
AMIN = 1e-10


def build_mel_filterbank() -> np.ndarray:
    import librosa

    return librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, htk=False, norm="slaney").astype(
        np.float32
    )


def load_waveform(path, sr: int = SAMPLE_RATE, duration: float = DURATION) -> np.ndarray:
    import librosa

    y, _ = librosa.load(path, sr=sr, mono=True)
    target = int(sr * duration)
    if y.size < target:
        y = np.pad(y, (0, target - y.size))
    else:
        y = y[:target]
    return y.astype(np.float32)


def stft_power(y: np.ndarray, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH) -> np.ndarray:
    window = np.hanning(n_fft).astype(np.float32)
    frames = []
    for start in range(0, len(y) - n_fft + 1, hop_length):
        frame = y[start : start + n_fft] * window
        spectrum = np.fft.rfft(frame, n=n_fft)
        frames.append((np.abs(spectrum) ** 2).astype(np.float32))
    return np.stack(frames, axis=1)


def power_to_db(mel: np.ndarray, amin: float = AMIN, top_db: float = TOP_DB) -> np.ndarray:
    log_spec = 10.0 * np.log10(np.maximum(amin, mel))
    log_spec -= log_spec.max()
    return np.maximum(log_spec, -top_db).astype(np.float32)


def log_mel(y: np.ndarray, mel_fb: np.ndarray | None = None) -> np.ndarray:
    bank = mel_fb if mel_fb is not None else build_mel_filterbank()
    power = stft_power(y)
    mel = bank @ power
    return power_to_db(mel)


class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes: int = len(CLASSES)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.35), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


def build_model(num_classes: int = len(CLASSES), pretrained: bool = False) -> SpectrogramCNN:
    del pretrained
    return SpectrogramCNN(num_classes=num_classes)
