"""Compare Python log-mel with a JS-equivalent numpy path for one clip."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from model import load_waveform, log_mel, N_FFT, HOP_LENGTH, N_MELS, N_FRAMES, AMIN, TOP_DB
from predict import load_checkpoint, predict


def js_hann(size: int) -> np.ndarray:
    if size == 1:
        return np.ones(1, dtype=np.float32)
    i = np.arange(size, dtype=np.float32)
    return (0.5 - 0.5 * np.cos((2 * np.pi * i) / (size - 1))).astype(np.float32)


def np_hann(size: int) -> np.ndarray:
    return np.hanning(size).astype(np.float32)


def stft(y: np.ndarray, window: np.ndarray) -> np.ndarray:
    frames = []
    for start in range(0, len(y) - N_FFT + 1, HOP_LENGTH):
        frame = y[start : start + N_FFT] * window
        spectrum = np.fft.rfft(frame, n=N_FFT)
        frames.append((np.abs(spectrum) ** 2).astype(np.float32))
    return np.stack(frames, axis=1)


def logmel(y: np.ndarray, bank: np.ndarray, window: np.ndarray) -> np.ndarray:
    power = stft(y, window)
    mel = bank @ power
    log_spec = 10.0 * np.log10(np.maximum(AMIN, mel))
    log_spec -= log_spec.max()
    return np.maximum(log_spec, -TOP_DB).astype(np.float32)


def main() -> None:
    model, classes, checkpoint, mel_fb = load_checkpoint(Path("weights/model.pth"))
    print("hann max abs diff", float(np.max(np.abs(js_hann(N_FFT) - np_hann(N_FFT)))))
    for name in ["leak", "no_leak", "environmental_noise"]:
        path = Path("samples") / f"{name}.wav"
        y = load_waveform(path)
        py = log_mel(y, mel_fb)
        js = logmel(y, mel_fb, js_hann(N_FFT))
        npw = logmel(y, mel_fb, np_hann(N_FFT))
        print(name, "py vs js", float(np.max(np.abs(py - js))), "py vs nphann", float(np.max(np.abs(py - npw))), "shape", py.shape)
        print(" python predict", predict(path, Path("weights/model.pth")))
    print("n_mels n_frames", N_MELS, N_FRAMES, "classes", classes)


if __name__ == "__main__":
    main()
