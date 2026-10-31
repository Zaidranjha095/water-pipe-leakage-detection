"""Load weights/model.pth and print class probabilities for one audio clip.

Usage:
    python predict.py --audio samples/leak.wav
    python predict.py --audio path/to/clip.wav --weights weights/model.pth
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import build_model, load_waveform, log_mel


def load_checkpoint(weights: Path):
    try:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(weights, map_location="cpu")
    classes = checkpoint["classes"]
    model = build_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    mel_fb = np.asarray(checkpoint["mel_filterbank"], dtype=np.float32)
    return model, classes, checkpoint, mel_fb


def predict(audio_path: Path, weights: Path) -> dict:
    model, classes, checkpoint, mel_fb = load_checkpoint(weights)
    waveform = load_waveform(audio_path, sr=checkpoint.get("sample_rate", 16000), duration=checkpoint.get("duration", 1.0))
    features = log_mel(waveform, mel_fb)[None, None]
    batch = torch.from_numpy(features)
    with torch.no_grad():
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)[0]
    pairs = sorted(
        ((classes[index], float(probabilities[index])) for index in range(len(classes))),
        key=lambda item: item[1],
        reverse=True,
    )
    return {
        "audio": str(audio_path),
        "class": pairs[0][0],
        "confidence": round(pairs[0][1], 6),
        "probabilities": {name: round(score, 6) for name, score in pairs},
        "logits": [round(float(value), 6) for value in logits[0]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a leak class for one audio clip.")
    parser.add_argument("--audio", required=True, help="Path to a WAV (or other librosa-readable) clip.")
    parser.add_argument("--weights", default="weights/model.pth", help="Checkpoint saved by train.py.")
    args = parser.parse_args()
    print(json.dumps(predict(Path(args.audio), Path(args.weights))))


if __name__ == "__main__":
    main()
