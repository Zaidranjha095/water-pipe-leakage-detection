"""Train the spectrogram CNN, export weights/ONNX/metrics, and copy demo samples."""
from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from model import (
    ARCH,
    CLASSES,
    DURATION,
    HOP_LENGTH,
    N_FFT,
    N_FRAMES,
    N_MELS,
    SAMPLE_RATE,
    build_mel_filterbank,
    build_model,
    load_waveform,
    log_mel,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
WEIGHTS_DIR = ROOT / "weights"
OUTPUTS_DIR = ROOT / "outputs"
SAMPLES_DIR = ROOT / "samples"
SEED = 42
EPOCHS = 40
PATIENCE = 8
BATCH_SIZE = 32
LR = 1e-3


class SpectrogramSet(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.features = features
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        spec = self.features[index].copy()
        if self.augment:
            if random.random() < 0.5:
                shift = random.randint(-4, 4)
                spec = np.roll(spec, shift, axis=1)
            if random.random() < 0.4:
                f0 = random.randint(0, spec.shape[0] - 1)
                width = random.randint(1, 4)
                spec[f0 : f0 + width] = spec.min()
            if random.random() < 0.4:
                t0 = random.randint(0, spec.shape[1] - 1)
                width = random.randint(1, 6)
                spec[:, t0 : t0 + width] = spec.min()
            spec = spec + np.random.normal(0, 0.4, spec.shape).astype(np.float32)
        tensor = torch.from_numpy(spec[None])
        return tensor, int(self.labels[index])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if not torch.cuda.is_available():
        print("CUDA is not available. Training on CPU.")
        return torch.device("cpu")
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"Using {name} (sm_{major}{minor})")
    try:
        probe = torch.randn(64, 64, device="cuda")
        probe = probe @ probe
        torch.cuda.synchronize()
        del probe
    except Exception as exc:
        print(f"CUDA smoke test failed ({exc}). Training on CPU.")
        return torch.device("cpu")
    return torch.device("cuda")


def collect_files() -> tuple[list[Path], list[int]]:
    paths: list[Path] = []
    labels: list[int] = []
    for index, name in enumerate(CLASSES):
        folder = DATA_DIR / name
        files = sorted(path for path in folder.glob("*") if path.is_file())
        if not files:
            raise SystemExit(f"No clips in {folder}. Run python download_dataset.py first.")
        paths.extend(files)
        labels.extend([index] * len(files))
    return paths, labels


def split_paths(paths: list[Path], labels: list[int]):
    train_paths, rest_paths, train_y, rest_y = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=SEED
    )
    val_paths, test_paths, val_y, test_y = train_test_split(
        rest_paths, rest_y, test_size=0.50, stratify=rest_y, random_state=SEED
    )
    return (train_paths, train_y), (val_paths, val_y), (test_paths, test_y)


def features_for(paths: list[Path], mel_fb: np.ndarray) -> np.ndarray:
    return np.stack([log_mel(load_waveform(path), mel_fb) for path in paths]).astype(np.float32)


def run_epoch(model, loader, optimizer, loss_fn, device, train: bool) -> tuple[float, float]:
    model.train(train)
    total = 0.0
    correct = 0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        if train:
            loss.backward()
            optimizer.step()
        total += float(loss.item()) * len(yb)
        correct += int((logits.argmax(1) == yb).sum().item())
        n += len(yb)
    return total / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def predict_split(model, features: np.ndarray, device) -> np.ndarray:
    model.eval()
    xb = torch.from_numpy(features[:, None]).to(device)
    logits = model(xb)
    return logits.argmax(1).cpu().numpy(), logits.cpu().numpy()


def export_onnx(model, device, path: Path) -> None:
    model.eval()
    dummy = torch.zeros(1, 1, N_MELS, N_FRAMES, device=device)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["spectrogram"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )


def copy_samples(model, test_paths, test_y, test_features, device) -> dict[str, str]:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    preds, logits = predict_split(model, test_features, device)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    chosen: dict[str, str] = {}
    for class_index, name in enumerate(CLASSES):
        best = None
        best_conf = -1.0
        for row, path in enumerate(test_paths):
            if test_y[row] == class_index and preds[row] == class_index:
                conf = float(probs[row, class_index])
                if conf > best_conf:
                    best_conf = conf
                    best = path
        if best is None:
            best = next(path for path, label in zip(test_paths, test_y) if label == class_index)
        dest = SAMPLES_DIR / f"{name}.wav"
        if best.suffix.lower() != ".wav":
            import soundfile as sf

            y = load_waveform(best)
            sf.write(dest, y, SAMPLE_RATE)
        else:
            shutil.copy2(best, dest)
        chosen[name] = dest.name
    return chosen


def count_map(labels: list[int]) -> dict[str, int]:
    return {name: int(sum(1 for item in labels if item == index)) for index, name in enumerate(CLASSES)}


def main() -> None:
    samples_only = "--samples-only" in sys.argv
    set_seed(SEED)
    device = select_device()
    paths, labels = collect_files()
    (train_paths, train_y), (val_paths, val_y), (test_paths, test_y) = split_paths(paths, labels)
    mel_fb = build_mel_filterbank()
    print("Computing log-mel spectrograms...")
    train_x = features_for(train_paths, mel_fb)
    val_x = features_for(val_paths, mel_fb)
    test_x = features_for(test_paths, mel_fb)
    print(f"Spectrogram shape: {train_x[0].shape} (expected {N_MELS}x{N_FRAMES})")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if samples_only:
        checkpoint = torch.load(WEIGHTS_DIR / "model.pth", map_location=device, weights_only=False)
        model = build_model(num_classes=len(CLASSES), pretrained=False).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        copy_samples(model, test_paths, test_y, test_x, device)
        print("Refreshed samples/")
        return

    train_loader = DataLoader(SpectrogramSet(train_x, np.array(train_y), augment=True), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SpectrogramSet(val_x, np.array(val_y)), batch_size=BATCH_SIZE)

    counts = np.bincount(train_y, minlength=len(CLASSES)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1) / len(CLASSES)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    model = build_model(num_classes=len(CLASSES), pretrained=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = []
    best_val = -1.0
    best_state = None
    stale = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, loss_fn, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer, loss_fn, device, False)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )
        print(f"epoch {epoch:02d}  train {train_acc:.3f}  val {val_acc:.3f}  loss {train_loss:.4f}")
        if val_acc > best_val + 1e-4:
            best_val = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.to(device)
    preds, logits = predict_split(model, test_x, device)
    report = classification_report(test_y, preds, target_names=CLASSES, output_dict=True, zero_division=0)
    matrix = confusion_matrix(test_y, preds, labels=list(range(len(CLASSES)))).tolist()
    per_class = {
        name: {
            "precision": float(report[name]["precision"]),
            "recall": float(report[name]["recall"]),
            "f1": float(report[name]["f1-score"]),
            "support": int(report[name]["support"]),
        }
        for name in CLASSES
    }
    parameter_count = sum(p.numel() for p in model.parameters())
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    metrics = {
        "task": "audio-classification",
        "class_order": CLASSES,
        "counts": {
            "all": count_map(labels),
            "train": {**count_map(train_y), "total": len(train_y)},
            "val": {**count_map(val_y), "total": len(val_y)},
            "test": {**count_map(test_y), "total": len(test_y)},
            "total": len(labels),
        },
        "training": {
            "seed": SEED,
            "epochs_ran": len(history),
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "best_val_accuracy": float(best_val),
            "class_weights": {name: float(weights[index]) for index, name in enumerate(CLASSES)},
            "parameter_count": int(parameter_count),
            "gpu": gpu_name,
            "arch": ARCH,
            "sample_rate": SAMPLE_RATE,
            "duration": DURATION,
            "n_mels": N_MELS,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "n_frames": N_FRAMES,
        },
        "history": history,
        "test": {
            "accuracy": float((np.array(test_y) == preds).mean()),
            "macro_f1": float(f1_score(test_y, preds, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(test_y, preds, average="weighted", zero_division=0)),
            "confusion_matrix": matrix,
            "per_class": per_class,
        },
    }
    checkpoint = {
        "state_dict": model.cpu().state_dict(),
        "classes": CLASSES,
        "arch": ARCH,
        "sample_rate": SAMPLE_RATE,
        "duration": DURATION,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "n_frames": N_FRAMES,
        "mel_filterbank": mel_fb.tolist(),
    }
    torch.save(checkpoint, WEIGHTS_DIR / "model.pth")
    config = {
        "classes": CLASSES,
        "arch": ARCH,
        "sample_rate": SAMPLE_RATE,
        "duration": DURATION,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "n_frames": N_FRAMES,
        "output": "logits",
        "mel_filterbank": mel_fb.tolist(),
    }
    (WEIGHTS_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (WEIGHTS_DIR / "class_names.json").write_text(json.dumps(CLASSES, indent=2), encoding="utf-8")
    (OUTPUTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUTPUTS_DIR / "classification_report.txt").write_text(
        classification_report(test_y, preds, target_names=CLASSES, zero_division=0),
        encoding="utf-8",
    )

    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 4.2))
    plt.plot(epochs, [row["train_accuracy"] for row in history], label="train accuracy")
    plt.plot(epochs, [row["val_accuracy"] for row in history], label="val accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "training_curves.png", dpi=140)
    plt.close()

    plt.figure(figsize=(5.2, 4.6))
    plt.imshow(matrix, cmap="Greens")
    plt.xticks(range(len(CLASSES)), CLASSES, rotation=20, ha="right")
    plt.yticks(range(len(CLASSES)), CLASSES)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            plt.text(j, i, str(value), ha="center", va="center")
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=140)
    plt.close()

    model.to(device)
    export_onnx(model, device, WEIGHTS_DIR / "model.onnx")
    copy_samples(model, test_paths, test_y, test_x, device)
    print(json.dumps({"accuracy": metrics["test"]["accuracy"], "macro_f1": metrics["test"]["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
