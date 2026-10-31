# Smartphone Audio Water-Pipe Leakage Detection

Deep Learning semester project. A spectrogram CNN classifies one-second water-pipe audio as leak, no leak, or environmental noise, trained on the public [Zenodo 18631450](https://zenodo.org/records/18631450) acoustic set.

**Test accuracy:** 94.00%  
**Macro F1:** 93.13%  
**Clips:** 1000 (700 train / 150 validation / 150 test)

## Load the weights

The checkpoint is a normal Git file at `weights/model.pth` (no Git LFS). It contains `state_dict`, `classes`, sample rate 16000, duration 1.0, mel settings, and `arch`.

```python
import torch
from model import build_model

checkpoint = torch.load("weights/model.pth", map_location="cpu", weights_only=False)
model = build_model(num_classes=len(checkpoint["classes"]), pretrained=False)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
```

Class order: leak, no_leak, environmental_noise. The network returns logits.

```bash
python predict.py --audio samples/leak.wav
```

`model_manifest.json` lists the same paths. `weights/model.onnx` is the browser copy.

## Project layout

| Path | Role |
| --- | --- |
| `train.py` | Stratified split, CNN training, metrics, ONNX export |
| `model.py` | Spectrogram CNN and log-mel features |
| `predict.py` | JSON prediction for one audio clip |
| `download_dataset.py` | Downloads the Zenodo RAR archives |
| `collect_audio.py` | Optional 1 s smartphone recordings |
| `weights/` | Checkpoint, ONNX, class names, config |
| `outputs/metrics.json` | Accuracy, F1, confusion matrix, training history |
| `report/PROJECT_REPORT.md` | Written report |
| `presentation/presentation.pptx` | Slides with speaker notes |
| `presentation/VOICEOVER_SCRIPT.md` | Text to read for the YouTube recording |
| `web/` | Static site deployed on Vercel |

## Reproduce

```bash
python -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python download_dataset.py
.venv\Scripts\python train.py
.venv\Scripts\python build_docs.py
node prepare_site.js
```

The CUDA 12.8 wheel is required for an RTX 50-series GPU. `train.py` uses the CPU if CUDA cannot run a kernel. The audio archive stays in `data/` and is gitignored.

## Site

`node prepare_site.js` copies the weights, plots, samples, report, and slides into `web/`. Deploy the repository on Vercel. The build command is already in `vercel.json`.

## Dataset citation

Acoustic data for the manuscript entitled 'Self-supervised acoustic leakage detection for water distribution systems: A real-time diagnosis framework under data scarcity'. Zenodo, 2026. https://doi.org/10.5281/zenodo.18631450

Zaid Ahmed · SU92-BSCSM-F23-553 · Deep Learning
