"""Write the report, slides, voiceover, LinkedIn draft, checklist, and README."""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent
INK = RGBColor(0x17, 0x21, 0x1C)
GREEN = RGBColor(0x1C, 0x46, 0x36)
AMBER = RGBColor(0xC4, 0xA1, 0x6A)
CREAM = RGBColor(0xF7, 0xF5, 0xF1)
PAPER = RGBColor(0xFF, 0xFD, 0xF8)
MUTED = RGBColor(0x5C, 0x67, 0x5F)
LINE = RGBColor(0xE3, 0xDD, 0xD4)
FONT = "Calibri"
SERIF = "Georgia"


def pct(value: float) -> str:
    return f"{100 * float(value):.2f}%"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def confusion_pairs(metrics: dict) -> list[tuple[int, str, str]]:
    classes = metrics["class_order"]
    matrix = metrics["test"]["confusion_matrix"]
    pairs = []
    for row, true_name in enumerate(classes):
        for column, pred_name in enumerate(classes):
            count = int(matrix[row][column])
            if row != column and count:
                pairs.append((count, true_name, pred_name))
    pairs.sort(key=lambda item: item[0], reverse=True)
    return pairs


def confusion_sentence(metrics: dict) -> str:
    pairs = confusion_pairs(metrics)
    if not pairs:
        return "Every test clip was classified correctly."
    bits = [f"{count} {true_name} predicted as {pred_name}" for count, true_name, pred_name in pairs[:3]]
    return "The largest confusions were " + "; ".join(bits) + "."


def display_url(config: dict, key: str, label: str) -> str:
    value = str(config.get(key, "")).strip()
    return value if value else f"[{label} — add this before submission]"


def write_report(config: dict, metrics: dict) -> None:
    classes = metrics["class_order"]
    counts = metrics["counts"]
    training = metrics["training"]
    test = metrics["test"]
    weights = training["class_weights"]
    matrix = test["confusion_matrix"]
    per_class_rows = "\n".join(
        f"| {name} | {pct(test['per_class'][name]['precision'])} | {pct(test['per_class'][name]['recall'])} | {pct(test['per_class'][name]['f1'])} | {test['per_class'][name]['support']} |"
        for name in classes
    )
    text = f"""# Smartphone Audio Water-Pipe Leakage Detection

## Abstract

This project trains a compact convolutional network to classify one-second acoustic clips of a water pipe as leak, no leak, or environmental noise. The public training set is 1,000 field recordings from a water-distribution leakage-detection site (Zenodo record 18631450). Audio is resampled to 16 kHz, converted to a 64-bin log-mel spectrogram, and classified by a four-block CNN. On a held-out test split of 150 clips the model reaches {pct(test['accuracy'])} accuracy and {pct(test['macro_f1'])} macro F1. Weights, an ONNX copy, a prediction script, and a browser demo are published for evaluation.

## Problem statement

Hidden household and distribution-pipe leaks waste water and can stay unnoticed until damage is visible. A short recording from a phone or a cheap microphone is a low-cost screening signal. This project asks whether a small deep network can separate leak sounds from a quiet pipe and from ordinary background noise.

The system is an educational demonstration. It is not a plumber, a utility alarm, or a safety controller.

## Objectives

- Train a convolutional classifier on a public water-pipe acoustic dataset.
- Keep a held-out test set out of model selection and report accuracy, per-class F1, and a confusion matrix.
- Publish `weights/model.pth`, `model.py:build_model()`, and `python predict.py --audio <path>`.
- Deploy a static page that runs the ONNX network in the browser and links the report and slides.
- Document an optional smartphone recording path for later domain-specific clips.

## Dataset

The clips come from the Zenodo record *Acoustic data for the manuscript entitled 'Self-supervised acoustic leakage detection for water distribution systems: A real-time diagnosis framework under data scarcity'* ([https://zenodo.org/records/18631450](https://zenodo.org/records/18631450)). Most leak and no-leak recordings were collected at an outdoor leakage-detection training base in Dongguan, China. The environmental-noise folder mixes on-site noise with public environmental clips.

Class counts in the copy used here:

| Class | Clips |
| --- | ---: |
| leak | {counts['all']['leak']} |
| no_leak | {counts['all']['no_leak']} |
| environmental_noise | {counts['all']['environmental_noise']} |
| **total** | **{counts['total']}** |

The split is stratified with seed 42: {counts['train']['total']} train, {counts['val']['total']} validation, {counts['test']['total']} test (70% / 15% / 15%). Early stopping watches validation accuracy. The test set is used once, after the checkpoint is chosen.

These sensors are not smartphone microphones. The live page still accepts a WAV as an experimental evaluation interface. Phone performance has not been independently validated. `collect_audio.py` records extra 1-second 16 kHz clips into `data/raw/<class>/` if household data are added later. Phone acoustics are a domain shift; they are not the training distribution.

The full archive is not committed. `download_dataset.py` fetches the three Zenodo RAR files.

## Architecture

Each clip is loaded as mono audio, resampled to 16 kHz, and padded or truncated to 1.0 s (16,000 samples). A Hann-window STFT (`n_fft=1024`, `hop_length=256`, `center=False`) produces 59 frames. A 64-band Slaney mel filterbank turns power spectra into a log-mel spectrogram, referenced to the clip maximum and clipped at −80 dB. The tensor shape is `1 × 64 × 59`.

`model.py` defines `SpectrogramCNN`: four convolution blocks (16-32-64-128 channels) with batch normalization, ReLU, max-pooling, adaptive average pooling, dropout 0.35, and a linear head to three logits. The network has {training['parameter_count']:,} parameters. `build_model(num_classes=3, pretrained=False)` reconstructs it. The checkpoint stores `state_dict`, `classes`, spectrogram settings, and the mel filterbank so Python and the browser share the same features.

## Training

Training ran on {training['gpu']} with batch size {training['batch_size']}. The loss is cross-entropy with inverse-frequency weights on the training split only: leak {weights['leak']:.3f}, no_leak {weights['no_leak']:.3f}, environmental_noise {weights['environmental_noise']:.3f}. Adam used learning rate {training['learning_rate']}. Spectrograms were augmented with time rolls, frequency masks, time masks, and Gaussian noise.

The run lasted {training['epochs_ran']} epochs with early stopping patience {training['patience']}. Best validation accuracy was {pct(training['best_val_accuracy'])} at epoch 15. Later validation scores moved on the 150-clip split because spectrogram augment is stochastic and the val set is small; the saved weights are the peak checkpoint, not the last epoch. That checkpoint was reloaded before the test evaluation.

## Results

On the held-out test set the model reaches {pct(test['accuracy'])} accuracy ({int(round(test['accuracy'] * counts['test']['total']))}/{counts['test']['total']}), {pct(test['macro_f1'])} macro F1, and {pct(test['weighted_f1'])} weighted F1. Macro F1 treats the small noise class equally with leak.

{confusion_sentence(metrics)}

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
{per_class_rows}

![Training and validation accuracy](../outputs/training_curves.png)

![Test confusion matrix](../outputs/confusion_matrix.png)

Confusion matrix (rows true, columns predicted), class order leak / no_leak / environmental_noise:

```
{json.dumps(matrix)}
```

## Web deployment and repository

The static site in `web/` states the problem, the dataset, the code layout, and these results. Sample buttons use field-sensor clips from the training distribution. Uploading a WAV is an experimental check: the browser computes the same log-mel spectrogram and runs the ONNX network with ONNX Runtime Web.

Canonical files for an automated check:

- `weights/model.pth` — PyTorch checkpoint in the repository
- `weights/model.onnx` — ONNX export in the repository; the live page copies it to `downloads/model.onnx` at build (`site_onnx` in `model_manifest.json`)
- `weights/config.json` — class order, STFT settings, and mel filterbank
- `weights/class_names.json` — class order alone
- `model_manifest.json` — paths and the prediction command
- `model.py` — `build_model()`
- `predict.py` — prints JSON probabilities for one clip
- `outputs/metrics.json` — the numbers in this report

Command-line prediction:

```bash
python predict.py --audio samples/leak.wav
```

Loading the checkpoint in Python:

```python
import torch
from model import build_model

checkpoint = torch.load("weights/model.pth", map_location="cpu", weights_only=False)
model = build_model(num_classes=len(checkpoint["classes"]), pretrained=False)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
```

Class order is leak, no_leak, environmental_noise. The network returns logits.

## Limitations

- Training audio comes from field acoustic sensors at a leakage-training site, not from a typical smartphone next to a kitchen tap.
- Environmental noise has only {counts['all']['environmental_noise']} clips, so its F1 is a noisier estimate than leak.
- Background speech, overlapping appliances, and other pipe materials are outside this distribution.
- The demo can be confidently wrong on a phone recording. It is not a safety instruction.

## Reproducibility

```bash
python -m venv .venv
.venv\\Scripts\\python -m pip install -r requirements.txt
.venv\\Scripts\\python download_dataset.py
.venv\\Scripts\\python train.py
.venv\\Scripts\\python build_docs.py
node prepare_site.js
```

CPU training also finishes, because the spectrograms are small. This machine used CUDA 12.8 on an RTX 5070. `train.py` falls back to CPU if a CUDA kernel cannot run.

## References

1. Acoustic data for *Self-supervised acoustic leakage detection for water distribution systems: A real-time diagnosis framework under data scarcity*. Zenodo, 2026. https://zenodo.org/records/18631450
2. {config['datasetCitation']}
3. McFee, Brian, et al. librosa. STFT and mel filterbank used for features.
4. ONNX Runtime Web 1.22 for in-browser inference.

{config['studentName']} · {config['rollNumber']} · {config['course']}
"""
    path = ROOT / "report" / "PROJECT_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def add_rect(slide, left, top, width, height, fill):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    return shape


def set_run(paragraph, text, size, color, bold=False, font=FONT):
    paragraph.clear()
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.name = font
    return run


def add_text_box(slide, left, top, width, height, text, size, color, bold=False, font=FONT, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    set_run(p, text, size, color, bold, font)
    return box


def add_bullets(slide, left, top, width, height, lines, size=18):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for index, line in enumerate(lines):
        p = tf.paragraphs[0] if index == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(8)
        set_run(p, line, size, INK)
    return box


def blank_slide(prs, title, notes):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_rect(slide, Inches(0), Inches(0), Inches(13.333), Inches(7.5), CREAM)
    add_rect(slide, Inches(0), Inches(0), Inches(13.333), Inches(0.12), GREEN)
    add_text_box(slide, Inches(0.7), Inches(0.35), Inches(12), Inches(0.7), title, 28, GREEN, True, SERIF)
    slide.notes_slide.notes_text_frame.text = notes
    return slide


def write_slides(config: dict, metrics: dict) -> None:
    from slide_deck import build_presentation

    build_presentation(config, metrics, ROOT / "presentation" / "presentation.pptx")
    return
    test = metrics["test"]
    training = metrics["training"]
    counts = metrics["counts"]
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    notes = {
        "Title": "Assalamualaikum. Mein Zaid Ahmed, roll SU92-BSCSM-F23-553. Deep Learning ka project smartphone audio se water-pipe leakage detection hai. Model teen classes predict karta hai: leak, no leak, aur environmental noise.",
        "Problem": "Pipe leak aksar late milta hai. Short acoustic clip se pehla screening decision possible hai, lekin yeh plumber ki jagah nahi leta.",
        "Question": "Kya ek second ki recording se leak, quiet pipe, aur background noise alag kiye ja sakte hain? Training public field clips par hai, demo phone WAV bhi accept karta hai.",
        "Dataset": f"Zenodo 18631450. {counts['total']} clips: leak {counts['all']['leak']}, no leak {counts['all']['no_leak']}, noise {counts['all']['environmental_noise']}. Split 700 / 150 / 150, seed 42.",
        "Features": "16 kilohertz, one second, Hann STFT 1024, hop 256, 64 mel bands. Shape 1 x 64 x 59. Same filterbank checkpoint aur browser dono use karte hain.",
        "Model": f"Four convolution blocks, 98,019 parameters. Dropout 0.35, three logits. build_model() model.py mein hai.",
        "Training": f"{training['gpu']} par Adam 1e-3, class weights, 23 epochs, patience 8. Best val accuracy {pct(training['best_val_accuracy'])}.",
        "Results": f"Test accuracy {pct(test['accuracy'])}, macro F1 {pct(test['macro_f1'])}. {confusion_sentence(metrics)}",
        "Demo": "weights/model.pth PyTorch checkpoint hai. ONNX browser chalata hai. predict.py ek clip par JSON deta hai.",
        "Limits": "Field sensors phone mic nahi. Noise class chhoti hai. Safety system nahi hai. Optional collect_audio.py se ghar ki clips baad mein add ho sakti hain.",
        "Close": f"Links: GitHub {display_url(config, 'githubUrl', 'GitHub URL')}. Site {display_url(config, 'vercelUrl', 'Vercel URL')}. YouTube {display_url(config, 'youtubeUrl', 'YouTube URL')}. Shukriya.",
    }

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_rect(slide, Inches(0), Inches(0), Inches(13.333), Inches(7.5), CREAM)
    add_rect(slide, Inches(0), Inches(0), Inches(0.18), Inches(7.5), GREEN)
    add_text_box(slide, Inches(0.9), Inches(1.6), Inches(11), Inches(0.4), "DEEP LEARNING SEMESTER PROJECT", 14, AMBER, True)
    add_text_box(slide, Inches(0.9), Inches(2.1), Inches(11.5), Inches(1.8), config["title"], 36, INK, True, SERIF)
    add_text_box(
        slide,
        Inches(0.9),
        Inches(4.3),
        Inches(11),
        Inches(1.2),
        f"{config['studentName']}  ·  {config['rollNumber']}\n{config['course']}",
        20,
        MUTED,
    )
    slide.notes_slide.notes_text_frame.text = notes["Title"]

    slide = blank_slide(prs, "Why acoustic leak screening", notes["Problem"])
    add_bullets(
        slide,
        Inches(0.8),
        Inches(1.3),
        Inches(11.5),
        Inches(5),
        [
            "A leak can run for a long time before anyone sees water damage.",
            "A one-second microphone clip is cheap to capture at a tap or pipe.",
            "The model is a first sort into leak, quiet pipe, or background noise.",
            "It is a course demo, not a utility alarm or a safety system.",
        ],
    )

    slide = blank_slide(prs, "Research question", notes["Question"])
    add_bullets(
        slide,
        Inches(0.8),
        Inches(1.3),
        Inches(11.5),
        Inches(5),
        [
            "Can a compact CNN tell leak audio from a quiet pipe and from noise?",
            "Training uses public field recordings from a leakage-detection site.",
            "The intended capture device for the demo is a smartphone WAV file.",
            "Phone microphones are a domain shift; the report states that limit.",
        ],
    )

    slide = blank_slide(prs, "Dataset", notes["Dataset"])
    add_bullets(
        slide,
        Inches(0.8),
        Inches(1.3),
        Inches(11.5),
        Inches(5),
        [
            f"Zenodo 18631450: {counts['total']} one-second clips.",
            f"leak {counts['all']['leak']}  ·  no_leak {counts['all']['no_leak']}  ·  environmental_noise {counts['all']['environmental_noise']}.",
            f"Stratified split seed 42: {counts['train']['total']} / {counts['val']['total']} / {counts['test']['total']}.",
            "download_dataset.py fetches the RAR files; audio is not stored in Git.",
        ],
    )

    slide = blank_slide(prs, "From waveform to spectrogram", notes["Features"])
    add_bullets(
        slide,
        Inches(0.8),
        Inches(1.3),
        Inches(11.5),
        Inches(5),
        [
            "16 kHz mono, 1.0 second, pad or crop to 16,000 samples.",
            "Hann STFT, n_fft 1024, hop 256, 59 frames, 64 mel bands.",
            "Log-mel referenced to the clip maximum, clipped at −80 dB.",
            "Python and the browser share the saved mel filterbank.",
        ],
    )

    slide = blank_slide(prs, "Spectrogram CNN", notes["Model"])
    add_bullets(
        slide,
        Inches(0.8),
        Inches(1.3),
        Inches(11.5),
        Inches(5),
        [
            "Four conv blocks: 16, 32, 64, 128 channels, batch norm, ReLU.",
            f"{training['parameter_count']:,} parameters and a 3-class linear head.",
            "Dropout 0.35. Output is logits, then softmax in predict.py / the page.",
            "Rebuild with build_model() from model.py, then load state_dict.",
        ],
    )

    slide = blank_slide(prs, "Training protocol", notes["Training"])
    add_bullets(
        slide,
        Inches(0.8),
        Inches(1.3),
        Inches(11.5),
        Inches(5),
        [
            f"Device: {training['gpu']}. Batch {training['batch_size']}. Adam lr {training['learning_rate']}.",
            "Class-weighted cross-entropy; noise class weight ≈ 2.92.",
            f"{training['epochs_ran']} epochs, patience {training['patience']}, best val {pct(training['best_val_accuracy'])}.",
            "Time roll, frequency mask, time mask, and Gaussian noise on spectrograms.",
        ],
    )

    slide = blank_slide(prs, "Held-out test results", notes["Results"])
    add_bullets(
        slide,
        Inches(0.8),
        Inches(1.3),
        Inches(11.5),
        Inches(5),
        [
            f"Accuracy {pct(test['accuracy'])} ({int(round(test['accuracy'] * counts['test']['total']))}/{counts['test']['total']}).",
            f"Macro F1 {pct(test['macro_f1'])}. Weighted F1 {pct(test['weighted_f1'])}.",
            f"leak F1 {pct(test['per_class']['leak']['f1'])}  ·  no_leak {pct(test['per_class']['no_leak']['f1'])}  ·  noise {pct(test['per_class']['environmental_noise']['f1'])}.",
            confusion_sentence(metrics),
        ],
    )

    slide = blank_slide(prs, "What an evaluator can download", notes["Demo"])
    add_bullets(
        slide,
        Inches(0.8),
        Inches(1.3),
        Inches(11.5),
        Inches(5),
        [
            "weights/model.pth — load state_dict with build_model() from model.py.",
            "weights/model.onnx in the repo; the Vercel page copies it to downloads/model.onnx.",
            "python predict.py --audio samples/leak.wav",
            "Report, slides, and this walkthrough are linked from the site.",
        ],
    )

    slide = blank_slide(prs, "Limits and optional phone clips", notes["Limits"])
    add_bullets(
        slide,
        Inches(0.8),
        Inches(1.3),
        Inches(11.5),
        Inches(5),
        [
            "Field hydrophones / training-base mics are not a kitchen smartphone.",
            "Only 114 noise clips, 17 in test, so that F1 is a small-sample estimate.",
            "collect_audio.py can add 1 s phone recordings without changing the labels.",
            "Do not use the demo as a leak-repair or safety decision.",
        ],
    )

    slide = blank_slide(prs, "Links", notes["Close"])
    add_bullets(
        slide,
        Inches(0.8),
        Inches(1.3),
        Inches(11.5),
        Inches(5),
        [
            f"GitHub: {display_url(config, 'githubUrl', 'GitHub URL')}",
            f"Site: {display_url(config, 'vercelUrl', 'Vercel URL')}",
            f"Walkthrough: {display_url(config, 'youtubeUrl', 'YouTube URL')}",
            f"LinkedIn: {display_url(config, 'linkedinUrl', 'LinkedIn URL')}",
            "Dataset: https://zenodo.org/records/18631450",
        ],
    )

    out = ROOT / "presentation" / "presentation.pptx"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        prs.save(out)
    except PermissionError:
        print("presentation.pptx is open, so the slides were not updated.")


def write_voiceover(config: dict, metrics: dict) -> None:
    test = metrics["test"]
    training = metrics["training"]
    counts = metrics["counts"]
    text = f"""# Voiceover script (Roman Urdu)

Har heading ek slide hai. Animation khud chalti hai. Slide ke beech mein click mat karna, warna agli slide chali jayegi. Slides khud Roman Urdu mein hain; yeh script thori lambi hai recording ke liye.

## Title
Assalamualaikum. Mein Zaid Ahmed hoon, roll number SU92-BSCSM-F23-553. Project smartphone audio se water-pipe leakage detection hai. Model ek second ki recording sun kar teen labels deta hai: leak, no leak, aur environmental noise.

## Masla
Pipe ka leak der se nazar aata hai. Phone se ek second ki clip lena sasta hai. Yeh pehla screening tool hai, plumber ya safety system nahi.

## Sawal
Sawal yeh hai ke compact CNN leak, quiet pipe, aur background noise alag kar sakta hai ya nahi. Training public field dataset par hui. Demo phone ki WAV bhi le leta hai, lekin phone mic training distribution nahi hai.

## Dataset
Dataset Zenodo record 18631450 hai. Kul {counts['total']} clips: leak {counts['all']['leak']}, no leak {counts['all']['no_leak']}, noise {counts['all']['environmental_noise']}. Stratified split seed 42: {counts['train']['total']} train, {counts['val']['total']} validation, {counts['test']['total']} test.

## Features
Har clip 16 kilohertz, one second. Hann STFT, n fft 1024, hop 256, 64 mel bands. Shape 1 by 64 by 59. Mel filterbank checkpoint ke saath save hai taake browser wahi features banaye.

## Model
Char convolution blocks hain, {training['parameter_count']:,} parameters. Dropout ke baad teen logits. Architecture model.py ke build_model se wapas banti hai.

## Training
Training {training['gpu']} par hui. Adam learning rate 1e-3, class-weighted loss, {training['epochs_ran']} epochs, patience {training['patience']}. Best validation accuracy {pct(training['best_val_accuracy'])}. Uske baad test ek dafa chala.

## Results
Test accuracy {pct(test['accuracy'])} hai, {int(round(test['accuracy'] * counts['test']['total']))} out of {counts['test']['total']}. Macro F1 {pct(test['macro_f1'])} hai. Sab se bari ghaltiyan: 3 leak ko no leak, 3 no leak ko leak, aur 1 no leak ko noise.

## Files
Evaluator weights/model.pth download kare. ONNX file browser ke liye hai. Command hai python predict.py --audio samples/leak.wav.

## Limits
Field sensors kitchen phone nahi. Noise class chhoti hai. collect_audio.py se baad mein ghar ki clips add ho sakti hain. Demo ko safety decision mat banao.

## Links
GitHub {display_url(config, 'githubUrl', 'GitHub URL')}. Site {display_url(config, 'vercelUrl', 'Vercel URL')}. Walkthrough {display_url(config, 'youtubeUrl', 'YouTube URL')}. LinkedIn {display_url(config, 'linkedinUrl', 'LinkedIn URL')}. Shukriya.
"""
    path = ROOT / "presentation" / "VOICEOVER_SCRIPT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_linkedin(config: dict, metrics: dict) -> None:
    test = metrics["test"]
    text = f"""# LinkedIn post

Copy the text below. Replace the bracketed URLs if they are still placeholders, then publish the post and paste its URL into `project.config.json` (`linkedinUrl`).

---

I built an acoustic water-pipe leak detector for my Deep Learning semester project.

A compact CNN classifies one-second clips as leak, no leak, or environmental noise. Training used 1,000 public field recordings (Zenodo 18631450). On a held-out test split the model reached {pct(test['accuracy'])} accuracy and {pct(test['macro_f1'])} macro F1. The page runs the same network in the browser from a WAV file.

- Live demo and report: {display_url(config, 'vercelUrl', 'Vercel URL')}
- Code and weights: {display_url(config, 'githubUrl', 'GitHub URL')}
- Walkthrough: {display_url(config, 'youtubeUrl', 'YouTube URL')}

Weights are at `weights/model.pth`. Training audio is from field sensors, so a phone clip is a domain shift — this is a course project, not a plumbing tool.

#DeepLearning #AudioClassification #ComputerVision
"""
    path = ROOT / "submission" / "LINKEDIN_POST.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_checklist(config: dict, metrics: dict) -> None:
    test = metrics["test"]
    text = f"""# Submission checklist

Measured test accuracy: {pct(test['accuracy'])}. Macro F1: {pct(test['macro_f1'])}.

- [x] Put your name and roll number in `project.config.json`.
- [x] Re-run `python build_docs.py` and `node prepare_site.js` so the report and slides use that name.
- [ ] Create a public GitHub repository and push this project. Do not enable Git LFS for the weights.
- [ ] Confirm these files open from the repository: `weights/model.pth`, `weights/model.onnx`, `weights/config.json`, `model.py`, `predict.py`, `model_manifest.json`.
- [ ] Put the repository URL in `project.config.json` as `githubUrl`.
- [ ] Deploy the site to Vercel from this folder (`npx vercel`). The build command is `node prepare_site.js`.
- [ ] Open the Vercel URL and check the demo, `report.html`, `downloads/model.pth`, and `downloads/presentation.pptx`.
- [ ] Put the Vercel URL in `project.config.json` as `vercelUrl`.
- [ ] Record the voiceover from `presentation/VOICEOVER_SCRIPT.md` while showing `presentation/presentation.pptx`.
- [ ] Upload the video to YouTube as Public or Unlisted. Store the URL as `youtubeUrl`.
- [ ] Publish `submission/LINKEDIN_POST.md` and store the post URL as `linkedinUrl`.
- [ ] Run `node prepare_site.js` again and redeploy so the live page contains the GitHub, YouTube, and LinkedIn links.
- [ ] Submit the instructor form with the Vercel URL, GitHub repository, YouTube video, and LinkedIn post.

Current placeholders:

- GitHub: {display_url(config, 'githubUrl', 'GitHub URL')}
- Vercel: {display_url(config, 'vercelUrl', 'Vercel URL')}
- YouTube: {display_url(config, 'youtubeUrl', 'YouTube URL')}
- LinkedIn: {display_url(config, 'linkedinUrl', 'LinkedIn URL')}
"""
    (ROOT / "submission" / "CHECKLIST.md").write_text(text, encoding="utf-8")


def write_readme(config: dict, metrics: dict) -> None:
    test = metrics["test"]
    counts = metrics["counts"]
    text = f"""# {config['title']}

Deep Learning semester project. A spectrogram CNN classifies one-second water-pipe audio as leak, no leak, or environmental noise, trained on the public [Zenodo 18631450](https://zenodo.org/records/18631450) acoustic set.

**Test accuracy:** {pct(test['accuracy'])}  
**Macro F1:** {pct(test['macro_f1'])}  
**Clips:** {counts['total']} ({counts['train']['total']} train / {counts['val']['total']} validation / {counts['test']['total']} test)

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

`model_manifest.json` lists repository paths (`weights/`) and site aliases (`downloads/`). The browser loads `downloads/model.onnx`, copied from `weights/model.onnx` at build.

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
.venv\\Scripts\\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\\Scripts\\python -m pip install -r requirements.txt
.venv\\Scripts\\python download_dataset.py
.venv\\Scripts\\python train.py
.venv\\Scripts\\python build_docs.py
node prepare_site.js
```

The CUDA 12.8 wheel is required for an RTX 50-series GPU. `train.py` uses the CPU if CUDA cannot run a kernel. The audio archive stays in `data/` and is gitignored.

## Site

`node prepare_site.js` copies the weights, plots, samples, report, and slides into `web/`. Deploy the repository on Vercel. The build command is already in `vercel.json`.

## Dataset citation

{config['datasetCitation']}

{config['studentName']} · {config['rollNumber']} · {config['course']}
"""
    (ROOT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    config = load_json(ROOT / "project.config.json")
    metrics = load_json(ROOT / "outputs" / "metrics.json")
    write_report(config, metrics)
    write_slides(config, metrics)
    write_voiceover(config, metrics)
    write_linkedin(config, metrics)
    write_checklist(config, metrics)
    write_readme(config, metrics)
    print("Wrote report, slides, voiceover, LinkedIn draft, checklist, and README.")


if __name__ == "__main__":
    main()
