# Voiceover script (Roman Urdu)

Har heading ek slide hai. Recording se pehle slides open rakho. Mid-slide click mat karna.

## Title
Assalamualaikum. Mein Zaid Ahmed hoon, roll number SU92-BSCSM-F23-553. Deep Learning ka semester project smartphone audio se water-pipe leakage detection hai. Model ek second ki recording sun kar teen labels deta hai: leak, no leak, aur environmental noise.

## Why acoustic leak screening
Pipe ka leak der se nazar aata hai. Microphone se short clip lena sasta hai. Yeh pehla screening tool hai, plumber ya safety system nahi.

## Research question
Sawal yeh hai ke compact CNN leak, quiet pipe, aur background noise alag kar sakta hai ya nahi. Training public field dataset par hui. Demo phone ki WAV bhi le leta hai, lekin phone mic training distribution nahi hai.

## Dataset
Dataset Zenodo record 18631450 hai. Kul 1000 clips: leak 500, no leak 386, noise 114. Stratified split seed 42: 700 train, 150 validation, 150 test.

## From waveform to spectrogram
Har clip 16 kilohertz, one second. Hann STFT, n fft 1024, hop 256, 64 mel bands. Shape 1 by 64 by 59. Mel filterbank checkpoint ke saath save hai taake browser wahi features banaye.

## Spectrogram CNN
Char convolution blocks hain, 98,019 parameters. Dropout ke baad teen logits. Architecture model.py ke build_model se wapas banti hai.

## Training protocol
Training NVIDIA GeForce RTX 5070 par hui. Adam learning rate 1e-3, class-weighted loss, 23 epochs, patience 8. Best validation accuracy 98.00%. Uske baad test ek dafa chala.

## Held-out test results
Test accuracy 94.00% hai, 141 out of 150. Macro F1 93.13% hai. The largest confusions were 3 leak predicted as no_leak; 3 no_leak predicted as leak; 1 no_leak predicted as environmental_noise.

## What an evaluator can download
Evaluator weights/model.pth download kare. ONNX file browser ke liye hai. Command hai python predict.py --audio samples/leak.wav.

## Limits and optional phone clips
Field sensors kitchen phone nahi. Noise class chhoti hai. collect_audio.py se baad mein ghar ki clips add ho sakti hain. Demo ko safety decision mat banao.

## Links
GitHub https://github.com/Zaidranjha095/water-pipe-leakage-detection. Site https://water-pipe-leakage-detection.vercel.app. Walkthrough [YouTube URL — add this before submission]. LinkedIn [LinkedIn URL — add this before submission]. Shukriya.
