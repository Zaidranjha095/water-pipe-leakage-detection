# Voiceover script (Roman Urdu)

Har heading ek slide hai. Animation khud chalti hai. Slide ke beech mein click mat karna, warna agli slide chali jayegi. Slides khud Roman Urdu mein hain; yeh script thori lambi hai recording ke liye.

## Title
Assalamualaikum. Mein Zaid Ahmed hoon, roll number SU92-BSCSM-F23-553. Project smartphone audio se water-pipe leakage detection hai. Model ek second ki recording sun kar teen labels deta hai: leak, no leak, aur environmental noise.

## Masla
Pipe ka leak der se nazar aata hai. Phone se ek second ki clip lena sasta hai. Yeh pehla screening tool hai, plumber ya safety system nahi.

## Sawal
Sawal yeh hai ke compact CNN leak, quiet pipe, aur background noise alag kar sakta hai ya nahi. Training public field dataset par hui. Demo phone ki WAV bhi le leta hai, lekin phone mic training distribution nahi hai.

## Dataset
Dataset Zenodo record 18631450 hai. Kul 1000 clips: leak 500, no leak 386, noise 114. Stratified split seed 42: 700 train, 150 validation, 150 test.

## Features
Har clip 16 kilohertz, one second. Hann STFT, n fft 1024, hop 256, 64 mel bands. Shape 1 by 64 by 59. Mel filterbank checkpoint ke saath save hai taake browser wahi features banaye.

## Model
Char convolution blocks hain, 98,019 parameters. Dropout ke baad teen logits. Architecture model.py ke build_model se wapas banti hai.

## Training
Training NVIDIA GeForce RTX 5070 par hui. Adam learning rate 1e-3, class-weighted loss, 23 epochs, patience 8. Best validation accuracy 98.00%. Uske baad test ek dafa chala.

## Results
Test accuracy 94.00% hai, 141 out of 150. Macro F1 93.13% hai. Sab se bari ghaltiyan: 3 leak ko no leak, 3 no leak ko leak, aur 1 no leak ko noise.

## Files
Evaluator weights/model.pth download kare. ONNX file browser ke liye hai. Command hai python predict.py --audio samples/leak.wav.

## Limits
Field sensors kitchen phone nahi. Noise class chhoti hai. collect_audio.py se baad mein ghar ki clips add ho sakti hain. Demo ko safety decision mat banao.

## Links
GitHub https://github.com/Zaidranjha095/water-pipe-leakage-detection. Site https://water-pipe-leakage-detection.vercel.app. Walkthrough https://youtu.be/wDVgfvbutMc. LinkedIn https://www.linkedin.com/posts/zaid-ahmad-0bb173252_deeplearning-audioclassification-computervision-share-7522126288854151168-DFHo. Shukriya.
