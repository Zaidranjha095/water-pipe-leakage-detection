# Optional smartphone recording guide

The submitted model is trained on the public Zenodo water-distribution acoustic set. Household phone clips are an extra path, not required for the current metrics.

If you add your own recordings, keep the same labels as the trained network:

- `leak` — dripping or escaping water at a tap or pipe joint
- `no_leak` — pipe/tap closed, no water sound
- `environmental_noise` — room noise, speech-free background, no leak

Record 1-second clips at 16 kHz. Hold the phone still, 10–30 cm from the pipe or tap. Do not speak. Keep people and private conversations out of the audio.

```powershell
cd "E:\Deep learning"
.\.venv\Scripts\python.exe collect_audio.py --label leak --count 40
.\.venv\Scripts\python.exe collect_audio.py --label no_leak --count 40
.\.venv\Scripts\python.exe collect_audio.py --label environmental_noise --count 40
```

Clips land in `data/raw/<class>/`. Retrain only after you have enough extra files, then re-export the site. Phone acoustics will not match the field sensors exactly, so treat any drop in accuracy as expected domain shift.
