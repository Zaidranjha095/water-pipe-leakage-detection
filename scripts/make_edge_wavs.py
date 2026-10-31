"""Write short edge-case WAVs for demo testing. Not part of the dataset."""
from pathlib import Path
import numpy as np
import soundfile as sf

out = Path("scripts/edge_wavs")
out.mkdir(parents=True, exist_ok=True)

silence = np.zeros(16000, dtype=np.float32)
sf.write(out / "silent_1s_16k.wav", silence, 16000)

short = 0.2 * np.sin(2 * np.pi * 440 * np.arange(4000) / 16000).astype(np.float32)
sf.write(out / "short_0p25s_16k.wav", short, 16000)

long = 0.2 * np.sin(2 * np.pi * 440 * np.arange(48000) / 16000).astype(np.float32)
sf.write(out / "long_3s_16k.wav", long, 16000)

stereo_l = 0.2 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
stereo_r = 0.1 * np.sin(2 * np.pi * 880 * np.arange(16000) / 16000).astype(np.float32)
sf.write(out / "stereo_1s_16k.wav", np.stack([stereo_l, stereo_r], axis=1), 16000)

odd = 0.2 * np.sin(2 * np.pi * 440 * np.arange(22050) / 22050).astype(np.float32)
sf.write(out / "mono_1s_22k.wav", odd, 22050)

(out / "corrupt.wav").write_bytes(b"RIFFXXXXWAVEfmt not a real wav")
(out / "empty.wav").write_bytes(b"")
print("wrote", list(p.name for p in out.iterdir()))
