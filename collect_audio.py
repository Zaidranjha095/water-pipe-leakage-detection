"""Optional: record 1-second household clips with the default microphone."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

from model import CLASSES, SAMPLE_RATE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, choices=CLASSES)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    target = args.out / args.label
    target.mkdir(parents=True, exist_ok=True)
    print(f"Recording {args.count} clips for {args.label}. Press Enter for each clip; Ctrl+C to stop.")
    for index in range(args.count):
        input(f"[{index + 1}/{args.count}] Press Enter, then make the condition...")
        audio = sd.rec(int(args.seconds * args.sample_rate), samplerate=args.sample_rate, channels=1, dtype="float32")
        sd.wait()
        filename = target / f"phone_{args.label}_{int(time.time())}_{index:03d}.wav"
        write(filename, args.sample_rate, np.squeeze(audio))
        print(f"saved {filename}")


if __name__ == "__main__":
    main()
