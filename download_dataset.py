"""Download the public water-distribution leak audio set from Zenodo."""
from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARCHIVE_DIR = DATA_DIR / "archives"

RECORD_URL = "https://zenodo.org/records/18631450"
FILES = {
    "leak": {
        "url": "https://zenodo.org/records/18631450/files/leak%20acoustic%20data.rar?download=1",
        "name": "leak_acoustic_data.rar",
    },
    "no_leak": {
        "url": "https://zenodo.org/records/18631450/files/no%20leak%20acoustic%20data.rar?download=1",
        "name": "no_leak_acoustic_data.rar",
    },
    "environmental_noise": {
        "url": "https://zenodo.org/records/18631450/files/environmental%20noise.rar?download=1",
        "name": "environmental_noise.rar",
    },
}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}


def _progress(block: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    done = min(100, int(block * block_size * 100 / total))
    print(f"\r  {done:3d}%", end="", flush=True)


def extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(destination)
        return
    seven = [
        shutil.which("7z"),
        shutil.which("7za"),
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ]
    for binary in seven:
        if binary and Path(binary).exists():
            subprocess.check_call([binary, "x", "-y", f"-o{destination}", str(archive)])
            return
    unrar = shutil.which("unrar")
    if unrar:
        subprocess.check_call([unrar, "x", "-o+", str(archive), str(destination) + "\\"])
        return
    import rarfile

    rarfile.RarFile(archive).extractall(destination)


def iter_audio(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
            yield path


def flatten_class(extracted: Path, label: str) -> int:
    target = RAW_DIR / label
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for index, source in enumerate(sorted(iter_audio(extracted))):
        destination = target / f"{label}_{index:04d}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        count += 1
    if count == 0:
        raise SystemExit(f"No audio files found after extracting {label}. See {extracted}")
    return count


def main() -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading acoustic clips from {RECORD_URL}")
    for label, spec in FILES.items():
        archive = ARCHIVE_DIR / spec["name"]
        if not archive.exists() or archive.stat().st_size < 1000:
            print(f"Fetching {label}...")
            urlretrieve(spec["url"], archive, _progress)
            print()
        extracted = DATA_DIR / "extracted" / label
        if extracted.exists():
            shutil.rmtree(extracted)
        extract_archive(archive, extracted)
        n = flatten_class(extracted, label)
        print(f"{label}: {n} clips")
    print(f"Ready under {RAW_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
