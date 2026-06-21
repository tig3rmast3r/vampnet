from __future__ import annotations

import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
OUTPUT_DIR = BASE_DIR / "gradio-outputs"
DESTINATION_DIR = BASE_DIR / "gradio-export"


def timestamp_for(folder: Path) -> str:
    output = folder / "output.wav"
    source = output if output.is_file() else folder
    return datetime.fromtimestamp(source.stat().st_mtime).strftime("%Y%m%d%H%M%S")


def suffix_for(index: int) -> str:
    letters = "abcdefghijklmnopqrstuvwxyz"
    return letters[index % len(letters)]


def variant_label(path: Path) -> str:
    if path.name == "variant_1.wav":
        return "original"
    if path.stem.startswith("variant_"):
        try:
            variant = int(path.stem.split("_")[-1]) - 1
        except ValueError:
            return path.stem
        return f"variant{variant}"
    return path.stem


def export_folder(folder: Path, timestamp: str, suffix: str) -> int:
    copied = 0
    output = folder / "output.wav"
    if output.is_file():
        shutil.copy2(output, DESTINATION_DIR / f"{timestamp}_output_{suffix}.wav")
        copied += 1

    variants_dir = folder / "variants"
    if variants_dir.is_dir():
        for source in sorted(variants_dir.glob("variant_*.wav")):
            label = variant_label(source)
            shutil.copy2(source, DESTINATION_DIR / f"{timestamp}_{label}_{suffix}.wav")
            copied += 1
    return copied


def main() -> int:
    DESTINATION_DIR.mkdir(parents=True, exist_ok=True)
    if not OUTPUT_DIR.is_dir():
        print(f"missing {OUTPUT_DIR}")
        return 1

    folders_by_timestamp: dict[str, list[Path]] = defaultdict(list)
    for folder in sorted(OUTPUT_DIR.iterdir()):
        if not folder.is_dir() or folder.name in {"tmp", "saved"}:
            continue
        if not (folder / "output.wav").is_file() and not (folder / "variants").is_dir():
            continue
        folders_by_timestamp[timestamp_for(folder)].append(folder)

    copied = 0
    for timestamp, folders in sorted(folders_by_timestamp.items()):
        for index, folder in enumerate(sorted(folders)):
            copied += export_folder(folder, timestamp, suffix_for(index))

    print(f"completed: copied {copied} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
