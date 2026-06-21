#!/usr/bin/env python3
"""Pre-tokenize audio datasets with codec.pth for tokenized training mode.

Pipeline per file:
1. load full audio
2. to mono + resample to codec sample_rate
3. apply same train transform chain:
   - normalize to -24 LUFS (VolumeNorm equivalent)
   - ensure peak <= 1.0 (RescaleAudio equivalent)
4. encode with codec
5. save `.tokens.npz` in mirrored folder under output root

Also stores optional `valid_starts` token offsets inferred from random salient
probes using `loudness_cutoff` (default -35), so training can prefer louder
chunks when sampling from tokenized files.
"""

from __future__ import annotations

import argparse
import math
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from audiotools import AudioSignal
from lac.model.lac import LAC as DAC


AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
    ".mp4",
    ".aac",
    ".opus",
    ".aif",
    ".aiff",
    ".wma",
}


@dataclass
class Counters:
    scanned: int = 0
    processed: int = 0
    skipped_existing: int = 0
    failed: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize dataset audio files into codec code files."
    )
    parser.add_argument(
        "--input-roots",
        nargs="+",
        required=True,
        help="Input dataset roots (e.g., /data/DATASET /data/DATASET_no_kick).",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output root folder that will contain tokenized mirrors.",
    )
    parser.add_argument(
        "--codec-ckpt",
        default="models/vampnet/codec.pth",
        help="Path to codec checkpoint (.pth).",
    )
    parser.add_argument(
        "--n-codebooks",
        type=int,
        default=4,
        help="How many leading codec codebooks to save.",
    )
    parser.add_argument(
        "--loudness-cutoff",
        type=float,
        default=-35.0,
        help="Salient probing cutoff in LUFS used to build valid_starts.",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=10.0,
        help="Chunk duration (seconds) used to infer valid_starts.",
    )
    parser.add_argument(
        "--max-salient-attempts",
        type=int,
        default=32,
        help="Random probing attempts per file for valid_starts discovery.",
    )
    parser.add_argument(
        "--max-valid-starts",
        type=int,
        default=16,
        help="Maximum number of valid starts stored per file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base seed for random probing.",
    )
    parser.add_argument(
        "--encode-chunk-seconds",
        type=float,
        default=30.0,
        help="Chunk length (seconds) used for codec encoding of long files.",
    )
    parser.add_argument(
        "--encode-overlap-seconds",
        type=float,
        default=0.5,
        help="Symmetric overlap (seconds) between encoding chunks.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "File-level worker threads. On CUDA, codec.encode is serialized "
            "across workers to keep VRAM stable."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing token files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most this many audio files (0 = all).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for codec inference.",
    )
    parser.add_argument(
        "--report-dir",
        default="reports",
        help="Directory for summary and index reports.",
    )
    return parser.parse_args()


def collect_audio_files(input_roots: list[Path], limit: int = 0) -> list[tuple[Path, Path]]:
    files: list[tuple[Path, Path]] = []
    for root in input_roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            files.append((root, path))
            if limit > 0 and len(files) >= limit:
                return files
    return files


def build_output_path(output_root: Path, root: Path, path: Path) -> Path:
    rel = path.relative_to(root)
    # Preserve extension in name to avoid collisions between same stem
    # with different formats.
    out_name = f"{rel.name}.tokens.npz"
    return output_root / root.name / rel.parent / out_name


def probe_salient_starts(
    path: Path,
    duration_s: float,
    chunk_duration_s: float,
    token_rate: float,
    loudness_cutoff: float,
    max_attempts: int,
    max_valid: int,
    seed: int,
) -> np.ndarray:
    if duration_s <= chunk_duration_s:
        return np.array([0], dtype=np.int32)

    max_offset = max(duration_s - chunk_duration_s, 0.0)
    rng = random.Random(seed)
    starts: set[int] = set()

    for _ in range(max_attempts):
        offset = rng.uniform(0.0, max_offset)
        try:
            excerpt = AudioSignal(str(path), offset=offset, duration=chunk_duration_s)
            loudness = float(excerpt.loudness())
        except Exception:
            continue
        if loudness > loudness_cutoff:
            start = int(round(offset * token_rate))
            starts.add(max(0, start))
            if len(starts) >= max_valid:
                break

    if not starts:
        return np.empty((0,), dtype=np.int32)
    return np.array(sorted(starts), dtype=np.int32)


def process_file(
    path: Path,
    out_path: Path,
    codec: DAC,
    device: str,
    encode_lock: threading.Lock | None,
    n_codebooks: int,
    encode_chunk_seconds: float,
    encode_overlap_seconds: float,
    loudness_cutoff: float,
    chunk_duration: float,
    max_salient_attempts: int,
    max_valid_starts: int,
    seed: int,
) -> dict:
    signal = AudioSignal(str(path))
    signal = signal.to_mono()
    if int(signal.sample_rate) != int(codec.sample_rate):
        signal = signal.resample(int(codec.sample_rate))

    # Same train transform chain:
    # VolumeNorm(("const", -24)) + RescaleAudio()
    signal = signal.normalize(-24.0)
    signal = signal.ensure_max_of_audio(1.0)

    with torch.no_grad():
        codes = encode_codes_chunked(
            codec=codec,
            samples=signal.samples,
            sample_rate=int(signal.sample_rate),
            device=str(device),
            encode_lock=encode_lock,
            n_codebooks=int(n_codebooks),
            chunk_seconds=float(encode_chunk_seconds),
            overlap_seconds=float(encode_overlap_seconds),
        )

    token_rate = float(codec.sample_rate) / float(codec.hop_length)
    valid_starts = probe_salient_starts(
        path=path,
        duration_s=float(signal.duration),
        chunk_duration_s=float(chunk_duration),
        token_rate=token_rate,
        loudness_cutoff=float(loudness_cutoff),
        max_attempts=int(max_salient_attempts),
        max_valid=int(max_valid_starts),
        seed=int(seed),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        codes=codes,
        valid_starts=valid_starts,
        sample_rate=np.int32(codec.sample_rate),
        hop_length=np.int32(codec.hop_length),
        n_codebooks=np.int32(codes.shape[0]),
        n_frames=np.int32(codes.shape[1]),
        token_rate=np.float32(token_rate),
    )

    return {
        "token_path": str(out_path),
        "source_path": str(path),
        "n_codebooks": int(codes.shape[0]),
        "n_frames": int(codes.shape[1]),
        "valid_start_count": int(valid_starts.size),
    }


def _build_chunk_starts(total_samples: int, chunk_samples: int, overlap_samples: int) -> list[int]:
    if total_samples <= chunk_samples:
        return [0]

    stride = chunk_samples - (2 * overlap_samples)
    if stride <= 0:
        raise ValueError(
            "Invalid chunking parameters: stride <= 0. "
            "Increase --encode-chunk-seconds or decrease --encode-overlap-seconds."
        )

    starts = [0]
    while True:
        next_start = starts[-1] + stride
        if next_start + chunk_samples >= total_samples:
            last_start = max(total_samples - chunk_samples, 0)
            if last_start > starts[-1]:
                starts.append(last_start)
            break
        starts.append(next_start)
    return starts


def encode_codes_chunked(
    codec: DAC,
    samples: torch.Tensor,
    sample_rate: int,
    device: str,
    encode_lock: threading.Lock | None,
    n_codebooks: int,
    chunk_seconds: float,
    overlap_seconds: float,
) -> np.ndarray:
    total_samples = int(samples.shape[-1])
    if total_samples <= 0:
        raise ValueError("Audio has no samples after preprocessing.")

    hop_length = int(codec.hop_length)
    sr = int(sample_rate)

    chunk_samples = int(round(float(chunk_seconds) * sr))
    overlap_samples = int(round(float(overlap_seconds) * sr))
    if chunk_samples <= 0:
        raise ValueError("--encode-chunk-seconds must be > 0.")
    if overlap_samples < 0:
        raise ValueError("--encode-overlap-seconds must be >= 0.")

    starts = _build_chunk_starts(
        total_samples=total_samples,
        chunk_samples=chunk_samples,
        overlap_samples=overlap_samples,
    )

    overlap_tokens = int(math.ceil(float(overlap_samples) / float(hop_length))) if overlap_samples > 0 else 0
    code_chunks: list[np.ndarray] = []
    last_idx = len(starts) - 1

    for idx, start in enumerate(starts):
        end = min(start + chunk_samples, total_samples)
        chunk = samples[..., start:end]
        if chunk.numel() == 0:
            continue

        def _encode() -> np.ndarray:
            chunk_on_device = chunk.to(device)
            encoded = codec.encode(chunk_on_device, sr)["codes"]
            codes_np = (
                encoded[0, :n_codebooks, :]
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint16, copy=False)
            )
            return codes_np

        if encode_lock is None:
            chunk_codes = _encode()
        else:
            with encode_lock:
                chunk_codes = _encode()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        if len(starts) == 1 or overlap_tokens == 0:
            code_chunks.append(chunk_codes)
            continue

        left = overlap_tokens if idx > 0 else 0
        right = overlap_tokens if idx < last_idx else 0
        right_bound = chunk_codes.shape[1] - right if right > 0 else chunk_codes.shape[1]
        if left >= right_bound:
            raise ValueError(
                "Chunk trimming removed all tokens. "
                "Reduce --encode-overlap-seconds or increase --encode-chunk-seconds."
            )
        code_chunks.append(chunk_codes[:, left:right_bound])

    if not code_chunks:
        raise RuntimeError("Failed to produce any token chunks.")

    codes = np.concatenate(code_chunks, axis=1)
    expected_frames = int(math.ceil(float(total_samples) / float(hop_length)))
    if codes.shape[1] > expected_frames:
        codes = codes[:, :expected_frames]
    elif codes.shape[1] < expected_frames:
        pad = np.repeat(codes[:, -1:], expected_frames - codes.shape[1], axis=1)
        codes = np.concatenate([codes, pad], axis=1)
    return codes


def main() -> int:
    args = parse_args()
    started = time.time()

    input_roots = [Path(p).expanduser().resolve() for p in args.input_roots]
    output_root = Path(args.output_root).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    missing = [str(p) for p in input_roots if not p.exists()]
    if missing:
        print("Missing input roots:")
        for item in missing:
            print(f"- {item}")
        return 2

    codec = DAC.load(str(Path(args.codec_ckpt).expanduser().resolve()), map_location="cpu")
    codec = codec.eval().to(args.device)
    workers = max(int(args.workers), 1)
    if str(args.device).startswith("cuda") and workers > 1:
        print(
            "info: CUDA mode with workers>1: codec.encode is serialized across workers "
            "to keep VRAM usage stable."
        )
    encode_lock = (
        threading.Lock()
        if str(args.device).startswith("cuda") and workers > 1
        else None
    )

    files = collect_audio_files(input_roots, limit=int(args.limit))
    counters = Counters(scanned=len(files))

    ts = time.strftime("%Y%m%d_%H%M%S")
    index_path = report_dir / f"token_index_{ts}.jsonl"
    summary_path = report_dir / f"token_summary_{ts}.json"

    tasks: list[tuple[int, Path, Path]] = []
    for i, (root, path) in enumerate(files, start=1):
        out_path = build_output_path(output_root=output_root, root=root, path=path)
        if out_path.exists() and not args.overwrite:
            counters.skipped_existing += 1
            continue
        tasks.append((i, path, out_path))

    def _run_task(task: tuple[int, Path, Path]) -> tuple[int, Path, Path, dict | None, str | None]:
        i, path, out_path = task
        try:
            entry = process_file(
                path=path,
                out_path=out_path,
                codec=codec,
                device=str(args.device),
                encode_lock=encode_lock,
                n_codebooks=int(args.n_codebooks),
                encode_chunk_seconds=float(args.encode_chunk_seconds),
                encode_overlap_seconds=float(args.encode_overlap_seconds),
                loudness_cutoff=float(args.loudness_cutoff),
                chunk_duration=float(args.chunk_duration),
                max_salient_attempts=int(args.max_salient_attempts),
                max_valid_starts=int(args.max_valid_starts),
                seed=int(args.seed) + i,
            )
            return (i, path, out_path, entry, None)
        except Exception as exc:
            return (i, path, out_path, None, repr(exc))

    with index_path.open("w", encoding="utf-8") as index_file:
        done = 0
        if workers == 1:
            iterator = map(_run_task, tasks)
        else:
            executor = ThreadPoolExecutor(max_workers=workers)
            iterator = executor.map(_run_task, tasks)

        try:
            for i, path, out_path, entry, err in iterator:
                done += 1
                if err is None and entry is not None:
                    counters.processed += 1
                    index_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                else:
                    counters.failed += 1
                    index_file.write(
                        json.dumps(
                            {
                                "token_path": str(out_path),
                                "source_path": str(path),
                                "error": err,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                seen = done + counters.skipped_existing
                if seen % 100 == 0:
                    print(
                        f"progress {seen}/{len(files)} processed={counters.processed} "
                        f"skipped_existing={counters.skipped_existing} failed={counters.failed}"
                    )
        finally:
            if workers != 1:
                executor.shutdown(wait=True)

    elapsed = time.time() - started
    summary = {
        "input_roots": [str(p) for p in input_roots],
        "output_root": str(output_root),
        "codec_ckpt": str(Path(args.codec_ckpt).expanduser().resolve()),
        "n_codebooks": int(args.n_codebooks),
        "loudness_cutoff": float(args.loudness_cutoff),
        "chunk_duration": float(args.chunk_duration),
        "max_salient_attempts": int(args.max_salient_attempts),
        "max_valid_starts": int(args.max_valid_starts),
        "encode_chunk_seconds": float(args.encode_chunk_seconds),
        "encode_overlap_seconds": float(args.encode_overlap_seconds),
        "workers": int(workers),
        "device": str(args.device),
        "scanned": counters.scanned,
        "processed": counters.processed,
        "skipped_existing": counters.skipped_existing,
        "failed": counters.failed,
        "elapsed_sec": elapsed,
        "index_jsonl": str(index_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
