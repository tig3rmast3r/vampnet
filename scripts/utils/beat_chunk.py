#!/usr/bin/env python3
"""Extract a fixed-length chunk from an audio file starting at a downbeat-aligned
position.

Workflow:
    1. Load the WaveBeat beat tracker (checkpoint at ``models/wavebeat.pth``).
    2. Detect beats and downbeats for the input audio.
    3. Estimate the global BPM from the median inter-downbeat interval.
    4. Build the list of valid start positions: every downbeat (beat 1 of the
       measure) plus the beat that is two positions after each downbeat
       (beat 3 of the measure in 4/4 with quarter-note downbeats). This is
       what the user asked for: never start on beat 2 or beat 4, so the
       chunk does not begin on an "off-beat clap".
    5. Filter out positions where the chunk would extend past the end of
       the audio.
    6. Pick a random valid start (or fall back to 0.0s with a warning).
    7. Write the chunk to ``<input_stem>_beat_chunk.wav`` next to the input.
    8. Print the detected BPM and the chosen start to stdout.

Usage:
    python scripts/utils/beat_chunk.py path/to/audio.wav
    python scripts/utils/beat_chunk.py path/to/audio.wav --duration 8 --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Ensure the project package (``vampnet``) is importable when the script is
# run as a standalone file via ``python scripts/utils/beat_chunk.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import audiotools as at
import torch

from vampnet.beats import WaveBeat


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WAVEBEAT_CKPT = REPO_ROOT / "models" / "wavebeat.pth"
DEFAULT_CHUNK_SECONDS = 10.0
# Avoid the first and last minute of the track: intros and outros often
# have very different rhythm (drum fills, fades, ambient sections) that
# would skew the user's preview of the loop region. Only sample the
# "body" of the track.
AVOID_EDGES_SECONDS = 60.0
# Raw BPM estimates below this threshold are doubled. WaveBeat
# occasionally reports downbeats on half-measure boundaries in fast
# tempo tracks, halving the apparent BPM. In dance/EDM contexts tracks
# below 90 BPM are rare; doubling the raw estimate usually recovers the
# correct "tempo da ballo".
BPM_FLOOR_FOR_DOUBLING = 90.0
BPM_CEILING_FOR_HALVING = 220.0


def _normalize_bpm_to_dance_range(bpm: float) -> float:
    """Fold common half/double-time estimates into the dance BPM range."""
    bpm = float(bpm)
    if bpm <= 0:
        return 0.0
    if bpm < 45.0:
        bpm *= 4.0
    elif bpm < BPM_FLOOR_FOR_DOUBLING:
        bpm *= 2.0
    while bpm > BPM_CEILING_FOR_HALVING:
        bpm /= 2.0
    return bpm


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    total = float(np.sum(weights))
    if total <= 0:
        return float(np.median(values))
    cutoff = total * 0.5
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def _robust_weighted_bpm(raw_values: np.ndarray, normalized_values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    if normalized_values.size == 0:
        return 0.0, 0.0
    center = _weighted_median(normalized_values, weights)
    deviations = np.abs(normalized_values - center)
    mad = float(np.median(deviations))
    tolerance = max(0.35, mad * 3.0)
    keep = deviations <= tolerance
    if not np.any(keep):
        keep = np.ones_like(normalized_values, dtype=bool)
    kept_weights = weights[keep]
    return (
        float(np.average(raw_values[keep], weights=kept_weights)),
        float(np.average(normalized_values[keep], weights=kept_weights)),
    )


def _pairwise_bpm_estimates(
    times: np.ndarray,
    beats_per_step: float,
    lags: tuple[int, ...],
    weight_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    times = np.unique(times[np.isfinite(times)])
    if times.size < 2:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty
    raw_values = []
    normalized_values = []
    weights = []
    for lag in lags:
        if lag <= 0 or times.size <= lag:
            continue
        spans = times[lag:] - times[:-lag]
        valid = spans > 0
        if not np.any(valid):
            continue
        spans = spans[valid]
        raw = 60.0 * beats_per_step * lag / spans
        normalized = np.asarray([_normalize_bpm_to_dance_range(v) for v in raw], dtype=float)
        plausible = (normalized >= 60.0) & (normalized <= BPM_CEILING_FOR_HALVING)
        if not np.any(plausible):
            continue
        raw_values.append(raw[plausible])
        normalized_values.append(normalized[plausible])
        weights.append(spans[plausible] * weight_scale)
    if not raw_values:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty
    return (
        np.concatenate(raw_values),
        np.concatenate(normalized_values),
        np.concatenate(weights),
    )


def _estimate_bpm(beats: np.ndarray, downbeats: np.ndarray) -> tuple[float, float]:
    """Estimate BPM from long-range beat/downbeat pairs.

    Adjacent beat intervals are sensitive to WaveBeat onset jitter. This
    estimator compares beats/downbeats at progressively larger lags and
    weights longer time spans more heavily, so a few milliseconds of onset
    error have much less impact on the final BPM.

    Returns (raw_bpm, normalized_bpm) where normalized_bpm is the
    user-facing BPM after half/double-time folding.
    """
    raw_parts = []
    normalized_parts = []
    weight_parts = []
    if len(downbeats) >= 4:
        raw, normalized, weights = _pairwise_bpm_estimates(
            downbeats,
            beats_per_step=4.0,
            lags=(1, 2, 4, 8, 16, 32),
            weight_scale=1.5,
        )
        if normalized.size:
            raw_parts.append(raw)
            normalized_parts.append(normalized)
            weight_parts.append(weights)
    if len(beats) >= 4:
        raw, normalized, weights = _pairwise_bpm_estimates(
            beats,
            beats_per_step=1.0,
            lags=(4, 8, 16, 32, 64, 128),
            weight_scale=1.0,
        )
        if normalized.size:
            raw_parts.append(raw)
            normalized_parts.append(normalized)
            weight_parts.append(weights)
    if not normalized_parts:
        return 0.0, 0.0
    return _robust_weighted_bpm(
        np.concatenate(raw_parts),
        np.concatenate(normalized_parts),
        np.concatenate(weight_parts),
    )


def _candidate_starts(beats: np.ndarray, downbeats: np.ndarray) -> np.ndarray:
    """Return sorted unique candidate start times.

    Allowed positions are:
      * every downbeat (beat 1 of the measure)
      * the beat strictly two positions after a downbeat (beat 3 of the
        measure in 4/4 with quarter-note downbeats)

    Returns an empty array if no downbeats were detected.
    """
    if len(downbeats) == 0:
        return np.array([], dtype=float)
    starts: set[float] = set()
    beats_arr = np.asarray(beats, dtype=float)
    downbeats_arr = np.asarray(downbeats, dtype=float)
    for d in downbeats_arr:
        starts.add(float(d))
        idx = int(np.searchsorted(beats_arr, d, side="right"))
        if idx + 1 < len(beats_arr):
            starts.add(float(beats_arr[idx + 1]))
    return np.array(sorted(starts), dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Input audio file")
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_CHUNK_SECONDS,
        help=f"Chunk length in seconds (default: {DEFAULT_CHUNK_SECONDS:.0f})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: system entropy)",
    )
    parser.add_argument(
        "--wavebeat-ckpt",
        type=Path,
        default=DEFAULT_WAVEBEAT_CKPT,
        help=f"Path to wavebeat checkpoint (default: {DEFAULT_WAVEBEAT_CKPT})",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="Inference device for wavebeat (default: cuda). Use --no-cuda as a shortcut for cpu.",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_const",
        const="cpu",
        dest="device",
        help="Disable GPU and run wavebeat on CPU (shortcut for --device cpu).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output WAV path (default: <input_stem>_beat_chunk.wav)",
    )
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"error: input file not found: {args.audio}", file=sys.stderr)
        return 1
    if not args.wavebeat_ckpt.exists():
        print(f"error: wavebeat checkpoint not found: {args.wavebeat_ckpt}", file=sys.stderr)
        return 1
    if args.duration <= 0:
        print("error: --duration must be > 0", file=sys.stderr)
        return 1

    if args.seed is not None:
        random.seed(args.seed)

    # Default to CUDA; if the user asked for cuda but no GPU is available,
    # fall back to CPU with a warning so the script never fails on a
    # CPU-only box just because --device defaults to cuda.
    if args.device == "cuda" and not torch.cuda.is_available():
        print(
            "warning: --device cuda requested but no GPU is available; "
            "falling back to CPU.",
            file=sys.stderr,
        )
        args.device = "cpu"

    print(f"Loading WaveBeat from {args.wavebeat_ckpt} ({args.device})...")
    tracker = WaveBeat(ckpt_path=str(args.wavebeat_ckpt), device=args.device)
    print(f"Loading audio: {args.audio}")
    sig = at.AudioSignal(str(args.audio))
    sample_rate = int(sig.sample_rate)
    duration = float(sig.duration)

    print("Detecting beats and downbeats...")
    beats, downbeats = tracker.extract_beats(sig)
    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)

    raw_bpm, bpm = _estimate_bpm(beats, downbeats)
    if raw_bpm > 0 and bpm != raw_bpm:
        print(
            f"Detected BPM: {bpm:.2f} (raw {raw_bpm:.2f} doubled because it was below "
            f"{BPM_FLOOR_FOR_DOUBLING:.0f}; {len(beats)} beats, {len(downbeats)} downbeats, "
            f"duration {duration:.2f}s, sample rate {sample_rate} Hz)"
        )
    else:
        print(
            f"Detected BPM: {bpm:.2f} "
            f"({len(beats)} beats, {len(downbeats)} downbeats, "
            f"duration {duration:.2f}s, sample rate {sample_rate} Hz)"
        )

    # Build the candidate start positions (downbeats + beat 3 of each
    # measure) and then restrict to the "body" of the track: avoid the
    # first and last AVOID_EDGES_SECONDS so we never sample the intro or
    # outro (which usually have a different rhythm from the loop body).
    # If the track is shorter than 2 * AVOID_EDGES_SECONDS + chunk_len
    # we relax the constraint proportionally.
    candidates = _candidate_starts(beats, downbeats)
    body_start = AVOID_EDGES_SECONDS
    body_end = duration - AVOID_EDGES_SECONDS
    if body_end <= body_start + args.duration:
        # Track is too short for the 60s edge guard: scale it down so we
        # still leave half the chunk length of margin on each side.
        scale = max(0.0, (duration - args.duration) / 2.0)
        body_start = min(AVOID_EDGES_SECONDS, scale)
        body_end = max(duration - AVOID_EDGES_SECONDS, duration - scale)
    candidates = candidates[
        (candidates >= body_start) & (candidates + args.duration <= body_end)
    ]
    if candidates.size == 0:
        print(
            f"warning: no valid start position in the track body "
            f"[{body_start:.1f}s, {body_end:.1f}s] leaves room for a "
            f"{args.duration:.0f}s chunk. Falling back to time 0.0s.",
            file=sys.stderr,
        )
        start = 0.0
    else:
        start = float(random.choice(candidates))
    print(
        f"Selected start: {start:.3f}s "
        f"(beat 1 or beat 3 of a 4/4 measure, body window "
        f"[{body_start:.1f}s, {body_end:.1f}s])"
    )

    start_frame = int(round(start * sample_rate))
    end_frame = start_frame + int(round(args.duration * sample_rate))
    end_frame = min(end_frame, sig.audio_data.shape[-1])
    chunk_samples = sig.audio_data[..., start_frame:end_frame]
    # audiotools expects (channels, samples). Preserve the input layout.
    if chunk_samples.ndim == 1:
        chunk_samples = chunk_samples[np.newaxis, :]
    chunk_sig = at.AudioSignal(chunk_samples, sample_rate=sample_rate)
    out_path = args.output or args.audio.with_name(f"{args.audio.stem}_beat_chunk.wav")
    chunk_sig.write(str(out_path))
    print(
        f"Wrote chunk: {out_path} "
        f"({chunk_samples.shape[-1] / sample_rate:.2f}s, "
        f"{chunk_samples.shape[0]} channel(s))"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
