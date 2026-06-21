#!/usr/bin/env python3
"""Deduplicate audio datasets using strict duration and audio preview matching."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import concurrent.futures

import librosa
import numpy as np


DEFAULT_EXTENSIONS = ("mp3", "flac", "m4a")
AUX_EXTENSIONS = (".flac", ".wav", ".aif", ".aiff", ".ogg", ".m4a", ".mp3", ".opus", ".ac3")
DEFAULT_DURATION_ABS_SEC = 1.0
DEFAULT_DURATION_REL_PCT = 1.0
DEFAULT_WORKERS = max(4, (os.cpu_count() or 4))
DEFAULT_PREVIEW_SECONDS = 12.0
DEFAULT_PREVIEW_SR = 11025
DEFAULT_MAX_SHIFT_MS = 350.0
DEFAULT_MIN_OVERLAP_MS = 2000.0
DEFAULT_ENV_HOP_MS = 10.0
DEFAULT_ENV_SMOOTH_MS = 25.0

DEFAULT_CONFIDENT_ENV_CORR = 0.970
DEFAULT_CONFIDENT_WAVE_COS = 0.940
DEFAULT_CONFIDENT_WAVE_MAE = 0.360
DEFAULT_AMBIGUOUS_ENV_CORR = 0.940
DEFAULT_AMBIGUOUS_WAVE_COS = 0.900
DEFAULT_AMBIGUOUS_WAVE_MAE = 0.450

FORMAT_RANK = {"flac": 5, "wav": 4, "aiff": 3, "aif": 3, "ogg": 2, "mp3": 1}
LABEL_PRIORITY = {"gabry": 0, "gabry_laptop": 1, "gabry_toclean": 2}


@dataclass(frozen=True)
class Config:
    dataset_root: Path | None
    dataset_label: str
    dir_gabry: Path | None
    dir_gabry_laptop: Path | None
    dir_gabry_toclean: Path | None
    gabry_no_kick: Path | None
    gabry_stems: Path | None
    quarantine_dir: Path
    report_dir: Path
    plan_json: Path
    actions_dryrun_csv: Path
    actions_apply_csv: Path
    summary_dryrun_json: Path
    summary_apply_json: Path
    laptop_delete_csv: Path
    duplicates_delete_csv: Path
    ambiguous_review_csv: Path
    extensions: tuple[str, ...]
    duration_abs_sec: float
    duration_rel_pct: float
    apply_plan: Path | None
    apply_ambiguous: bool
    prompt_apply: bool
    workers: int
    ffprobe_bin: str
    preview_seconds: float
    preview_sr: int
    max_shift_ms: float
    min_overlap_ms: float
    env_hop_ms: float
    env_smooth_ms: float
    confident_env_corr: float
    confident_wave_cos: float
    confident_wave_mae: float
    ambiguous_env_corr: float
    ambiguous_wave_cos: float
    ambiguous_wave_mae: float
    allow_output_inside_input: bool


@dataclass(frozen=True)
class MediaEntry:
    path: Path
    label: str
    root_dir: Path
    relative: Path
    duration_sec: float
    ext: str


@dataclass(frozen=True)
class AudioPreview:
    waveform: np.ndarray
    envelope: np.ndarray
    hop_samples: int


@dataclass(frozen=True)
class PairMatch:
    left_index: int
    right_index: int
    duration_delta_sec: float
    env_corr: float
    wave_cos: float
    wave_mae: float


@dataclass
class PlannedMove:
    move_id: int
    cluster_id: int
    confidence: str
    source: Path
    source_label: str
    source_relative: Path
    destination: Path
    keep_target: Path
    keep_label: str
    reason: str
    action_kind: str
    duration_delta_sec: float | None = None
    env_corr: float | None = None
    wave_cos: float | None = None
    wave_mae: float | None = None
    status: str = "planned"
    final_destination: Path | None = None
    error: str | None = None


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        rank_left = self.rank[root_left]
        rank_right = self.rank[root_right]
        if rank_left < rank_right:
            self.parent[root_left] = root_right
        elif rank_left > rank_right:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1

    def components(self) -> dict[int, list[int]]:
        grouped: dict[int, list[int]] = defaultdict(list)
        for index in range(len(self.parent)):
            grouped[self.find(index)].append(index)
        return grouped


class AudioMatcher:
    def __init__(self, entries: list[MediaEntry], config: Config) -> None:
        self.entries = entries
        self.config = config
        self.cache: dict[int, AudioPreview | None] = {}
        self.pair_cache: dict[tuple[int, int], PairMatch | None] = {}
        self.preview_loads = 0

    def get_preview(self, index: int) -> AudioPreview | None:
        cached = self.cache.get(index)
        if cached is not None or index in self.cache:
            return cached

        entry = self.entries[index]
        try:
            audio, _ = librosa.load(
                str(entry.path),
                sr=self.config.preview_sr,
                mono=True,
                duration=self.config.preview_seconds,
            )
        except Exception as error:
            logging.warning("preview_load_failed %s (%s)", entry.path, error)
            self.cache[index] = None
            return None

        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.size < max(256, int(self.config.preview_sr * 0.75)):
            self.cache[index] = None
            return None

        peak = float(np.max(np.abs(waveform)))
        if peak <= 0:
            self.cache[index] = None
            return None
        waveform = waveform / peak

        # Light pre-emphasis reduces codec coloration impact before envelope extraction.
        pre = np.empty_like(waveform)
        pre[0] = waveform[0]
        pre[1:] = waveform[1:] - (0.97 * waveform[:-1])
        envelope = np.abs(pre)

        smooth = max(1, int(round(self.config.preview_sr * self.config.env_smooth_ms / 1000.0)))
        if smooth > 1:
            kernel = np.ones(smooth, dtype=np.float32) / float(smooth)
            envelope = np.convolve(envelope, kernel, mode="same")

        hop = max(1, int(round(self.config.preview_sr * self.config.env_hop_ms / 1000.0)))
        env_ds = envelope[::hop]
        env_std = float(np.std(env_ds))
        if env_std <= 1.0e-8:
            self.cache[index] = None
            return None
        env_norm = (env_ds - float(np.mean(env_ds))) / env_std

        self.preview_loads += 1
        preview = AudioPreview(waveform=waveform, envelope=env_norm.astype(np.float32), hop_samples=hop)
        self.cache[index] = preview
        return preview

    def compare(self, left_index: int, right_index: int) -> PairMatch | None:
        key = (left_index, right_index) if left_index < right_index else (right_index, left_index)
        cached = self.pair_cache.get(key)
        if cached is not None or key in self.pair_cache:
            return cached

        left = self.get_preview(key[0])
        right = self.get_preview(key[1])
        if left is None or right is None:
            self.pair_cache[key] = None
            return None

        max_lag_frames = max(0, int(round(self.config.max_shift_ms / self.config.env_hop_ms)))
        min_overlap_frames = max(1, int(round(self.config.min_overlap_ms / self.config.env_hop_ms)))
        env_corr, best_lag_frames = best_aligned_env_corr(
            left.envelope,
            right.envelope,
            max_lag_frames=max_lag_frames,
            min_overlap_frames=min_overlap_frames,
        )

        lag_samples = best_lag_frames * left.hop_samples
        wave_cos, wave_mae = aligned_wave_metrics(
            left.waveform,
            right.waveform,
            lag_samples=lag_samples,
            min_overlap_samples=max(1, int(round(self.config.preview_sr * self.config.min_overlap_ms / 1000.0))),
        )
        if wave_cos is None or wave_mae is None:
            self.pair_cache[key] = None
            return None

        match = PairMatch(
            left_index=key[0],
            right_index=key[1],
            duration_delta_sec=abs(self.entries[key[0]].duration_sec - self.entries[key[1]].duration_sec),
            env_corr=float(env_corr),
            wave_cos=float(wave_cos),
            wave_mae=float(wave_mae),
        )
        self.pair_cache[key] = match
        return match


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate one or more audio dataset roots using strict duration and audio preview matching."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Generic single dataset root to dedupe in-place; no no-kick/stems cleanup is planned.",
    )
    parser.add_argument(
        "--dataset-label",
        default="dataset",
        help="Report/quarantine label used with --dataset-root.",
    )
    parser.add_argument("--dir-gabry", type=Path, help="Main dataset root (highest priority). Can be used alone.")
    parser.add_argument("--dir-gabry-laptop", type=Path, help="Secondary dataset root (optional).")
    parser.add_argument("--dir-gabry-toclean", type=Path, help="Third dataset root (optional).")
    parser.add_argument("--gabry-no-kick", type=Path, default=None, help="Optional no-kick root for gabry.")
    parser.add_argument("--gabry-stems", type=Path, default=None, help="Optional stems root for gabry.")
    parser.add_argument("--quarantine-dir", type=Path, default=Path("./quarantine_duration_audio"))
    parser.add_argument("--report-dir", type=Path, default=Path("./reports_duration_audio"))
    parser.add_argument("--plan-json", type=Path, default=None, help="Path for generated reusable plan manifest.")
    parser.add_argument("--duration-abs-sec", type=float, default=DEFAULT_DURATION_ABS_SEC)
    parser.add_argument("--duration-rel-pct", type=float, default=DEFAULT_DURATION_REL_PCT)
    parser.add_argument("--extensions", default=",".join(DEFAULT_EXTENSIONS))
    parser.add_argument("--dry-run", action="store_true", help="Explicit dry-run mode (default when not applying a plan).")
    parser.add_argument("--apply-plan", type=Path, default=None, help="Apply an existing plan.json without recalculation.")
    parser.add_argument("--apply-ambiguous", action="store_true", help="Include ambiguous actions in apply mode.")
    parser.add_argument("--prompt-apply", action="store_true", help="After dry-run ask apply y/n using the same plan in memory.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel ffprobe workers.")
    parser.add_argument("--ffprobe-bin", default="ffprobe", help="ffprobe executable name/path.")

    parser.add_argument("--preview-seconds", type=float, default=DEFAULT_PREVIEW_SECONDS)
    parser.add_argument("--preview-sr", type=int, default=DEFAULT_PREVIEW_SR)
    parser.add_argument("--max-shift-ms", type=float, default=DEFAULT_MAX_SHIFT_MS)
    parser.add_argument("--min-overlap-ms", type=float, default=DEFAULT_MIN_OVERLAP_MS)
    parser.add_argument("--env-hop-ms", type=float, default=DEFAULT_ENV_HOP_MS)
    parser.add_argument("--env-smooth-ms", type=float, default=DEFAULT_ENV_SMOOTH_MS)

    parser.add_argument("--confident-env-corr", type=float, default=DEFAULT_CONFIDENT_ENV_CORR)
    parser.add_argument("--confident-wave-cos", type=float, default=DEFAULT_CONFIDENT_WAVE_COS)
    parser.add_argument("--confident-wave-mae", type=float, default=DEFAULT_CONFIDENT_WAVE_MAE)
    parser.add_argument("--ambiguous-env-corr", type=float, default=DEFAULT_AMBIGUOUS_ENV_CORR)
    parser.add_argument("--ambiguous-wave-cos", type=float, default=DEFAULT_AMBIGUOUS_WAVE_COS)
    parser.add_argument("--ambiguous-wave-mae", type=float, default=DEFAULT_AMBIGUOUS_WAVE_MAE)
    parser.add_argument(
        "--allow-output-inside-input",
        action="store_true",
        help="Allow report/quarantine paths inside an input root. Default blocks this to avoid rescanning quarantined audio.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def parse_extensions(raw: str) -> tuple[str, ...]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for token in raw.split(","):
        value = token.strip().lower().lstrip(".")
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    if not cleaned:
        raise ValueError("No valid extensions provided.")
    return tuple(cleaned)


def normalize_label(raw: str) -> str:
    label = raw.strip().lower().replace(" ", "_")
    cleaned = "".join(char for char in label if char.isalnum() or char in {"_", "-"})
    if not cleaned:
        raise ValueError("--dataset-label must contain at least one alphanumeric character.")
    return cleaned


def path_is_inside(path: Path, root: Path) -> bool:
    try:
        path.expanduser().resolve().relative_to(root.expanduser().resolve())
    except ValueError:
        return False
    return True


def duration_close(left: float, right: float, abs_sec: float, rel_pct: float) -> bool:
    delta = abs(left - right)
    if delta > abs_sec:
        return False
    if rel_pct <= 0:
        return True
    rel_threshold = min(left, right) * (rel_pct / 100.0)
    return delta <= rel_threshold


def classify_pair(match: PairMatch, config: Config) -> str | None:
    if (
        match.env_corr >= config.confident_env_corr
        and match.wave_cos >= config.confident_wave_cos
        and match.wave_mae <= config.confident_wave_mae
    ):
        return "confident"
    if (
        match.env_corr >= config.ambiguous_env_corr
        and match.wave_cos >= config.ambiguous_wave_cos
        and match.wave_mae <= config.ambiguous_wave_mae
    ):
        return "ambiguous"
    return None


def best_aligned_env_corr(
    left: np.ndarray,
    right: np.ndarray,
    *,
    max_lag_frames: int,
    min_overlap_frames: int,
) -> tuple[float, int]:
    best = -1.0
    best_lag = 0
    len_left = len(left)
    len_right = len(right)
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        if lag >= 0:
            left_seg = left[lag:]
            right_seg = right[: len(left_seg)]
        else:
            left_seg = left[: len_left + lag]
            right_seg = right[-lag:]
        overlap = min(len(left_seg), len(right_seg))
        if overlap < min_overlap_frames:
            continue
        left_seg = left_seg[:overlap]
        right_seg = right_seg[:overlap]

        left_center = left_seg - float(np.mean(left_seg))
        right_center = right_seg - float(np.mean(right_seg))
        denom = float(np.linalg.norm(left_center) * np.linalg.norm(right_center))
        if denom <= 1.0e-9:
            continue
        score = float(np.dot(left_center, right_center) / denom)
        if score > best:
            best = score
            best_lag = lag
    return best, best_lag


def aligned_wave_metrics(
    left: np.ndarray,
    right: np.ndarray,
    *,
    lag_samples: int,
    min_overlap_samples: int,
) -> tuple[float | None, float | None]:
    if lag_samples >= 0:
        left_seg = left[lag_samples:]
        right_seg = right[: len(left_seg)]
    else:
        left_seg = left[: len(left) + lag_samples]
        right_seg = right[-lag_samples:]

    overlap = min(len(left_seg), len(right_seg))
    if overlap < min_overlap_samples:
        return None, None
    left_seg = left_seg[:overlap]
    right_seg = right_seg[:overlap]

    left_center = left_seg - float(np.mean(left_seg))
    right_center = right_seg - float(np.mean(right_seg))
    left_norm = float(np.linalg.norm(left_center))
    right_norm = float(np.linalg.norm(right_center))
    if left_norm <= 1.0e-9 or right_norm <= 1.0e-9:
        return None, None

    left_unit = left_center / left_norm
    right_unit = right_center / right_norm
    cos = float(np.dot(left_unit, right_unit))
    mae = float(np.mean(np.abs(left_unit - right_unit)))
    return cos, mae


def scan_audio(root: Path, extensions: tuple[str, ...]) -> list[Path]:
    root = root.expanduser().resolve()
    allowed = {f".{extension}" for extension in extensions}
    return sorted(
        [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in allowed and not path.name.startswith("._")
        ]
    )


def probe_duration(path: Path, ffprobe_bin: str) -> float | None:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    output = result.stdout.strip()
    if not output:
        return None
    try:
        duration = float(output)
    except ValueError:
        return None
    if duration <= 0:
        return None
    return duration


def collect_entries(
    root: Path,
    label: str,
    extensions: tuple[str, ...],
    workers: int,
    ffprobe_bin: str,
) -> list[MediaEntry]:
    files = scan_audio(root, extensions)
    if not files:
        return []

    logging.info("[%s] scanning durations with ffprobe: files=%d", label, len(files))
    entries: list[MediaEntry] = []
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(probe_duration, path, ffprobe_bin): path for path in files}
        for future in concurrent.futures.as_completed(future_map):
            path = future_map[future]
            try:
                duration = future.result()
            except Exception:
                duration = None
            done += 1
            if done == len(files) or done % max(1, len(files) // 10) == 0:
                logging.info("[%s] ffprobe progress %d/%d", label, done, len(files))
            if duration is None:
                logging.warning("[%s] skip_unreadable_duration %s", label, path)
                continue
            relative = path.resolve().relative_to(root)
            entries.append(
                MediaEntry(
                    path=path.resolve(),
                    label=label,
                    root_dir=root,
                    relative=relative,
                    duration_sec=duration,
                    ext=path.suffix.lower().lstrip("."),
                )
            )
    entries.sort(key=lambda item: str(item.path))
    return entries


def build_pair_matches(entries: list[MediaEntry], config: Config) -> tuple[list[PairMatch], list[PairMatch]]:
    matcher = AudioMatcher(entries, config)
    confident: list[PairMatch] = []
    ambiguous: list[PairMatch] = []

    ordered = sorted(range(len(entries)), key=lambda index: entries[index].duration_sec)
    processed = 0
    candidate_count = 0
    for left_pos, left_index in enumerate(ordered):
        left_duration = entries[left_index].duration_sec
        for right_pos in range(left_pos + 1, len(ordered)):
            right_index = ordered[right_pos]
            right_duration = entries[right_index].duration_sec
            if right_duration - left_duration > config.duration_abs_sec:
                break
            if not duration_close(left_duration, right_duration, config.duration_abs_sec, config.duration_rel_pct):
                continue
            candidate_count += 1
            match = matcher.compare(left_index, right_index)
            if match is None:
                continue
            level = classify_pair(match, config)
            if level == "confident":
                confident.append(match)
            elif level == "ambiguous":
                ambiguous.append(match)
        processed += 1
        if processed == len(ordered) or processed % max(1, len(ordered) // 10) == 0:
            logging.info(
                "pair_scan %d/%d candidates=%d confident=%d ambiguous=%d previews_loaded=%d",
                processed,
                len(ordered),
                candidate_count,
                len(confident),
                len(ambiguous),
                matcher.preview_loads,
            )

    confident.sort(key=lambda item: (item.left_index, item.right_index))
    ambiguous.sort(key=lambda item: (item.left_index, item.right_index))
    return confident, ambiguous


def build_components(size: int, edges: Iterable[tuple[int, int]]) -> list[list[int]]:
    dsu = DisjointSet(size)
    for left, right in edges:
        dsu.union(left, right)
    components = [members for members in dsu.components().values() if len(members) > 1]
    return sorted((sorted(component) for component in components), key=lambda members: members[0])


def select_keep_entry(member_indexes: list[int], entries: list[MediaEntry]) -> int:
    min_priority = min(LABEL_PRIORITY.get(entries[index].label, 999) for index in member_indexes)
    preferred = [index for index in member_indexes if LABEL_PRIORITY.get(entries[index].label, 999) == min_priority]
    preferred.sort(
        key=lambda index: (
            -FORMAT_RANK.get(entries[index].ext, 0),
            str(entries[index].path),
        )
    )
    return preferred[0]


def unique_target(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    index = 1
    while True:
        candidate = path.with_name(f"{stem}__{index}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def build_quarantine_destination(config: Config, source_label: str, source_relative: Path) -> Path:
    return config.quarantine_dir / source_label / source_relative


def find_related_nokick_paths(entry: MediaEntry, config: Config) -> list[Path]:
    if config.gabry_no_kick is None:
        return []
    base = (config.gabry_no_kick / entry.relative).with_suffix("")
    matches: list[Path] = []
    for extension in AUX_EXTENSIONS:
        candidate = base.with_suffix(extension)
        if candidate.exists() and candidate.is_file():
            matches.append(candidate.resolve())
    return sorted(set(matches))


def find_related_stem_paths(entry: MediaEntry, config: Config) -> list[Path]:
    if config.gabry_stems is None:
        return []
    stem_dir = (config.gabry_stems / entry.relative.parent).resolve()
    if not stem_dir.exists() or not stem_dir.is_dir():
        return []
    prefix = f"{entry.relative.stem}_"
    matches = [
        path.resolve()
        for path in stem_dir.iterdir()
        if path.is_file()
        and path.name.startswith(prefix)
        and path.suffix.lower() in AUX_EXTENSIONS
    ]
    return sorted(matches)


def add_move(
    planned_by_source: dict[Path, PlannedMove],
    move: PlannedMove,
) -> None:
    existing = planned_by_source.get(move.source)
    if existing is not None:
        return
    planned_by_source[move.source] = move


def build_pair_lookup(confident_pairs: list[PairMatch], ambiguous_pairs: list[PairMatch]) -> dict[tuple[int, int], PairMatch]:
    lookup: dict[tuple[int, int], PairMatch] = {}
    for pair in confident_pairs + ambiguous_pairs:
        key = (pair.left_index, pair.right_index) if pair.left_index < pair.right_index else (pair.right_index, pair.left_index)
        lookup[key] = pair
    return lookup


def build_moves_for_clusters(
    clusters: list[list[int]],
    entries: list[MediaEntry],
    pair_lookup: dict[tuple[int, int], PairMatch],
    config: Config,
    confidence: str,
    start_cluster_id: int,
    start_move_id: int,
) -> tuple[list[PlannedMove], int, int, dict[int, int]]:
    planned_by_source: dict[Path, PlannedMove] = {}
    cluster_id = start_cluster_id
    move_id = start_move_id
    keep_map: dict[int, int] = {}

    for members in clusters:
        keep_index = select_keep_entry(members, entries)
        keep_map[cluster_id] = keep_index
        keep_entry = entries[keep_index]
        for member_index in members:
            if member_index == keep_index:
                continue
            source_entry = entries[member_index]
            key = (member_index, keep_index) if member_index < keep_index else (keep_index, member_index)
            metrics = pair_lookup.get(key)
            duration_delta_sec = metrics.duration_delta_sec if metrics is not None else abs(source_entry.duration_sec - keep_entry.duration_sec)
            env_corr = metrics.env_corr if metrics is not None else None
            wave_cos = metrics.wave_cos if metrics is not None else None
            wave_mae = metrics.wave_mae if metrics is not None else None

            move_id += 1
            add_move(
                planned_by_source,
                PlannedMove(
                    move_id=move_id,
                    cluster_id=cluster_id,
                    confidence=confidence,
                    source=source_entry.path,
                    source_label=source_entry.label,
                    source_relative=source_entry.relative,
                    destination=build_quarantine_destination(config, source_entry.label, source_entry.relative),
                    keep_target=keep_entry.path,
                    keep_label=keep_entry.label,
                    reason=f"{confidence} duplicate (duration+audio)",
                    action_kind="main_audio",
                    duration_delta_sec=duration_delta_sec,
                    env_corr=env_corr,
                    wave_cos=wave_cos,
                    wave_mae=wave_mae,
                ),
            )

            if confidence == "confident" and source_entry.label == "gabry":
                for related_path in find_related_nokick_paths(source_entry, config):
                    move_id += 1
                    if config.gabry_no_kick is None:
                        continue
                    related_rel = related_path.relative_to(config.gabry_no_kick)
                    add_move(
                        planned_by_source,
                        PlannedMove(
                            move_id=move_id,
                            cluster_id=cluster_id,
                            confidence=confidence,
                            source=related_path,
                            source_label="gabry_no_kick",
                            source_relative=related_rel,
                            destination=build_quarantine_destination(config, "gabry_no_kick", related_rel),
                            keep_target=keep_entry.path,
                            keep_label=keep_entry.label,
                            reason="cleanup no_kick for removed gabry duplicate",
                            action_kind="aux_no_kick",
                            duration_delta_sec=duration_delta_sec,
                            env_corr=env_corr,
                            wave_cos=wave_cos,
                            wave_mae=wave_mae,
                        ),
                    )
                for related_path in find_related_stem_paths(source_entry, config):
                    move_id += 1
                    if config.gabry_stems is None:
                        continue
                    related_rel = related_path.relative_to(config.gabry_stems)
                    add_move(
                        planned_by_source,
                        PlannedMove(
                            move_id=move_id,
                            cluster_id=cluster_id,
                            confidence=confidence,
                            source=related_path,
                            source_label="gabry_stems",
                            source_relative=related_rel,
                            destination=build_quarantine_destination(config, "gabry_stems", related_rel),
                            keep_target=keep_entry.path,
                            keep_label=keep_entry.label,
                            reason="cleanup stems for removed gabry duplicate",
                            action_kind="aux_stem",
                            duration_delta_sec=duration_delta_sec,
                            env_corr=env_corr,
                            wave_cos=wave_cos,
                            wave_mae=wave_mae,
                        ),
                    )
        cluster_id += 1

    moves = sorted(planned_by_source.values(), key=lambda item: str(item.source))
    return moves, cluster_id, move_id, keep_map


def split_ambiguous_clusters(
    entries: list[MediaEntry],
    confident_clusters: list[list[int]],
    ambiguous_pairs: list[PairMatch],
) -> list[list[int]]:
    removed_by_confident: set[int] = set()
    for cluster in confident_clusters:
        keep = select_keep_entry(cluster, entries)
        for member in cluster:
            if member != keep:
                removed_by_confident.add(member)

    candidates = [
        (pair.left_index, pair.right_index)
        for pair in ambiguous_pairs
        if pair.left_index not in removed_by_confident and pair.right_index not in removed_by_confident
    ]
    return build_components(len(entries), candidates)


def move_to_dict(move: PlannedMove) -> dict[str, object]:
    return {
        "move_id": move.move_id,
        "cluster_id": move.cluster_id,
        "confidence": move.confidence,
        "source": str(move.source),
        "source_label": move.source_label,
        "source_relative": str(move.source_relative),
        "destination": str(move.destination),
        "keep_target": str(move.keep_target),
        "keep_label": move.keep_label,
        "reason": move.reason,
        "action_kind": move.action_kind,
        "duration_delta_sec": move.duration_delta_sec,
        "env_corr": move.env_corr,
        "wave_cos": move.wave_cos,
        "wave_mae": move.wave_mae,
        "status": move.status,
        "final_destination": str(move.final_destination) if move.final_destination is not None else None,
        "error": move.error,
    }


def move_from_dict(payload: dict[str, object]) -> PlannedMove:
    final_destination = payload.get("final_destination")
    return PlannedMove(
        move_id=int(payload["move_id"]),
        cluster_id=int(payload["cluster_id"]),
        confidence=str(payload["confidence"]),
        source=Path(str(payload["source"])),
        source_label=str(payload["source_label"]),
        source_relative=Path(str(payload["source_relative"])),
        destination=Path(str(payload["destination"])),
        keep_target=Path(str(payload["keep_target"])),
        keep_label=str(payload["keep_label"]),
        reason=str(payload["reason"]),
        action_kind=str(payload["action_kind"]),
        duration_delta_sec=float(payload["duration_delta_sec"]) if payload.get("duration_delta_sec") is not None else None,
        env_corr=float(payload["env_corr"]) if payload.get("env_corr") is not None else None,
        wave_cos=float(payload["wave_cos"]) if payload.get("wave_cos") is not None else None,
        wave_mae=float(payload["wave_mae"]) if payload.get("wave_mae") is not None else None,
        status=str(payload.get("status", "planned")),
        final_destination=Path(str(final_destination)) if final_destination else None,
        error=str(payload["error"]) if payload.get("error") else None,
    )


def execute_moves(moves: list[PlannedMove], apply: bool) -> None:
    for move in moves:
        if not move.source.exists():
            move.status = "skipped_missing"
            move.error = "source_not_found"
            continue
        if not apply:
            move.status = "planned"
            move.final_destination = move.destination
            continue

        destination = unique_target(move.destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(move.source), str(destination))
        except OSError as error:
            move.status = "failed"
            move.error = str(error)
            continue

        move.status = "moved"
        move.final_destination = destination


def ensure_report_paths(config: Config) -> None:
    config.report_dir.mkdir(parents=True, exist_ok=True)
    config.quarantine_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        config.plan_json,
        config.actions_dryrun_csv,
        config.actions_apply_csv,
        config.summary_dryrun_json,
        config.summary_apply_json,
        config.laptop_delete_csv,
        config.duplicates_delete_csv,
        config.ambiguous_review_csv,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)


def write_actions_csv(path: Path, moves: list[PlannedMove]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "move_id",
                "cluster_id",
                "confidence",
                "source",
                "source_label",
                "source_relative",
                "destination",
                "keep_target",
                "keep_label",
                "reason",
                "action_kind",
                "duration_delta_sec",
                "env_corr",
                "wave_cos",
                "wave_mae",
                "status",
                "final_destination",
                "error",
            ],
        )
        writer.writeheader()
        for move in moves:
            writer.writerow(move_to_dict(move))


def write_laptop_delete_csv(path: Path, moves: list[PlannedMove]) -> None:
    rows = [
        move
        for move in moves
        if move.action_kind == "main_audio" and move.source_label == "gabry_laptop"
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source",
                "keep_target",
                "reason",
                "duration_delta_sec",
                "env_corr",
                "wave_cos",
                "wave_mae",
                "confidence",
                "status",
                "destination",
                "final_destination",
                "error",
            ],
        )
        writer.writeheader()
        for move in rows:
            writer.writerow(
                {
                    "source": str(move.source),
                    "keep_target": str(move.keep_target),
                    "reason": move.reason,
                    "duration_delta_sec": move.duration_delta_sec,
                    "env_corr": move.env_corr,
                    "wave_cos": move.wave_cos,
                    "wave_mae": move.wave_mae,
                    "confidence": move.confidence,
                    "status": move.status,
                    "destination": str(move.destination),
                    "final_destination": str(move.final_destination) if move.final_destination is not None else "",
                    "error": move.error or "",
                }
            )


def write_duplicates_delete_csv(path: Path, moves: list[PlannedMove]) -> None:
    rows = [move for move in moves if move.action_kind == "main_audio"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source",
                "source_label",
                "keep_target",
                "keep_label",
                "reason",
                "duration_delta_sec",
                "env_corr",
                "wave_cos",
                "wave_mae",
                "confidence",
                "status",
                "destination",
                "final_destination",
                "error",
            ],
        )
        writer.writeheader()
        for move in rows:
            writer.writerow(
                {
                    "source": str(move.source),
                    "source_label": move.source_label,
                    "keep_target": str(move.keep_target),
                    "keep_label": move.keep_label,
                    "reason": move.reason,
                    "duration_delta_sec": move.duration_delta_sec,
                    "env_corr": move.env_corr,
                    "wave_cos": move.wave_cos,
                    "wave_mae": move.wave_mae,
                    "confidence": move.confidence,
                    "status": move.status,
                    "destination": str(move.destination),
                    "final_destination": str(move.final_destination) if move.final_destination is not None else "",
                    "error": move.error or "",
                }
            )


def write_ambiguous_review_csv(path: Path, clusters: list[list[int]], entries: list[MediaEntry], pair_lookup: dict[tuple[int, int], PairMatch]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cluster_id",
                "cluster_size",
                "keep_target",
                "keep_label",
                "candidate_remove",
                "candidate_label",
                "duration_delta_sec_to_keep",
                "env_corr_to_keep",
                "wave_cos_to_keep",
                "wave_mae_to_keep",
            ],
        )
        writer.writeheader()
        for cluster_id, members in enumerate(clusters, 1):
            keep_index = select_keep_entry(members, entries)
            keep_entry = entries[keep_index]
            for member_index in members:
                if member_index == keep_index:
                    continue
                source_entry = entries[member_index]
                key = (member_index, keep_index) if member_index < keep_index else (keep_index, member_index)
                metrics = pair_lookup.get(key)
                writer.writerow(
                    {
                        "cluster_id": cluster_id,
                        "cluster_size": len(members),
                        "keep_target": str(keep_entry.path),
                        "keep_label": keep_entry.label,
                        "candidate_remove": str(source_entry.path),
                        "candidate_label": source_entry.label,
                        "duration_delta_sec_to_keep": metrics.duration_delta_sec if metrics else abs(source_entry.duration_sec - keep_entry.duration_sec),
                        "env_corr_to_keep": metrics.env_corr if metrics else None,
                        "wave_cos_to_keep": metrics.wave_cos if metrics else None,
                        "wave_mae_to_keep": metrics.wave_mae if metrics else None,
                    }
                )


def summarize_moves(moves: list[PlannedMove]) -> dict[str, int]:
    by_label: dict[str, int] = {}
    by_destination: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for move in moves:
        by_label[move.source_label] = by_label.get(move.source_label, 0) + 1
        by_destination[move.action_kind] = by_destination.get(move.action_kind, 0) + 1
        by_status[move.status] = by_status.get(move.status, 0) + 1

    summary: dict[str, int] = {
        "planned_moves": len(moves),
    }
    for key, value in by_label.items():
        summary[f"source_label:{key}"] = value
    for key, value in by_destination.items():
        summary[f"action_kind:{key}"] = value
    for key, value in by_status.items():
        summary[f"status:{key}"] = value
    return summary


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + os.linesep, encoding="utf-8")


def build_config(args: argparse.Namespace) -> Config:
    report_dir = args.report_dir.expanduser().resolve()
    quarantine_dir = args.quarantine_dir.expanduser().resolve()
    plan_json = args.plan_json.expanduser().resolve() if args.plan_json else report_dir / "plan.json"
    return Config(
        dataset_root=args.dataset_root.expanduser().resolve() if args.dataset_root else None,
        dataset_label=normalize_label(args.dataset_label),
        dir_gabry=args.dir_gabry.expanduser().resolve() if args.dir_gabry else None,
        dir_gabry_laptop=args.dir_gabry_laptop.expanduser().resolve() if args.dir_gabry_laptop else None,
        dir_gabry_toclean=args.dir_gabry_toclean.expanduser().resolve() if args.dir_gabry_toclean else None,
        gabry_no_kick=args.gabry_no_kick.expanduser().resolve() if args.gabry_no_kick else None,
        gabry_stems=args.gabry_stems.expanduser().resolve() if args.gabry_stems else None,
        quarantine_dir=quarantine_dir,
        report_dir=report_dir,
        plan_json=plan_json,
        actions_dryrun_csv=report_dir / "actions_dryrun.csv",
        actions_apply_csv=report_dir / "actions_apply.csv",
        summary_dryrun_json=report_dir / "summary_dryrun.json",
        summary_apply_json=report_dir / "summary_apply.json",
        laptop_delete_csv=report_dir / "gabry_laptop_to_delete.csv",
        duplicates_delete_csv=report_dir / "duplicates_to_delete.csv",
        ambiguous_review_csv=report_dir / "ambiguous_review.csv",
        extensions=parse_extensions(args.extensions),
        duration_abs_sec=args.duration_abs_sec,
        duration_rel_pct=args.duration_rel_pct,
        apply_plan=args.apply_plan.expanduser().resolve() if args.apply_plan else None,
        apply_ambiguous=args.apply_ambiguous,
        prompt_apply=args.prompt_apply,
        workers=args.workers,
        ffprobe_bin=args.ffprobe_bin,
        preview_seconds=args.preview_seconds,
        preview_sr=args.preview_sr,
        max_shift_ms=args.max_shift_ms,
        min_overlap_ms=args.min_overlap_ms,
        env_hop_ms=args.env_hop_ms,
        env_smooth_ms=args.env_smooth_ms,
        confident_env_corr=args.confident_env_corr,
        confident_wave_cos=args.confident_wave_cos,
        confident_wave_mae=args.confident_wave_mae,
        ambiguous_env_corr=args.ambiguous_env_corr,
        ambiguous_wave_cos=args.ambiguous_wave_cos,
        ambiguous_wave_mae=args.ambiguous_wave_mae,
        allow_output_inside_input=args.allow_output_inside_input,
    )


def validate_config(config: Config) -> None:
    if config.duration_abs_sec < 0:
        raise ValueError("--duration-abs-sec must be >= 0.")
    if config.duration_rel_pct < 0:
        raise ValueError("--duration-rel-pct must be >= 0.")
    if config.workers < 1:
        raise ValueError("--workers must be >= 1.")
    if config.apply_plan is not None and config.prompt_apply:
        raise ValueError("--prompt-apply cannot be combined with --apply-plan.")

    for name in (
        "confident_env_corr",
        "confident_wave_cos",
        "ambiguous_env_corr",
        "ambiguous_wave_cos",
    ):
        value = getattr(config, name)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be within 0..1.")
    for name in ("confident_wave_mae", "ambiguous_wave_mae"):
        value = getattr(config, name)
        if value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0.")

    if config.confident_env_corr < config.ambiguous_env_corr:
        raise ValueError("--confident-env-corr must be >= --ambiguous-env-corr.")
    if config.confident_wave_cos < config.ambiguous_wave_cos:
        raise ValueError("--confident-wave-cos must be >= --ambiguous-wave-cos.")
    if config.confident_wave_mae > config.ambiguous_wave_mae:
        raise ValueError("--confident-wave-mae must be <= --ambiguous-wave-mae.")

    if config.preview_seconds <= 0:
        raise ValueError("--preview-seconds must be > 0.")
    if config.preview_sr < 2000:
        raise ValueError("--preview-sr must be >= 2000.")
    if config.max_shift_ms < 0:
        raise ValueError("--max-shift-ms must be >= 0.")
    if config.min_overlap_ms <= 0:
        raise ValueError("--min-overlap-ms must be > 0.")
    if config.env_hop_ms <= 0:
        raise ValueError("--env-hop-ms must be > 0.")
    if config.env_smooth_ms <= 0:
        raise ValueError("--env-smooth-ms must be > 0.")

    if config.apply_plan is None:
        legacy_dirs = {
            "dir_gabry": config.dir_gabry,
            "dir_gabry_laptop": config.dir_gabry_laptop,
            "dir_gabry_toclean": config.dir_gabry_toclean,
        }
        if config.dataset_root is not None:
            if any(path is not None for path in legacy_dirs.values()):
                raise ValueError("--dataset-root cannot be combined with --dir-gabry/--dir-gabry-laptop/--dir-gabry-toclean.")
            if config.gabry_no_kick is not None or config.gabry_stems is not None:
                raise ValueError("--dataset-root cannot be combined with --gabry-no-kick or --gabry-stems.")
            if not config.dataset_root.exists() or not config.dataset_root.is_dir():
                raise FileNotFoundError(f"dataset root not found: {config.dataset_root}")

        provided_dirs = {
            "dataset_root": config.dataset_root,
            "dir_gabry": config.dir_gabry,
            "dir_gabry_laptop": config.dir_gabry_laptop,
            "dir_gabry_toclean": config.dir_gabry_toclean,
        }
        if all(path is None for path in provided_dirs.values()):
            raise ValueError(
                "At least one input directory is required (use --dataset-root, or --dir-gabry and optionally --dir-gabry-laptop/--dir-gabry-toclean)."
            )
        for name, path in provided_dirs.items():
            if path is None:
                continue
            if not path.exists() or not path.is_dir():
                raise FileNotFoundError(f"{name} directory not found: {path}")

        if not config.allow_output_inside_input:
            for name, path in provided_dirs.items():
                if path is None:
                    continue
                for output_name, output_path in (
                    ("report-dir", config.report_dir),
                    ("quarantine-dir", config.quarantine_dir),
                ):
                    if path_is_inside(output_path, path):
                        raise ValueError(
                            f"--{output_name} must be outside {name} ({path}) to avoid rescanning outputs. "
                            "Use --allow-output-inside-input only if you intentionally want this."
                        )
    else:
        if not config.apply_plan.exists() or not config.apply_plan.is_file():
            raise FileNotFoundError(f"Plan file not found: {config.apply_plan}")

    if config.gabry_no_kick is not None and (not config.gabry_no_kick.exists() or not config.gabry_no_kick.is_dir()):
        raise FileNotFoundError(f"gabry no_kick directory not found: {config.gabry_no_kick}")
    if config.gabry_stems is not None and (not config.gabry_stems.exists() or not config.gabry_stems.is_dir()):
        raise FileNotFoundError(f"gabry stems directory not found: {config.gabry_stems}")


def build_manifest_payload(
    config: Config,
    entries: list[MediaEntry],
    confident_pairs: list[PairMatch],
    ambiguous_pairs: list[PairMatch],
    confident_clusters: list[list[int]],
    ambiguous_clusters: list[list[int]],
    confident_moves: list[PlannedMove],
    ambiguous_moves: list[PlannedMove],
) -> dict[str, object]:
    counts_by_label: dict[str, int] = defaultdict(int)
    for entry in entries:
        counts_by_label[entry.label] += 1
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "dir_gabry": str(config.dir_gabry) if config.dir_gabry else None,
            "dir_gabry_laptop": str(config.dir_gabry_laptop) if config.dir_gabry_laptop else None,
            "dir_gabry_toclean": str(config.dir_gabry_toclean) if config.dir_gabry_toclean else None,
            "dataset_root": str(config.dataset_root) if config.dataset_root else None,
            "dataset_label": config.dataset_label,
            "gabry_no_kick": str(config.gabry_no_kick) if config.gabry_no_kick else None,
            "gabry_stems": str(config.gabry_stems) if config.gabry_stems else None,
            "quarantine_dir": str(config.quarantine_dir),
            "extensions": list(config.extensions),
        },
        "thresholds": {
            "duration_abs_sec": config.duration_abs_sec,
            "duration_rel_pct": config.duration_rel_pct,
            "preview_seconds": config.preview_seconds,
            "preview_sr": config.preview_sr,
            "max_shift_ms": config.max_shift_ms,
            "min_overlap_ms": config.min_overlap_ms,
            "env_hop_ms": config.env_hop_ms,
            "env_smooth_ms": config.env_smooth_ms,
            "confident_env_corr": config.confident_env_corr,
            "confident_wave_cos": config.confident_wave_cos,
            "confident_wave_mae": config.confident_wave_mae,
            "ambiguous_env_corr": config.ambiguous_env_corr,
            "ambiguous_wave_cos": config.ambiguous_wave_cos,
            "ambiguous_wave_mae": config.ambiguous_wave_mae,
        },
        "summary": {
            "entries_total": len(entries),
            "entries_by_label": dict(counts_by_label),
            "pair_matches_confident": len(confident_pairs),
            "pair_matches_ambiguous": len(ambiguous_pairs),
            "clusters_confident": len(confident_clusters),
            "clusters_ambiguous": len(ambiguous_clusters),
            "planned_moves_confident": len(confident_moves),
            "planned_moves_ambiguous": len(ambiguous_moves),
        },
        "actions_confident": [move_to_dict(move) for move in confident_moves],
        "actions_ambiguous": [move_to_dict(move) for move in ambiguous_moves],
    }


def run_dry(config: Config) -> int:
    if (
        config.dataset_root is None
        and config.dir_gabry is None
        and config.dir_gabry_laptop is None
        and config.dir_gabry_toclean is None
    ):
        raise ValueError("Missing input directories for dry-run mode.")

    entries: list[MediaEntry] = []
    if config.dataset_root is not None:
        entries.extend(collect_entries(config.dataset_root, config.dataset_label, config.extensions, config.workers, config.ffprobe_bin))
    if config.dir_gabry is not None:
        entries.extend(collect_entries(config.dir_gabry, "gabry", config.extensions, config.workers, config.ffprobe_bin))
    if config.dir_gabry_laptop is not None:
        entries.extend(
            collect_entries(
                config.dir_gabry_laptop,
                "gabry_laptop",
                config.extensions,
                config.workers,
                config.ffprobe_bin,
            )
        )
    if config.dir_gabry_toclean is not None:
        entries.extend(
            collect_entries(
                config.dir_gabry_toclean,
                "gabry_toclean",
                config.extensions,
                config.workers,
                config.ffprobe_bin,
            )
        )
    if not entries:
        raise RuntimeError("No input files found after scanning.")

    logging.info("Total entries with readable duration: %d", len(entries))
    confident_pairs, ambiguous_pairs = build_pair_matches(entries, config)
    confident_clusters = build_components(
        len(entries),
        ((pair.left_index, pair.right_index) for pair in confident_pairs),
    )
    ambiguous_clusters = split_ambiguous_clusters(entries, confident_clusters, ambiguous_pairs)
    pair_lookup = build_pair_lookup(confident_pairs, ambiguous_pairs)

    confident_moves, next_cluster_id, next_move_id, _ = build_moves_for_clusters(
        confident_clusters,
        entries,
        pair_lookup,
        config,
        confidence="confident",
        start_cluster_id=1,
        start_move_id=0,
    )
    ambiguous_moves, _, _, _ = build_moves_for_clusters(
        ambiguous_clusters,
        entries,
        pair_lookup,
        config,
        confidence="ambiguous",
        start_cluster_id=next_cluster_id,
        start_move_id=next_move_id,
    )

    execute_moves(confident_moves, apply=False)
    write_actions_csv(config.actions_dryrun_csv, confident_moves)
    write_laptop_delete_csv(config.laptop_delete_csv, confident_moves)
    write_duplicates_delete_csv(config.duplicates_delete_csv, confident_moves)
    write_ambiguous_review_csv(config.ambiguous_review_csv, ambiguous_clusters, entries, pair_lookup)

    manifest = build_manifest_payload(
        config=config,
        entries=entries,
        confident_pairs=confident_pairs,
        ambiguous_pairs=ambiguous_pairs,
        confident_clusters=confident_clusters,
        ambiguous_clusters=ambiguous_clusters,
        confident_moves=confident_moves,
        ambiguous_moves=ambiguous_moves,
    )
    write_json(config.plan_json, manifest)
    dry_summary = {
        "mode": "dry-run",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan_json": str(config.plan_json),
        "actions_dryrun_csv": str(config.actions_dryrun_csv),
        "laptop_delete_csv": str(config.laptop_delete_csv),
        "duplicates_delete_csv": str(config.duplicates_delete_csv),
        "ambiguous_review_csv": str(config.ambiguous_review_csv),
        "counts": summarize_moves(confident_moves),
        "manifest_summary": manifest["summary"],
    }
    write_json(config.summary_dryrun_json, dry_summary)
    logging.info(
        "Dry-run done: confident_moves=%d ambiguous_moves=%d plan=%s",
        len(confident_moves),
        len(ambiguous_moves),
        config.plan_json,
    )

    if config.prompt_apply:
        prompt = "Apply this plan now? [y/N]: "
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            answer = ""
        if answer in {"y", "yes"}:
            selected = list(confident_moves)
            if config.apply_ambiguous:
                selected.extend(ambiguous_moves)
            execute_moves(selected, apply=True)
            write_actions_csv(config.actions_apply_csv, selected)
            write_laptop_delete_csv(config.laptop_delete_csv, selected)
            write_duplicates_delete_csv(config.duplicates_delete_csv, selected)
            apply_summary = {
                "mode": "apply",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "from_plan": str(config.plan_json),
                "apply_ambiguous": config.apply_ambiguous,
                "counts": summarize_moves(selected),
            }
            write_json(config.summary_apply_json, apply_summary)
            logging.info("Apply from prompt done: moves=%d", len(selected))
        else:
            logging.info("Prompt apply declined.")
    return 0


def run_apply_from_plan(config: Config) -> int:
    if config.apply_plan is None:
        raise ValueError("Missing --apply-plan.")
    payload = json.loads(config.apply_plan.read_text(encoding="utf-8"))
    confident_moves = [move_from_dict(item) for item in payload.get("actions_confident", [])]
    ambiguous_moves = [move_from_dict(item) for item in payload.get("actions_ambiguous", [])]
    selected = list(confident_moves)
    if config.apply_ambiguous:
        selected.extend(ambiguous_moves)

    execute_moves(selected, apply=True)
    write_actions_csv(config.actions_apply_csv, selected)
    write_laptop_delete_csv(config.laptop_delete_csv, selected)
    write_duplicates_delete_csv(config.duplicates_delete_csv, selected)
    apply_summary = {
        "mode": "apply",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "from_plan": str(config.apply_plan),
        "apply_ambiguous": config.apply_ambiguous,
        "selected_moves": len(selected),
        "counts": summarize_moves(selected),
    }
    write_json(config.summary_apply_json, apply_summary)
    logging.info("Apply done from plan: selected_moves=%d", len(selected))
    return 0


def main() -> int:
    configure_logging()
    args = parse_args()
    config = build_config(args)
    validate_config(config)
    ensure_report_paths(config)

    if config.apply_plan is not None:
        return run_apply_from_plan(config)
    return run_dry(config)


if __name__ == "__main__":
    raise SystemExit(main())
