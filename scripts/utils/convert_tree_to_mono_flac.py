#!/usr/bin/env python3
"""Convert audio trees to mono FLAC in-place.

Behavior:
- Downmix stereo with explicit L+R averaging:
  pan=mono|c0=0.5*c0+0.5*c1
- For non-stereo multichannel, ffmpeg default downmix is used via -ac 1.
- Output format is always FLAC mono.
- Existing mono FLAC files are skipped.
- Source files are removed after successful conversion when output path differs
  (unless --keep-source is provided).

Dry-run is the default; pass --apply to execute writes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


AUDIO_EXTENSIONS = (
    "mp3",
    "flac",
    "wav",
    "ogg",
    "m4a",
    "aac",
    "opus",
    "aif",
    "aiff",
    "mp4",
    "wma",
)

# Supports normal names like "track.mp3" and collision-resolved names from
# previous normalization runs such as "track.mp3__e9ce3903".
NAME_RE = re.compile(
    rf"^(?P<stem>.+?)\.(?P<ext>{'|'.join(AUDIO_EXTENSIONS)})(?:__(?P<hash>[0-9a-f]{{8}}))?$",
    re.IGNORECASE,
)


@dataclass
class Job:
    idx: int
    src: Path
    target: Path
    ext: str


@dataclass
class JobResult:
    idx: int
    src: str
    target: str
    status: str
    info: str


@dataclass
class Stats:
    scanned_files: int = 0
    matched_audio: int = 0
    converted: int = 0
    skipped_mono_flac: int = 0
    skipped_conflict: int = 0
    failed_probe: int = 0
    failed_convert: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert audio trees to mono FLAC in-place (dry-run by default)."
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="Dataset roots to process recursively.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply conversion and file replacement. Default is dry-run.",
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Keep original source file when output path differs.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=5,
        help="FLAC compression level [0..12].",
    )
    parser.add_argument(
        "--on-conflict",
        choices=("skip", "suffix"),
        default="suffix",
        help=(
            "When target .flac already exists and differs from source path: "
            "'skip' leaves source untouched, 'suffix' writes <name>__convN.flac."
        ),
    )
    parser.add_argument(
        "--report-csv",
        default=f"reports/mono_flac_conversion_{int(time.time())}.csv",
        help="CSV report path.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N matched audio files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of matched audio files to process (0 = unlimited).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel worker threads for ffprobe/ffmpeg.",
    )
    return parser.parse_args()


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def parse_audio_name(name: str) -> tuple[str, str, str | None] | None:
    match = NAME_RE.match(name)
    if not match:
        return None
    stem = match.group("stem")
    ext = match.group("ext").lower()
    hash_suffix = match.group("hash")
    return stem, ext, hash_suffix


def build_target_name(stem: str, hash_suffix: str | None) -> str:
    if hash_suffix:
        return f"{stem}__{hash_suffix}.flac"
    return f"{stem}.flac"


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def discover_jobs(roots: list[Path], limit: int = 0) -> tuple[list[Job], int]:
    jobs: list[Job] = []
    scanned = 0
    for root in roots:
        for path in iter_files(root):
            scanned += 1
            parsed = parse_audio_name(path.name)
            if parsed is None:
                continue
            stem, ext, hash_suffix = parsed
            target = path.with_name(build_target_name(stem, hash_suffix))
            jobs.append(Job(idx=len(jobs), src=path, target=target, ext=ext))
            if limit > 0 and len(jobs) >= limit:
                return jobs, scanned
    return jobs, scanned


def probe_channels(path: Path) -> int | None:
    cp = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    if cp.returncode != 0:
        return None
    value = cp.stdout.strip()
    if not value.isdigit():
        return None
    return int(value)


def resolve_conflict(target: Path, reserved_targets: set[Path]) -> Path:
    if not target.exists() and target not in reserved_targets:
        return target
    idx = 1
    while True:
        candidate = target.with_name(f"{target.stem}__conv{idx}{target.suffix}")
        if not candidate.exists() and candidate not in reserved_targets:
            return candidate
        idx += 1


def convert_job(
    job: Job,
    args: argparse.Namespace,
    reserved_targets: set[Path],
    reserved_targets_lock: threading.Lock,
) -> JobResult:
    try:
        channels = probe_channels(job.src)
        if channels is None:
            return JobResult(job.idx, str(job.src), str(job.target), "probe_failed", "")

        # Already in desired format and channel count.
        if job.src == job.target and job.ext == "flac" and channels == 1:
            return JobResult(
                job.idx,
                str(job.src),
                str(job.target),
                "skip_mono_flac",
                str(channels),
            )

        final_target = job.target
        if final_target != job.src:
            with reserved_targets_lock:
                if final_target.exists() or final_target in reserved_targets:
                    if args.on_conflict == "skip":
                        return JobResult(
                            job.idx,
                            str(job.src),
                            str(final_target),
                            "skip_conflict",
                            str(channels),
                        )
                    final_target = resolve_conflict(final_target, reserved_targets)
                reserved_targets.add(final_target)

        if not args.apply:
            return JobResult(
                job.idx,
                str(job.src),
                str(final_target),
                "dryrun_convert",
                str(channels),
            )

        tmp_path = final_target.with_name(
            f"{final_target.stem}.tmp.{os.getpid()}.{job.idx}.flac"
        )
        ffmpeg_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(job.src),
            "-vn",
            "-sn",
            "-dn",
            "-map_metadata",
            "0",
        ]
        if channels == 2:
            ffmpeg_cmd += ["-af", "pan=mono|c0=0.5*c0+0.5*c1"]
        else:
            ffmpeg_cmd += ["-ac", "1"]
        ffmpeg_cmd += [
            "-f",
            "flac",
            "-c:a",
            "flac",
            "-compression_level",
            str(args.compression_level),
            str(tmp_path),
        ]

        cp = run(ffmpeg_cmd)
        if cp.returncode != 0 or not tmp_path.exists():
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            return JobResult(
                job.idx,
                str(job.src),
                str(final_target),
                "convert_failed",
                cp.stderr.strip()[:500],
            )

        os.replace(tmp_path, final_target)
        if final_target != job.src and not args.keep_source:
            job.src.unlink(missing_ok=True)
        return JobResult(
            job.idx,
            str(job.src),
            str(final_target),
            "converted",
            str(channels),
        )
    except Exception as exc:
        return JobResult(
            job.idx,
            str(job.src),
            str(job.target),
            "convert_failed",
            repr(exc)[:500],
        )


def update_stats(stats: Stats, result: JobResult) -> None:
    if result.status in {"converted", "dryrun_convert"}:
        stats.converted += 1
    elif result.status == "skip_mono_flac":
        stats.skipped_mono_flac += 1
    elif result.status == "skip_conflict":
        stats.skipped_conflict += 1
    elif result.status == "probe_failed":
        stats.failed_probe += 1
    elif result.status == "convert_failed":
        stats.failed_convert += 1


def print_progress(done: int, total: int, stats: Stats) -> None:
    print(
        "progress "
        f"{done}/{total} converted={stats.converted} "
        f"skip_mono={stats.skipped_mono_flac} "
        f"skip_conflict={stats.skipped_conflict} "
        f"probe_fail={stats.failed_probe} convert_fail={stats.failed_convert}"
    )


def main() -> int:
    args = parse_args()

    if args.compression_level < 0 or args.compression_level > 12:
        print("Invalid --compression-level (must be in [0, 12]).", file=sys.stderr)
        return 2
    if args.workers < 1:
        print("Invalid --workers (must be >= 1).", file=sys.stderr)
        return 2

    roots = [Path(item).expanduser().resolve() for item in args.roots]
    missing = [str(path) for path in roots if not path.exists()]
    if missing:
        print("Missing roots:", file=sys.stderr)
        for path in missing:
            print(f"- {path}", file=sys.stderr)
        return 2

    jobs, scanned = discover_jobs(roots, limit=args.limit)
    stats = Stats(scanned_files=scanned, matched_audio=len(jobs))
    results: list[JobResult | None] = [None] * len(jobs)

    reserved_targets: set[Path] = set()
    reserved_targets_lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                convert_job, job, args, reserved_targets, reserved_targets_lock
            )
            for job in jobs
        ]
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results[result.idx] = result
            update_stats(stats, result)
            if args.progress_every > 0 and done % args.progress_every == 0:
                print_progress(done, len(jobs), stats)

    report_path = Path(args.report_csv).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["src", "target", "status", "info"])
        for result in results:
            if result is None:
                continue
            writer.writerow([result.src, result.target, result.status, result.info])

    print(
        "done "
        f"dry_run={not args.apply} scanned_files={stats.scanned_files} "
        f"matched_audio={stats.matched_audio} converted={stats.converted} "
        f"skip_mono_flac={stats.skipped_mono_flac} "
        f"skip_conflict={stats.skipped_conflict} "
        f"failed_probe={stats.failed_probe} failed_convert={stats.failed_convert} "
        f"workers={args.workers} report={report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
