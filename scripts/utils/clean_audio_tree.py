#!/usr/bin/env python3
import argparse
import shutil
import subprocess
from pathlib import Path


AUDIO_EXTENSIONS = {
    ".wav",
    ".wave",
    ".aif",
    ".aiff",
    ".aifc",
    ".flac",
    ".mp3",
    ".mp2",
    ".m4a",
    ".m4b",
    ".m4p",
    ".m4r",
    ".m4v",
    ".mov",
    ".aac",
    ".ogg",
    ".oga",
    ".opus",
    ".wma",
    ".alac",
    ".ape",
    ".wv",
    ".tta",
    ".amr",
    ".ac3",
    ".dts",
    ".mka",
    ".caf",
    ".au",
    ".snd",
    ".ra",
}

KEEP_AS_IS_EXTENSIONS = {".mp3", ".ogg", ".flac"}


def _find_files(root_dir: Path):
    return [p for p in root_dir.rglob("*") if p.is_file()]


def _is_metadata_sidecar(file_path: Path) -> bool:
    # macOS AppleDouble sidecar files (e.g. ._track.wav) are not real audio.
    return file_path.name.startswith("._")


def _is_report_omitted_delete_file(file_path: Path) -> bool:
    # Omit from delete report, but still delete on confirmation.
    name_folded = file_path.name.casefold()
    return _is_metadata_sidecar(file_path) or name_folded == ".ds_store"


def _convert_to_flac_16bit(source_path: Path, destination_path: Path, overwrite: bool):
    def run_ffmpeg(tolerant: bool):
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y" if overwrite else "-n",
        ]
        if tolerant:
            # Better resilience with damaged streams (common on broken AAC sources).
            command += ["-err_detect", "ignore_err", "-fflags", "+discardcorrupt"]

        command += [
            "-i",
            str(source_path),
            "-vn",
            "-sn",
            "-dn",
            "-map",
            "a:0",
            "-c:a",
            "flac",
            "-sample_fmt",
            "s16",
            str(destination_path),
        ]
        return subprocess.run(command, capture_output=True, text=True)

    destination_existed_before = destination_path.exists()

    strict = run_ffmpeg(tolerant=False)
    if strict.returncode == 0:
        return True, "", False

    # Some malformed inputs produce a partial output even on failure.
    # Remove that temporary artifact so the recovery pass can run.
    if not destination_existed_before and destination_path.exists():
        try:
            destination_path.unlink()
        except OSError:
            pass

    tolerant = run_ffmpeg(tolerant=True)
    if tolerant.returncode == 0:
        return True, "", True

    # Keep the workspace clean if both passes failed and output did not exist
    # before this conversion attempt.
    if not destination_existed_before and destination_path.exists():
        try:
            destination_path.unlink()
        except OSError:
            pass

    err_text = (tolerant.stderr or strict.stderr or "").strip()
    return False, err_text, False


def _compact_ffmpeg_error(error_text: str, max_lines: int = 4) -> str:
    lines = [line.strip() for line in (error_text or "").splitlines() if line.strip()]
    if not lines:
        return "ffmpeg conversion failed (no error details available)"

    unique_lines = []
    seen = set()
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        unique_lines.append(line)

    if len(unique_lines) <= max_lines:
        return " | ".join(unique_lines)

    return " | ".join(unique_lines[:max_lines]) + f" | ... (+{len(unique_lines) - max_lines} more)"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scan a folder recursively, convert all audio except MP3/OGG "
            "to 16-bit FLAC, delete source after successful conversion, "
            "and optionally delete non-audio files after confirmation."
        )
    )
    parser.add_argument("root_dir", help="Root folder to scan (recursive).")
    parser.add_argument(
        "--trash-list",
        default=None,
        help=(
            "Path for the list of non-audio files to delete "
            "(default: <root_dir>/files_to_delete.txt)."
        ),
    )
    parser.add_argument(
        "--overwrite-flac",
        action="store_true",
        help="Overwrite FLAC destination files if they already exist.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Root folder not found: {root_dir}")

    trash_list_path = (
        Path(args.trash_list).expanduser().resolve()
        if args.trash_list
        else (root_dir / "files_to_delete.txt")
    )

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH.")

    all_files = _find_files(root_dir)
    non_audio_files = []
    hidden_delete_files = []
    delete_candidates = []
    conversion_candidates = []

    for file_path in all_files:
        if file_path.resolve() == trash_list_path:
            continue

        if _is_report_omitted_delete_file(file_path):
            hidden_delete_files.append(file_path)
            delete_candidates.append(file_path)
            continue

        suffix = file_path.suffix.lower()
        if suffix not in AUDIO_EXTENSIONS:
            non_audio_files.append(file_path)
            delete_candidates.append(file_path)
            continue

        if suffix not in KEEP_AS_IS_EXTENSIONS:
            conversion_candidates.append(file_path)

    converted_ok = 0
    converted_fail = 0
    converted_skip = 0
    source_deleted_ok = 0
    source_deleted_fail = 0
    interrupted = False

    for source_path in conversion_candidates:
        destination_path = source_path.with_suffix(".flac")
        if destination_path.exists() and not args.overwrite_flac:
            converted_skip += 1
            print(f"[SKIP] FLAC already exists: {destination_path}")
            continue

        try:
            ok, error_text, recovered = _convert_to_flac_16bit(
                source_path=source_path,
                destination_path=destination_path,
                overwrite=args.overwrite_flac,
            )
        except KeyboardInterrupt:
            interrupted = True
            print("")
            print("[WARN] Interrupted by user. Stopping conversions.")
            break

        if ok:
            converted_ok += 1
            if recovered:
                print(f"[OK] Converted with recovery mode: {source_path} -> {destination_path}")
            else:
                print(f"[OK] Converted: {source_path} -> {destination_path}")
            try:
                source_path.unlink()
                source_deleted_ok += 1
                print(f"[OK] Removed source: {source_path}")
            except Exception as exc:
                source_deleted_fail += 1
                print(f"[ERR] Converted but could not remove source: {source_path} ({exc})")
        else:
            converted_fail += 1
            print(f"[ERR] Conversion failed: {source_path}")
            if error_text:
                print(f"      {_compact_ffmpeg_error(error_text)}")

    trash_list_path.parent.mkdir(parents=True, exist_ok=True)
    with trash_list_path.open("w", encoding="utf-8") as trash_file:
        for file_path in sorted(non_audio_files):
            trash_file.write(f"{file_path}\n")

    print("")
    print(f"[INFO] Scan root: {root_dir}")
    print(f"[INFO] Converted OK: {converted_ok}")
    print(f"[INFO] Converted failed: {converted_fail}")
    print(f"[INFO] Converted skipped: {converted_skip}")
    print(f"[INFO] Source removed after conversion: {source_deleted_ok}")
    print(f"[INFO] Source remove errors: {source_deleted_fail}")
    print(f"[INFO] Non-audio files in report: {len(non_audio_files)}")
    print(f"[INFO] Hidden cleanup files omitted from report: {len(hidden_delete_files)}")
    print(f"[INFO] Total files deletable on confirmation: {len(delete_candidates)}")
    print(f"[INFO] Delete list written to: {trash_list_path}")
    if interrupted:
        print("[INFO] Conversion interrupted, deletion prompt skipped.")
        return 130

    if not delete_candidates:
        print("[INFO] Nothing to delete.")
        return 0

    try:
        answer = input(
            "Delete all deletable files now? (listed + hidden omitted) [y/N]: "
        ).strip().lower()
    except EOFError:
        print("[INFO] No interactive input available. Deletion cancelled by default.")
        return 0
    if answer not in {"y", "yes"}:
        print("[INFO] Deletion cancelled.")
        return 0

    deleted_ok = 0
    deleted_fail = 0
    for file_path in delete_candidates:
        try:
            file_path.unlink()
            deleted_ok += 1
        except Exception as exc:
            deleted_fail += 1
            print(f"[ERR] Delete failed: {file_path} ({exc})")

    print(f"[INFO] Deleted: {deleted_ok}, errors: {deleted_fail}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("")
        print("[WARN] Interrupted by user.")
        raise SystemExit(130)
