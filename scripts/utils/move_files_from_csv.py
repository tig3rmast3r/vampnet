#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
from pathlib import Path


def _strip_outer_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1]
    return value


def _extract_file_path(row: dict, path_column: str) -> str:
    raw = row.get(path_column, "")
    if raw:
        return _strip_outer_quotes(raw)

    # Fallback for CSV rows without headers or with unexpected headers:
    # try common column names, then the second column.
    for candidate in ("file_path", "path", "filepath"):
        raw = row.get(candidate, "")
        if raw:
            return _strip_outer_quotes(raw)

    values = list(row.values())
    if len(values) >= 2 and values[1]:
        return _strip_outer_quotes(values[1])
    return ""


def _resolve_input_path(path_value: str, hierarchy_root: Path) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (hierarchy_root / candidate).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Move files listed in an annotations CSV into a destination folder "
            "while preserving hierarchy relative to --hierarchy-root."
        )
    )
    parser.add_argument("--input-csv", required=True, help="Input annotations CSV path.")
    parser.add_argument("--output-csv", required=True, help="Output report CSV path.")
    parser.add_argument(
        "--hierarchy-root",
        required=True,
        help="Base folder used to compute relative hierarchy (e.g. /data/DATASET).",
    )
    parser.add_argument(
        "--dest-root",
        required=True,
        help="Destination root where files are moved (e.g. /data/DATASET_OUT).",
    )
    parser.add_argument(
        "--path-column",
        default="file_path",
        help="CSV column name containing the source file path (default: file_path).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview operations without moving files.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    hierarchy_root = Path(args.hierarchy_root).expanduser().resolve()
    dest_root = Path(args.dest_root).expanduser().resolve()

    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not hierarchy_root.is_dir():
        raise NotADirectoryError(f"Hierarchy root not found: {hierarchy_root}")
    dest_root.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    moved = 0
    skipped = 0

    with input_csv.open("r", newline="", encoding="utf-8") as f_in, output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(
            f_out,
            fieldnames=[
                "status",
                "source_path",
                "destination_path",
                "relative_path",
                "error",
            ],
        )
        writer.writeheader()

        for row in reader:
            path_value = _extract_file_path(row, args.path_column)
            if not path_value:
                skipped += 1
                writer.writerow(
                    {
                        "status": "skipped_no_path",
                        "source_path": "",
                        "destination_path": "",
                        "relative_path": "",
                        "error": "missing path value in row",
                    }
                )
                continue

            source_path = _resolve_input_path(path_value, hierarchy_root)
            source_key = str(source_path)

            if source_key in seen:
                skipped += 1
                writer.writerow(
                    {
                        "status": "skipped_duplicate",
                        "source_path": source_key,
                        "destination_path": "",
                        "relative_path": "",
                        "error": "duplicate source path in CSV",
                    }
                )
                continue
            seen.add(source_key)

            try:
                rel_path = source_path.relative_to(hierarchy_root)
            except ValueError:
                skipped += 1
                writer.writerow(
                    {
                        "status": "skipped_outside_root",
                        "source_path": source_key,
                        "destination_path": "",
                        "relative_path": "",
                        "error": "source is outside hierarchy root",
                    }
                )
                continue

            destination_path = (dest_root / rel_path).resolve()
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            if not source_path.exists():
                skipped += 1
                writer.writerow(
                    {
                        "status": "skipped_missing_source",
                        "source_path": source_key,
                        "destination_path": str(destination_path),
                        "relative_path": str(rel_path),
                        "error": "source file does not exist",
                    }
                )
                continue

            if destination_path.exists():
                skipped += 1
                writer.writerow(
                    {
                        "status": "skipped_destination_exists",
                        "source_path": source_key,
                        "destination_path": str(destination_path),
                        "relative_path": str(rel_path),
                        "error": "destination already exists",
                    }
                )
                continue

            if args.dry_run:
                moved += 1
                writer.writerow(
                    {
                        "status": "dry_run",
                        "source_path": source_key,
                        "destination_path": str(destination_path),
                        "relative_path": str(rel_path),
                        "error": "",
                    }
                )
                continue

            try:
                shutil.move(str(source_path), str(destination_path))
                moved += 1
                writer.writerow(
                    {
                        "status": "moved",
                        "source_path": source_key,
                        "destination_path": str(destination_path),
                        "relative_path": str(rel_path),
                        "error": "",
                    }
                )
            except Exception as exc:  # pragma: no cover - best-effort reporting
                skipped += 1
                writer.writerow(
                    {
                        "status": "error",
                        "source_path": source_key,
                        "destination_path": str(destination_path),
                        "relative_path": str(rel_path),
                        "error": str(exc),
                    }
                )

    print(f"done: moved_or_planned={moved} skipped={skipped} report={output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
