#!/usr/bin/env python3
"""
Recursive dataset filename normalizer.

- Removes non-ASCII characters via NFKD transliteration.
- Replaces special characters with underscores.
- Keeps naming coherent across multiple roots by using one global name map.
- Uses two-phase renaming (temp -> final) per folder to avoid swap/collision issues.

Dry-run by default. Use --apply to execute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


ALLOWED_CHARS_RE = re.compile(r"[^A-Za-z0-9._ \-]+")
WHITESPACE_RE = re.compile(r"\s+")
UNDERSCORE_RE = re.compile(r"_+")


@dataclass
class Stats:
    roots_checked: int = 0
    entries_scanned: int = 0
    unique_components: int = 0
    changed_components: int = 0
    renamed_files: int = 0
    renamed_dirs: int = 0
    collisions_resolved: int = 0
    dry_run: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize dataset names recursively across multiple roots (dry-run by default)."
    )
    parser.add_argument(
        "roots",
        nargs="+",
        help="Root folders to process recursively.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply filesystem renames. If omitted, only prints dry-run summary.",
    )
    parser.add_argument(
        "--report-json",
        default="",
        help="Optional path to write JSON report.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=30,
        help="How many sample renames to print in summary.",
    )
    return parser.parse_args()


def sanitize_component(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    ascii_name = ALLOWED_CHARS_RE.sub("_", ascii_name)
    ascii_name = WHITESPACE_RE.sub(" ", ascii_name).strip()
    ascii_name = UNDERSCORE_RE.sub("_", ascii_name)
    ascii_name = ascii_name.strip(" ")
    if ascii_name in {"", ".", ".."}:
        ascii_name = "unnamed"
    return ascii_name


def short_hash(value: str, size: int = 8) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:size]


def gather_unique_components(roots: Iterable[Path], stats: Stats) -> set[str]:
    unique: set[str] = set()
    for root in roots:
        stats.roots_checked += 1
        for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
            stats.entries_scanned += len(dirnames) + len(filenames)
            unique.update(dirnames)
            unique.update(filenames)
    return unique


def build_global_name_map(unique_components: Iterable[str], stats: Stats) -> tuple[Dict[str, str], Dict[str, List[str]]]:
    by_base: Dict[str, List[str]] = {}
    for original in unique_components:
        base = sanitize_component(original)
        by_base.setdefault(base, []).append(original)

    name_map: Dict[str, str] = {}
    collisions: Dict[str, List[str]] = {}
    for base, originals in by_base.items():
        originals_sorted = sorted(originals)
        if len(originals_sorted) == 1:
            name_map[originals_sorted[0]] = base
            continue

        collisions[base] = originals_sorted
        preferred = None
        if base in originals_sorted:
            preferred = base
        else:
            preferred = originals_sorted[0]

        used_targets = set()
        for original in originals_sorted:
            if original == preferred:
                target = base
            else:
                target = f"{base}__{short_hash(original)}"
            if target in used_targets:
                target = f"{target}_{short_hash(original + '_x', size=6)}"
            used_targets.add(target)
            name_map[original] = target

    stats.unique_components = len(set(unique_components))
    stats.changed_components = sum(1 for k, v in name_map.items() if k != v)
    stats.collisions_resolved = sum(len(v) for v in collisions.values())
    return name_map, collisions


def safe_temp_name(parent: Path, original: str) -> str:
    base = f".__rename_tmp__{short_hash(original)}"
    candidate = base
    idx = 0
    while (parent / candidate).exists():
        idx += 1
        candidate = f"{base}_{idx}"
    return candidate


def process_one_directory(
    parent: Path,
    names: List[str],
    name_map: Dict[str, str],
    apply: bool,
    stats: Stats,
    rename_samples: List[dict],
) -> None:
    final_to_originals: Dict[str, List[str]] = {}
    for name in names:
        final_name = name_map[name]
        final_to_originals.setdefault(final_name, []).append(name)

    duplicates = {k: v for k, v in final_to_originals.items() if len(v) > 1}
    if duplicates:
        details = "; ".join(f"{k} <- {v}" for k, v in list(duplicates.items())[:5])
        raise RuntimeError(f"Unresolvable collision in {parent}: {details}")

    changes = [(name, name_map[name]) for name in names if name_map[name] != name]
    if not changes:
        return

    if not apply:
        for old_name, new_name in changes:
            old_path = parent / old_name
            kind = "dir" if old_path.is_dir() else "file"
            if kind == "dir":
                stats.renamed_dirs += 1
            else:
                stats.renamed_files += 1
            if len(rename_samples) < 1000:
                rename_samples.append(
                    {
                        "kind": kind,
                        "old": str(old_path),
                        "new": str(parent / new_name),
                    }
                )
        return

    staged: List[tuple[str, str, str]] = []
    for old_name, new_name in changes:
        old_path = parent / old_name
        tmp_name = safe_temp_name(parent, old_name)
        tmp_path = parent / tmp_name
        os.rename(old_path, tmp_path)
        staged.append((old_name, tmp_name, new_name))

    try:
        for old_name, tmp_name, new_name in staged:
            tmp_path = parent / tmp_name
            final_path = parent / new_name
            if final_path.exists():
                raise RuntimeError(f"Target already exists after staging: {final_path}")
            os.rename(tmp_path, final_path)
            kind = "dir" if final_path.is_dir() else "file"
            if kind == "dir":
                stats.renamed_dirs += 1
            else:
                stats.renamed_files += 1
            if len(rename_samples) < 1000:
                rename_samples.append(
                    {
                        "kind": kind,
                        "old": str(parent / old_name),
                        "new": str(final_path),
                    }
                )
    except Exception:
        for old_name, tmp_name, _new_name in reversed(staged):
            tmp_path = parent / tmp_name
            old_path = parent / old_name
            if tmp_path.exists() and not old_path.exists():
                os.rename(tmp_path, old_path)
        raise


def process_root(root: Path, name_map: Dict[str, str], apply: bool, stats: Stats, rename_samples: List[dict]) -> None:
    for dirpath, dirnames, filenames in os.walk(root, topdown=False, followlinks=False):
        parent = Path(dirpath)
        process_one_directory(parent, filenames, name_map, apply, stats, rename_samples)
        process_one_directory(parent, dirnames, name_map, apply, stats, rename_samples)


def main() -> int:
    args = parse_args()
    roots = [Path(p).expanduser().resolve() for p in args.roots]
    missing = [str(p) for p in roots if not p.exists()]
    if missing:
        print("Missing roots:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 2

    stats = Stats(dry_run=not args.apply)
    unique_components = gather_unique_components(roots, stats)
    name_map, collisions = build_global_name_map(unique_components, stats)
    rename_samples: List[dict] = []

    for root in roots:
        process_root(root, name_map, args.apply, stats, rename_samples)

    payload = {
        "roots": [str(p) for p in roots],
        "dry_run": stats.dry_run,
        "stats": {
            "roots_checked": stats.roots_checked,
            "entries_scanned": stats.entries_scanned,
            "unique_components": stats.unique_components,
            "changed_components": stats.changed_components,
            "renamed_files": stats.renamed_files,
            "renamed_dirs": stats.renamed_dirs,
            "collisions_resolved_component_count": stats.collisions_resolved,
            "collision_bases_count": len(collisions),
        },
        "sample_renames": rename_samples[: max(0, args.preview)],
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.report_json:
        report_path = Path(args.report_json).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + os.linesep, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
