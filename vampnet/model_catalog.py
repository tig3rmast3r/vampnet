from __future__ import annotations

import gc
import re
from dataclasses import dataclass
from pathlib import Path

import torch


DURATION_PATTERN = re.compile(r"(?:^|[_-])(\d+(?:\.\d+)?)s(?:[_-]|$)", re.IGNORECASE)
SYSTEM_MODEL_NAMES = {"wavebeat.pth", "codec.pth", "codec_2024.pth", "codec_dac.pth"}


@dataclass(frozen=True)
class ModelCatalogEntry:
    id: str
    path: Path
    role: str
    chunk_size_s: float
    n_codebooks: int
    n_conditioning_codebooks: int
    n_layers: int
    n_heads: int
    embedding_dim: int
    vocab_size: int
    latent_dim: int

    @property
    def label(self) -> str:
        return (
            f"{self.id} [{self.chunk_size_s:g}s, "
            f"L{self.n_layers}, H{self.n_heads}, D{self.embedding_dim}]"
        )


def duration_from_name(path: Path, role: str) -> float:
    matches = DURATION_PATTERN.findall(path.stem)
    if len(matches) > 1:
        raise ValueError(f"Ambiguous duration tokens in {path.name}")
    if matches:
        duration = float(matches[0])
        if duration <= 0:
            raise ValueError(f"Invalid duration in {path.name}")
        return duration
    return 10.0 if role == "coarse" else 3.0


def _metadata_kwargs(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", mmap=True)
    try:
        if not isinstance(payload, dict):
            return {}
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            return {}
        kwargs = metadata.get("kwargs")
        return dict(kwargs) if isinstance(kwargs, dict) else {}
    finally:
        del payload
        gc.collect()


def inspect_model(path: Path, models_root: Path) -> ModelCatalogEntry | None:
    if path.name.lower() in SYSTEM_MODEL_NAMES:
        return None
    kwargs = _metadata_kwargs(path)
    required = {
        "n_heads", "n_layers", "n_codebooks", "n_conditioning_codebooks",
        "embedding_dim", "vocab_size", "latent_dim",
    }
    if not required.issubset(kwargs):
        return None
    n_codebooks = int(kwargs["n_codebooks"])
    n_conditioning = int(kwargs["n_conditioning_codebooks"])
    if n_codebooks <= 0 or n_conditioning < 0 or n_conditioning >= n_codebooks:
        return None
    role = "coarse" if n_conditioning == 0 else "c2f"
    resolved_root = models_root.resolve()
    resolved_path = path.resolve()
    entry_id = resolved_path.relative_to(resolved_root).as_posix()
    return ModelCatalogEntry(
        id=entry_id,
        path=resolved_path,
        role=role,
        chunk_size_s=duration_from_name(path, role),
        n_codebooks=n_codebooks,
        n_conditioning_codebooks=n_conditioning,
        n_layers=int(kwargs["n_layers"]),
        n_heads=int(kwargs["n_heads"]),
        embedding_dim=int(kwargs["embedding_dim"]),
        vocab_size=int(kwargs["vocab_size"]),
        latent_dim=int(kwargs["latent_dim"]),
    )


def scan_model_catalog(models_root: str | Path) -> list[ModelCatalogEntry]:
    root = Path(models_root).expanduser().resolve()
    if not root.is_dir():
        return []
    entries: list[ModelCatalogEntry] = []
    for path in sorted(root.rglob("*.pth")):
        try:
            entry = inspect_model(path, root)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            continue
        if entry is not None:
            entries.append(entry)
    return entries


def validate_model_pair(coarse: ModelCatalogEntry, c2f: ModelCatalogEntry) -> None:
    if coarse.role != "coarse" or c2f.role != "c2f":
        raise ValueError("Select one Coarse model and one C2F model")
    if coarse.n_codebooks != c2f.n_conditioning_codebooks:
        raise ValueError(
            f"Incompatible codebooks: Coarse has {coarse.n_codebooks}, "
            f"C2F expects {c2f.n_conditioning_codebooks} conditioning codebooks"
        )
    for field, label in (("vocab_size", "vocabulary"), ("latent_dim", "latent dimension")):
        if getattr(coarse, field) != getattr(c2f, field):
            raise ValueError(f"Incompatible {label}: {getattr(coarse, field)} vs {getattr(c2f, field)}")


def resolve_model_entry(entries: dict[str, ModelCatalogEntry], value: object) -> ModelCatalogEntry | None:
    selected = str(value or "")
    direct = entries.get(selected)
    if direct is not None:
        return direct
    return next((entry for entry in entries.values() if entry.label == selected), None)


def model_dropdown_labels(entries: dict[str, ModelCatalogEntry]) -> list[str]:
    return [entry.label for entry in entries.values()]


def model_dropdown_value(entries: dict[str, ModelCatalogEntry], active_id: str) -> str | None:
    entry = entries.get(str(active_id))
    return entry.label if entry is not None else None
