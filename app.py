from pathlib import Path
from typing import Tuple
import yaml
import tempfile
import uuid
import zipfile
import json
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import gc
import threading

import numpy as np
import audiotools as at
import argbind
import torch

import gradio as gr
from vampnet.interface import Interface, _load_model
from vampnet import mask as pmask
from vampnet.model_catalog import (
    model_dropdown_labels,
    model_dropdown_value,
    resolve_model_entry,
    scan_model_catalog,
    validate_model_pair,
)

Interface = argbind.bind(Interface)
# AudioLoader = argbind.bind(at.data.datasets.AudioLoader)

conf = argbind.parse_args()


from torch_pitch_shift import pitch_shift, get_fast_shifts
def shift_pitch(signal, interval: int):
    signal.samples = pitch_shift(
        signal.samples, 
        shift=interval, 
        sample_rate=signal.sample_rate
    )
    return signal

def load_interface():
    with argbind.scope(conf):
        interface = Interface()
        # loader = AudioLoader()
        print(f"interface device is {interface.device}")
        return interface




interface = load_interface()

MODELS_ROOT = Path("models").resolve()
MODEL_CATALOG = scan_model_catalog(MODELS_ROOT)
MODEL_ENTRIES = {entry.id: entry for entry in MODEL_CATALOG}
COARSE_MODEL_ENTRIES = {entry.id: entry for entry in MODEL_CATALOG if entry.role == "coarse"}
C2F_MODEL_ENTRIES = {entry.id: entry for entry in MODEL_CATALOG if entry.role == "c2f"}
MODEL_LOCK = threading.RLock()


def _configured_model_id(config_key: str, entries: dict) -> str:
    configured = conf.get(config_key) if hasattr(conf, "get") else None
    if configured:
        configured_path = Path(str(configured)).expanduser().resolve()
        for entry_id, entry in entries.items():
            if entry.path == configured_path:
                return entry_id
    return next(iter(entries), "")


ACTIVE_MODEL_IDS = {
    "coarse": _configured_model_id("Interface.coarse_ckpt", COARSE_MODEL_ENTRIES),
    "c2f": _configured_model_id("Interface.coarse2fine_ckpt", C2F_MODEL_ENTRIES),
}
ACTIVE_MODEL_LORAS = {
    "coarse": conf.get("Interface.coarse_lora_ckpt") if hasattr(conf, "get") else None,
    "c2f": conf.get("Interface.coarse2fine_lora_ckpt") if hasattr(conf, "get") else None,
}


def _active_model_path(role: str) -> str:
    entry = MODEL_ENTRIES.get(ACTIVE_MODEL_IDS.get(role, ""))
    return str(entry.path) if entry is not None else ""


def _select_models_locked(coarse_id: str, c2f_id: str) -> str:
    coarse_entry = resolve_model_entry(COARSE_MODEL_ENTRIES, coarse_id)
    c2f_entry = resolve_model_entry(C2F_MODEL_ENTRIES, c2f_id)
    if coarse_entry is None or c2f_entry is None:
        raise ValueError("Unknown model selection")
    validate_model_pair(coarse_entry, c2f_entry)
    requested = {"coarse": coarse_entry, "c2f": c2f_entry}
    changed = [role for role in ("coarse", "c2f") if ACTIVE_MODEL_IDS.get(role) != requested[role].id]
    if not changed:
        return (
            f"Active — Coarse: {coarse_entry.id} ({coarse_entry.chunk_size_s:g}s), "
            f"C2F: {c2f_entry.id} ({c2f_entry.chunk_size_s:g}s)"
        )

    with MODEL_LOCK:
        old_models = {role: getattr(interface, role) for role in changed}
        new_models = {}
        for model in old_models.values():
            model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            for role in changed:
                entry = requested[role]
                new_models[role] = _load_model(
                    ckpt=str(entry.path),
                    device=interface.device,
                    chunk_size_s=entry.chunk_size_s,
                )
        except Exception:
            for model in new_models.values():
                model.to("cpu")
            new_models.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for model in old_models.values():
                model.to(interface.device)
            raise

        for role, model in new_models.items():
            setattr(interface, role, model)
            ACTIVE_MODEL_IDS[role] = requested[role].id
            ACTIVE_MODEL_LORAS[role] = None
        old_models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return (
        f"Active — Coarse: {coarse_entry.id} ({coarse_entry.chunk_size_s:g}s), "
        f"C2F: {c2f_entry.id} ({c2f_entry.chunk_size_s:g}s)"
    )


def select_models(coarse_id: str, c2f_id: str) -> str:
    with MODEL_LOCK:
        return _select_models_locked(coarse_id, c2f_id)


OUT_DIR = Path("gradio-outputs")
OUT_DIR.mkdir(exist_ok=True, parents=True)


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime_model_info(model) -> dict:
    if model is None:
        return {}
    info = {"class": model.__class__.__name__}
    for field in (
        "chunk_size_s",
        "n_codebooks",
        "n_conditioning_codebooks",
        "vocab_size",
        "embedding_dim",
        "n_layers",
        "n_heads",
        "flash_attn",
    ):
        if hasattr(model, field):
            info[field] = _json_safe(getattr(model, field))
    try:
        info["parameter_count"] = int(sum(parameter.numel() for parameter in model.parameters()))
    except Exception:
        pass
    return info


def _write_settings(out_dir: Path, settings: dict) -> None:
    settings["updated_at_utc"] = _utc_now()
    destination = out_dir / "settings.json"
    temporary = out_dir / "settings.json.tmp"
    temporary.write_text(json.dumps(_json_safe(settings), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)


def _request_settings(data: dict) -> dict:
    return {
        "input_audio": str(data[input_audio]),
        "mask": {
            "periodic_period": data[periodic_p],
            "periodic_width": data[periodic_w],
            "onset_mask_width": data[onset_mask_width],
            "beat_mask_width_tokens": data[beat_mask_width],
            "beat_mask_downbeats_only": data[beat_mask_downbeats],
            "prefix_seconds": data[prefix_s],
            "suffix_seconds": data[suffix_s],
            "random_mask_intensity": data[rand_mask_intensity],
            "dropout": data[dropout],
            "first_upper_codebook_to_mask": data[n_mask_codebooks],
            "conditioning_codebooks": data[n_conditioning_codebooks],
            "manual_hint_edits": data[manual_hint_edits],
        },
        "sampling": {
            "mask_temperature": data[masktemp],
            "sample_temperature": data[sampletemp],
            "top_p": data[top_p],
            "typical_filtering": data[typical_filtering],
            "typical_mass": data[typical_mass],
            "typical_min_tokens": data[typical_min_tokens],
            "sample_cutoff": data[sample_cutoff],
            "fixed_sample_cutoff": data[fixed_sample_cutoff],
            "sample_cutoff_steps": data[sample_cutoff_steps],
            "num_steps": data[num_steps],
        },
        "generation": {
            "seed": data[seed],
            "batch_variant": data[batch_variant],
            "use_coarse2fine": data[use_coarse2fine],
            "pitch_shift_semitones": data[pitch_shift_amt],
            "stretch_factor": data[stretch_factor],
        },
    }


def load_audio(file):
    print(file)
    filepath = file.name
    sig = at.AudioSignal.salient_excerpt(
        filepath, 
        duration=interface.coarse.chunk_size_s
    )
    sig = interface.preprocess(sig)

    out_dir = OUT_DIR / "tmp" / str(uuid.uuid4())
    out_dir.mkdir(parents=True, exist_ok=True)
    sig.write(out_dir / "input.wav")
    return sig.path_to_file


def load_example_audio():
    return "./assets/example.wav"


def _vamp_impl(data, return_mask=False, return_variant_archive=False):
    if seed != 0:
        at.util.seed(data[seed])

    out_dir = OUT_DIR / str(uuid.uuid4())
    out_dir.mkdir()
    settings = {
        "schema_version": 1,
        "status": "started",
        "created_at_utc": _utc_now(),
        "output_directory": str(out_dir.resolve()),
        "invocation": "gradio_ui" if return_mask else "api",
        "request": _request_settings(data),
        "models": {
            "configured_paths": {
                "coarse": _config_value("Interface.coarse_ckpt"),
                "coarse_lora": _config_value("Interface.coarse_lora_ckpt"),
                "coarse2fine": _config_value("Interface.coarse2fine_ckpt"),
                "coarse2fine_lora": _config_value("Interface.coarse2fine_lora_ckpt"),
                "codec": _config_value("Interface.codec_ckpt"),
                "wavebeat": _config_value("Interface.wavebeat_ckpt"),
            },
            "coarse": _runtime_model_info(interface.coarse),
            "coarse2fine": _runtime_model_info(interface.c2f),
            "codec": {
                "class": interface.codec.__class__.__name__,
                "sample_rate": getattr(interface.codec, "sample_rate", None),
                "hop_length": getattr(interface.codec, "hop_length", None),
            },
            "device": str(interface.device),
        },
        "effective": {
            "coarse_mask_temperature": data[masktemp] * 10,
            "coarse_sampling_temperature": data[sampletemp],
            "c2f_mask_temperature": data[masktemp] * 10,
            "c2f_sampling_temperature": data[sampletemp],
            "sample_cutoff_steps": data[sample_cutoff_steps],
            "num_steps": data[num_steps],
            "seed": data[seed] if data[seed] > 0 else None,
        },
    }
    _write_settings(out_dir, settings)
    sig = at.AudioSignal(data[input_audio])
    sig = interface.preprocess(sig)

    loudness = sig.loudness()
    print(f"input loudness is {loudness}")

    if data[pitch_shift_amt] != 0:
        sig = shift_pitch(sig, data[pitch_shift_amt])

    processed_input_path = out_dir / "input.wav"
    sig.write(processed_input_path)
    settings["input"] = {
        "source_path": str(data[input_audio]),
        "processed_path": str(processed_input_path),
        "sample_rate": getattr(sig, "sample_rate", None),
        "duration_seconds": getattr(sig, "duration", None),
        "loudness_before_pitch_shift": loudness,
        "loudness_encoded": sig.loudness(),
    }
    _write_settings(out_dir, settings)

    z = interface.encode(sig)

    ncc = data[n_conditioning_codebooks]

    # build the mask
    mask = pmask.linear_random(z, data[rand_mask_intensity])
    mask = pmask.mask_and(
        mask, pmask.inpaint(
            z,
            interface.s2t(data[prefix_s]),
            interface.s2t(data[suffix_s])
        )
    )
    mask = pmask.mask_and(
        mask, pmask.periodic_mask(
            z,
            data[periodic_p],
            data[periodic_w],
            random_roll=True
        )
    )
    if data[onset_mask_width] > 0:
        mask = pmask.mask_or(
            mask, pmask.onset_mask(sig, z, interface, width=data[onset_mask_width])
        )
    if data[beat_mask_width] > 0:
        beat_mask = interface.make_beat_mask(
            sig,
            after_beat_s=_token_width_to_seconds(data[beat_mask_width]),
            mask_upbeats=not data[beat_mask_downbeats],
        )
        if beat_mask.shape != mask.shape:
            beat_mask = beat_mask[:, :1, :].repeat(1, mask.shape[1], 1)
        mask = pmask.mask_and(mask, beat_mask)
    mask = _apply_manual_hint_edits(mask, data)

    # these should be the last two mask ops
    mask = pmask.dropout(mask, data[dropout])
    mask = pmask.codebook_unmask(mask, ncc)
    mask = pmask.codebook_mask(mask, int(data[n_mask_codebooks]))

    manual_on, manual_off = _manual_hint_tokens(data)
    settings["mask_result"] = {
        "shape": list(mask.shape),
        "masked_tokens": int(mask[:, 0, :].sum().detach().cpu().item()),
        "total_tokens": int(mask[:, 0, :].numel()),
        "manual_on_tokens": manual_on,
        "manual_off_tokens": manual_off,
    }
    _write_settings(out_dir, settings)



    print(f"dropout {data[dropout]}")
    print(f"masktemp {data[masktemp]}")
    print(f"sampletemp {data[sampletemp]}")
    print(f"top_p {data[top_p]}")
    print(f"prefix_s {data[prefix_s]}")
    print(f"suffix_s {data[suffix_s]}")
    print(f"rand_mask_intensity {data[rand_mask_intensity]}")
    print(f"num_steps {data[num_steps]}")
    print(f"periodic_p {data[periodic_p]}")
    print(f"periodic_w {data[periodic_w]}")
    print(f"n_conditioning_codebooks {data[n_conditioning_codebooks]}")
    print(f"use_coarse2fine {data[use_coarse2fine]}")
    print(f"onset_mask_width {data[onset_mask_width]}")
    print(f"beat_mask_width_tokens {data[beat_mask_width]}")
    print(f"beat_mask_downbeats {data[beat_mask_downbeats]}")
    print(f"stretch_factor {data[stretch_factor]}")
    print(f"seed {data[seed]}")
    print(f"pitch_shift_amt {data[pitch_shift_amt]}")
    print(f"sample_cutoff {data[sample_cutoff]}")
    print(f"fixed_sample_cutoff {data[fixed_sample_cutoff]}")
    print(f"sample_cutoff_steps {data[sample_cutoff_steps]}")
    print(f"batch_variant {data[batch_variant]}")
    _top_p = data[top_p] if data[top_p] > 0 else None
    _seed = data[seed] if data[seed] > 0 else None
    _batch_variant = max(1, int(data[batch_variant]))

    zv = z.expand(_batch_variant, -1, -1)
    batch_mask = mask.expand(_batch_variant, -1, -1)
    zv, mask_z = interface.coarse_vamp(
        zv,
        mask=batch_mask,
        sampling_steps=data[num_steps],
        mask_temperature=data[masktemp] * 10,
        sampling_temperature=data[sampletemp],
        return_mask=True,
        typical_filtering=data[typical_filtering],
        typical_mass=data[typical_mass],
        typical_min_tokens=data[typical_min_tokens],
        top_p=_top_p,
        gen_fn=interface.coarse.generate,
        seed=_seed,
        sample_cutoff=data[sample_cutoff],
        fixed_sample_cutoff=data[fixed_sample_cutoff],
        sample_cutoff_steps=data[sample_cutoff_steps],
    )

    if data[use_coarse2fine]:
        zv = interface.coarse_to_fine(
            zv,
            mask_temperature=data[masktemp] * 10,
            sampling_temperature=data[sampletemp],
            mask=batch_mask,
            sampling_steps=data[num_steps],
            sample_cutoff=data[sample_cutoff],
            fixed_sample_cutoff=data[fixed_sample_cutoff],
            sample_cutoff_steps=data[sample_cutoff_steps],
            seed=_seed,
        )
    else:
        zv = zv[:, :interface.coarse.n_codebooks, :]

    selected_idx = _batch_variant - 1
    selected_zv = zv[selected_idx:selected_idx + 1]
    selected_mask_z = mask_z[selected_idx:selected_idx + 1] if mask_z is not None else None

    sig = interface.to_signal(selected_zv).cpu()
    print("done")

    output_loudness = sig.loudness()
    print(f"output loudness is {output_loudness}")
    sig = sig.normalize(loudness)
    normalized_loudness = sig.loudness()
    print(f"normalized loudness is {normalized_loudness}")

    output_filename = "output.wav"
    sig.write(out_dir / output_filename)
    output_path = sig.path_to_file
    variant_archive = None
    if return_variant_archive:
        variant_dir = out_dir / "variants"
        variant_dir.mkdir()
        for variant_idx in range(_batch_variant):
            variant_sig = interface.to_signal(zv[variant_idx:variant_idx + 1]).cpu()
            variant_sig = variant_sig.normalize(loudness)
            variant_sig.write(variant_dir / f"variant_{variant_idx + 1}.wav")
        variant_archive = str(out_dir / "variants.zip")
        with zipfile.ZipFile(variant_archive, "w") as zf:
            for file in sorted(variant_dir.glob("variant_*.wav")):
                zf.write(file, file.name)
    settings["status"] = "completed"
    settings["output"] = {
        "filename": output_filename,
        "path": str(output_path),
        "loudness_before_normalization": output_loudness,
        "loudness_after_normalization": normalized_loudness,
        "mask_audio_written": False,
        "variant_archive": variant_archive,
    }

    if return_mask:
        mask = interface.to_signal(selected_mask_z).cpu()
        mask.write(out_dir / "mask.wav")
        settings["output"]["mask_audio_written"] = True
        settings["output"]["mask_audio_path"] = str(mask.path_to_file)
        _write_settings(out_dir, settings)
        return output_path, mask.path_to_file
    _write_settings(out_dir, settings)
    if return_variant_archive:
        return output_path, variant_archive
    return output_path


def _vamp(data, return_mask=False, return_variant_archive=False):
    with MODEL_LOCK:
        return _vamp_impl(
            data,
            return_mask=return_mask,
            return_variant_archive=return_variant_archive,
        )

def vamp(data):
    return _vamp(data, return_mask=True)

def api_vamp(data):
    return _vamp(data, return_mask=False)


def _config_value(key):
    if key == "Interface.coarse_ckpt":
        return _active_model_path("coarse")
    if key == "Interface.coarse2fine_ckpt":
        return _active_model_path("c2f")
    if key == "Interface.coarse_lora_ckpt":
        return ACTIVE_MODEL_LORAS["coarse"]
    if key == "Interface.coarse2fine_lora_ckpt":
        return ACTIVE_MODEL_LORAS["c2f"]
    return conf.get(key) if hasattr(conf, "get") else None


def api_model_info_event():
    return json.dumps(
        {
            "coarse_ckpt": _config_value("Interface.coarse_ckpt"),
            "coarse2fine_ckpt": _config_value("Interface.coarse2fine_ckpt"),
            "active_model_ids": dict(ACTIVE_MODEL_IDS),
            "available_models": [
                {
                    "id": entry.id,
                    "role": entry.role,
                    "chunk_size_s": entry.chunk_size_s,
                    "n_codebooks": entry.n_codebooks,
                    "n_conditioning_codebooks": entry.n_conditioning_codebooks,
                    "n_layers": entry.n_layers,
                    "n_heads": entry.n_heads,
                    "embedding_dim": entry.embedding_dim,
                }
                for entry in MODEL_CATALOG
            ],
        }
    )


def _mask_zero_windows(mask, max_windows=2000):
    if mask is None:
        return []
    values = mask.detach().cpu()
    while values.ndim > 1:
        values = values[0]
    active = (values == 0).numpy().astype(bool)
    windows = []
    start = None
    for index, is_active in enumerate(active.tolist() + [False]):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            windows.append([interface.t2s(start), interface.t2s(index)])
            start = None
            if len(windows) >= max_windows:
                break
    return windows


def _mask_from_active(active, reference):
    mask = torch.ones_like(reference)
    mask[active] = 0
    return mask


def _token_width_to_seconds(tokens):
    return int(tokens) * interface.codec.hop_length / interface.codec.sample_rate


def _manual_hint_tokens(data):
    raw = data.get(manual_hint_edits, "") if "manual_hint_edits" in globals() else ""
    if not raw:
        return [], []
    try:
        payload = json.loads(raw)
    except Exception:
        return [], []
    on = [int(value) for value in payload.get("on", [])]
    off = [int(value) for value in payload.get("off", [])]
    return on, off


def _apply_manual_hint_edits(mask, data):
    manual_on, manual_off = _manual_hint_tokens(data)
    if not manual_on and not manual_off:
        return mask
    mask = mask.clone()
    length = mask.shape[-1]
    for token in manual_on:
        if 0 <= token < length:
            mask[:, :, token] = 0
    for token in manual_off:
        if 0 <= token < length:
            mask[:, :, token] = 1
    return mask


def _beat_windows_from_times(times, after_s):
    windows = []
    for value in np.asarray(times).tolist():
        start = max(0.0, float(value))
        windows.append([start, start + float(after_s)])
    return windows


def _hint_preview(data):
    if seed != 0:
        at.util.seed(data[seed])

    sig = at.AudioSignal(data[input_audio])
    sig = interface.preprocess(sig)
    z = interface.encode(sig)
    preview = {
        "hint_preview_version": 2,
        "duration": float(sig.duration),
        "periodic_windows": [],
        "periodic_accepted_windows": [],
        "periodic_rejected_windows": [],
        "onset_windows": [],
        "beat_windows": [],
        "downbeat_windows": [],
        "manual_on_windows": [],
        "manual_off_windows": [],
        "beats": [],
        "downbeats": [],
        "errors": [],
    }

    periodic = None
    onset = None

    try:
        periodic = pmask.periodic_mask(
            z,
            data[periodic_p],
            data[periodic_w],
            random_roll=True,
        )
        preview["periodic_windows"] = _mask_zero_windows(periodic)
        preview["periodic_accepted_windows"] = preview["periodic_windows"]
    except Exception as exc:
        preview["errors"].append(f"periodic: {exc}")

    if data[onset_mask_width] > 0:
        try:
            onset = pmask.onset_mask(sig, z, interface, width=data[onset_mask_width])
            preview["onset_windows"] = _mask_zero_windows(onset)
            if periodic is not None:
                periodic_active = periodic == 0
                onset_active = onset == 0
                accepted = periodic_active & onset_active
                rejected = periodic_active & ~onset_active
                preview["periodic_accepted_windows"] = _mask_zero_windows(_mask_from_active(accepted, periodic))
                preview["periodic_rejected_windows"] = _mask_zero_windows(_mask_from_active(rejected, periodic))
        except Exception as exc:
            preview["errors"].append(f"onset: {exc}")

    if data[beat_mask_width] > 0:
        try:
            assert interface.beat_tracker is not None, "No beat tracker loaded"
            beats, downbeats = interface.beat_tracker.extract_beats(sig)
            beats = np.asarray(beats, dtype=float)
            downbeats = np.asarray(downbeats, dtype=float)
            if data[beat_mask_downbeats]:
                active_beats = np.asarray([], dtype=float)
            else:
                downbeat_tokens = set(interface.s2t(downbeats).tolist())
                beat_tokens = interface.s2t(beats)
                active_beats = beats[[int(token) not in downbeat_tokens for token in beat_tokens]]
            after_s = _token_width_to_seconds(data[beat_mask_width])
            preview["beats"] = active_beats.tolist()
            preview["downbeats"] = downbeats.tolist()
            preview["beat_windows"] = _beat_windows_from_times(active_beats, after_s)
            preview["downbeat_windows"] = _beat_windows_from_times(downbeats, after_s)
        except Exception as exc:
            preview["errors"].append(f"beat: {exc}")

    manual_on, manual_off = _manual_hint_tokens(data)
    preview["manual_on_windows"] = [
        [interface.t2s(token), interface.t2s(token + 1)]
        for token in manual_on
    ]
    preview["manual_off_windows"] = [
        [interface.t2s(token), interface.t2s(token + 1)]
        for token in manual_off
    ]

    return preview


def api_hint_preview_event(*values):
    return json.dumps(_hint_preview(as_data(_inputs, values)))
        
def save_vamp(data):
    out_dir = OUT_DIR / "saved" / str(uuid.uuid4())
    out_dir.mkdir(parents=True, exist_ok=True)

    sig_in = at.AudioSignal(data[input_audio])
    sig_out = at.AudioSignal(data[output_audio])

    sig_in.write(out_dir / "input.wav")
    sig_out.write(out_dir / "output.wav")
    
    _data = {
        "masktemp": data[masktemp],
        "sampletemp": data[sampletemp],
        "top_p": data[top_p],
        "prefix_s": data[prefix_s],
        "suffix_s": data[suffix_s],
        "rand_mask_intensity": data[rand_mask_intensity],
        "num_steps": data[num_steps],
        "notes": data[notes_text],
        "periodic_period": data[periodic_p],
        "periodic_width": data[periodic_w],
        "n_conditioning_codebooks": data[n_conditioning_codebooks], 
        "use_coarse2fine": data[use_coarse2fine],
        "stretch_factor": data[stretch_factor],
        "seed": data[seed],
        "samplecutoff": data[sample_cutoff],
        "fixed_sample_cutoff": data[fixed_sample_cutoff],
        "sample_cutoff_steps": data[sample_cutoff_steps],
        "batch_variant": data[batch_variant],
    }

    # save with yaml
    with open(out_dir / "data.yaml", "w") as f:
        yaml.dump(_data, f)

    import zipfile
    zip_path = str(out_dir.with_suffix(".zip"))
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file in out_dir.iterdir():
            zf.write(file, file.name)

    return f"saved! your save code is {out_dir.stem}", zip_path


def harp_vamp(_input_audio, _beat_mask_width, _sampletemp):

    out_dir = OUT_DIR / str(uuid.uuid4())
    out_dir.mkdir()
    sig = at.AudioSignal(_input_audio)
    sig = interface.preprocess(sig)

    z = interface.encode(sig)

    # build the mask
    mask = pmask.linear_random(z, 1.0)
    if _beat_mask_width > 0:
        beat_mask = interface.make_beat_mask(
            sig,
            after_beat_s=_token_width_to_seconds(_beat_mask_width),
        )
        mask = pmask.mask_and(mask, beat_mask)

    # save the mask as a txt file
    zv, mask_z = interface.coarse_vamp(
        z, 
        mask=mask,
        sampling_temperature=_sampletemp,
        return_mask=True, 
        gen_fn=interface.coarse.generate,
    )


    zv = interface.coarse_to_fine(
        zv, 
        sampling_temperature=_sampletemp,
        mask=mask,
    )

    sig = interface.to_signal(zv).cpu()
    print("done")

    sig.write(out_dir / "output.wav")

    return sig.path_to_file

with gr.Blocks() as demo:

    with gr.Row():
        with gr.Column():
            gr.Markdown("# VampNet Audio Vamping")
            with gr.Row():
                coarse_model_choice = gr.Dropdown(
                    label="Coarse model",
                    choices=model_dropdown_labels(COARSE_MODEL_ENTRIES),
                    value=model_dropdown_value(COARSE_MODEL_ENTRIES, ACTIVE_MODEL_IDS["coarse"]),
                    interactive=True,
                )
                c2f_model_choice = gr.Dropdown(
                    label="C2F model",
                    choices=model_dropdown_labels(C2F_MODEL_ENTRIES),
                    value=model_dropdown_value(C2F_MODEL_ENTRIES, ACTIVE_MODEL_IDS["c2f"]),
                    interactive=True,
                )
            model_selection_status = gr.Markdown(
                select_models(ACTIVE_MODEL_IDS["coarse"], ACTIVE_MODEL_IDS["c2f"])
            )
            gr.Markdown("""## Description:
            This is a demo of the VampNet, a generative audio model that transforms the input audio based on the chosen settings. 
            You can control the extent and nature of variation with a set of manual controls and presets. 
            Use this interface to experiment with different mask settings and explore the audio outputs.
            """)

            gr.Markdown("""
            ## Instructions:
            1. You can start by uploading some audio, or by loading the example audio. 
            2. Choose a preset for the vamp operation, or manually adjust the controls to customize the mask settings. 
            3. Click the "generate (vamp)!!!" button to apply the vamp operation. Listen to the output audio.
            4. Optionally, you can add some notes and save the result. 
            5. You can also use the output as the new input and continue experimenting!
            """)
    with gr.Row():
        with gr.Column():


            manual_audio_upload = gr.File(
                label="upload some audio (trimmed to the selected Coarse duration)",
                file_types=["audio"]
            )
            load_example_audio_button = gr.Button("or load example audio")

            input_audio = gr.Audio(
                label="input audio",
                interactive=False, 
                type="filepath",
            )

            audio_mask = gr.Audio(
                label="audio mask (listen to this to hear the mask hints)",
                interactive=False, 
                type="filepath",
            )

            # connect widgets
            load_example_audio_button.click(
                fn=load_example_audio,
                inputs=[],
                outputs=[ input_audio]
            )

            manual_audio_upload.change(
                fn=load_audio,
                inputs=[manual_audio_upload],
                outputs=[ input_audio]
            )
                
        # mask settings
        with gr.Column():


            presets = {
                    "unconditional": {
                        "periodic_p": 0,
                        "onset_mask_width": 0,
                        "beat_mask_width": 0,
                        "beat_mask_downbeats": False,
                    }, 
                    "slight periodic variation": {
                        "periodic_p": 5,
                        "onset_mask_width": 5,
                        "beat_mask_width": 0,
                        "beat_mask_downbeats": False,
                    },
                    "moderate periodic variation": {
                        "periodic_p": 13,
                        "onset_mask_width": 5,
                        "beat_mask_width": 0,
                        "beat_mask_downbeats": False,
                    },
                    "strong periodic variation": {
                        "periodic_p": 17,
                        "onset_mask_width": 5,
                        "beat_mask_width": 0,
                        "beat_mask_downbeats": False,
                    },
                    "very strong periodic variation": {
                        "periodic_p": 21,
                        "onset_mask_width": 5,
                        "beat_mask_width": 0,
                        "beat_mask_downbeats": False,
                    },
                    "beat-driven variation": {
                        "periodic_p": 0,
                        "onset_mask_width": 0,
                        "beat_mask_width": 3,
                        "beat_mask_downbeats": False,
                    },
                    "beat-driven variation (downbeats only)": {
                        "periodic_p": 0,
                        "onset_mask_width": 0,
                        "beat_mask_width": 3,
                        "beat_mask_downbeats": True,
                    },
                    "beat-driven variation (downbeats only, strong)": {
                        "periodic_p": 0,
                        "onset_mask_width": 0,
                        "beat_mask_width": 1,
                        "beat_mask_downbeats": True,
                    },
                }

            preset = gr.Dropdown(
                label="preset", 
                choices=list(presets.keys()),
                value="strong periodic variation",
            )
            load_preset_button = gr.Button("load_preset")

            with gr.Accordion("manual controls", open=True):
                periodic_p = gr.Slider(
                    label="periodic prompt  (0 - unconditional, 2 - lots of hints, 8 - a couple of hints, 16 - occasional hint, 32 - very occasional hint, etc)",
                    minimum=0,
                    maximum=128, 
                    step=1,
                    value=3, 
                )


                onset_mask_width = gr.Slider(
                    label="onset mask width (multiplies with the periodic mask, 1 step ~= 10milliseconds) ",
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=5,
                )

                beat_mask_width = gr.Slider(
                    label="beat prompt width (tokens, 1 token ~= 17.4 ms)",
                    minimum=0,
                    maximum=12,
                    step=1,
                    value=0,
                )
                beat_mask_downbeats = gr.Checkbox(
                    label="beat mask downbeats only?", 
                    value=False
                )

                n_mask_codebooks = gr.Number(
                    label="first upper codebook level to mask",
                    value=9,
                )


                with gr.Accordion("extras ", open=False):
                    pitch_shift_amt = gr.Slider(
                        label="pitch shift amount (semitones)",
                        minimum=-12,
                        maximum=12,
                        step=1,
                        value=0,
                    )

                    rand_mask_intensity = gr.Slider(
                        label="random mask intensity. (If this is less than 1, scatters prompts throughout the audio, should be between 0.9 and 1.0)",
                        minimum=0.0,
                        maximum=1.0,
                        value=1.0
                    )

                    periodic_w = gr.Slider(
                        label="periodic prompt width (steps, 1 step ~= 10milliseconds)",
                        minimum=1,
                        maximum=20,
                        step=1,
                        value=1,
                    )
                    n_conditioning_codebooks = gr.Number(
                        label="number of conditioning codebooks. probably 0", 
                        value=0,
                        precision=0,
                    )

                    stretch_factor = gr.Slider(
                        label="time stretch factor",
                        minimum=0,
                        maximum=64, 
                        step=1,
                        value=1, 
                    )

            preset_outputs = {
                periodic_p, 
                onset_mask_width, 
                beat_mask_width,
                beat_mask_downbeats,
            }

            def load_preset(_preset):
                return tuple(presets[_preset].values())

            load_preset_button.click(
                fn=load_preset,
                inputs=[preset],
                outputs=preset_outputs
            )


            with gr.Accordion("prefix/suffix prompts", open=False):
                prefix_s = gr.Slider(
                    label="prefix hint length (seconds)",
                    minimum=0.0,
                    maximum=10.0,
                    value=0.0
                )
                suffix_s = gr.Slider(
                    label="suffix hint length (seconds)",
                    minimum=0.0,
                    maximum=10.0,
                    value=0.0
                )

            masktemp = gr.Slider(
                label="mask temperature",
                minimum=0.0,
                maximum=100.0,
                value=1.5
            )
            sampletemp = gr.Slider(
                label="sample temperature",
                minimum=0.1,
                maximum=10.0,
                value=1.0, 
                step=0.001
            )
        


            with gr.Accordion("sampling settings", open=False):
                top_p = gr.Slider(
                    label="top p (0.0 = off)",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0
                )
                typical_filtering = gr.Checkbox(
                    label="typical filtering ",
                    value=False
                )
                typical_mass = gr.Slider( 
                    label="typical mass (should probably stay between 0.1 and 0.5)",
                    minimum=0.01,
                    maximum=0.99,
                    value=0.15
                )
                typical_min_tokens = gr.Slider(
                    label="typical min tokens (should probably stay between 1 and 256)",
                    minimum=1,
                    maximum=256,
                    step=1,
                    value=64
                )
                sample_cutoff = gr.Slider(
                    label="sample cutoff (%)",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5, 
                    step=0.01
                )
                fixed_sample_cutoff = gr.Checkbox(
                    label="fixed cutoff",
                    value=False
                )
                sample_cutoff_steps = gr.Slider(
                    label="sample cutoff steps",
                    minimum=0,
                    maximum=128,
                    step=1,
                    value=0
                )

            use_coarse2fine = gr.Checkbox(
                label="use coarse2fine",
                value=True, 
                visible=False
            )

            num_steps = gr.Slider(
                label="number of steps (should normally be between 12 and 36)",
                minimum=1,
                maximum=128,
                step=1,
                value=36
            )

            dropout = gr.Slider(
                label="mask dropout",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.0
            )


            seed = gr.Number(
                label="seed (0 for random)",
                value=0,
                precision=0,
            )

            batch_variant = gr.Slider(
                label="batch variant (generate N, keep last)",
                minimum=1,
                maximum=8,
                step=1,
                value=1,
            )



        # mask settings
        with gr.Column():

            # lora_choice = gr.Dropdown(
            #     label="lora choice", 
            #     choices=list(loras.keys()),
            #     value=LORA_NONE, 
            #     visible=False
            # )

            vamp_button = gr.Button("generate (vamp)!!!")
            output_audio = gr.Audio(
                label="output audio",
                interactive=False,
                type="filepath"
            )
            api_variant_archive = gr.File(
                label="api variant archive",
                interactive=False,
                visible=False,
            )
            manual_hint_edits = gr.Textbox(
                label="manual hint edits",
                value="",
                visible=False,
            )

            notes_text = gr.Textbox(
                label="type any notes about the generated audio here", 
                value="",
                interactive=True
            )
            save_button = gr.Button("save vamp")
            download_file = gr.File(
                label="vamp to download will appear here",
                interactive=False
            )
            use_as_input_button = gr.Button("use output as input")
            
            thank_you = gr.Markdown("")


    _inputs = [
            input_audio, 
            periodic_p,
            onset_mask_width,
            beat_mask_width,
            beat_mask_downbeats,
            n_mask_codebooks,
            pitch_shift_amt,
            rand_mask_intensity, 
            periodic_w,
            n_conditioning_codebooks, 
            stretch_factor, 
            prefix_s,
            suffix_s,
            masktemp,
            sampletemp,
            top_p,
            typical_filtering,
            typical_mass,
            typical_min_tokens,
            sample_cutoff,
            fixed_sample_cutoff,
            sample_cutoff_steps,
            use_coarse2fine,
            num_steps,
            dropout,
            seed, 
            batch_variant,
            manual_hint_edits,
        ]

    def as_data(components, values):
        return dict(zip(components, values))

    def vamp_event(*values):
        return _vamp(as_data(_inputs, values), return_mask=True)

    def api_vamp_event(*values):
        return _vamp(
            as_data(_inputs, values),
            return_mask=False,
            return_variant_archive=True,
        )

    def save_vamp_event(*values):
        components = _inputs + [notes_text, output_audio]
        return save_vamp(as_data(components, values))

    def select_models_event(coarse_id, c2f_id):
        coarse_entry = resolve_model_entry(COARSE_MODEL_ENTRIES, coarse_id)
        c2f_entry = resolve_model_entry(C2F_MODEL_ENTRIES, c2f_id)
        if coarse_entry is None:
            coarse_entry = COARSE_MODEL_ENTRIES.get(ACTIVE_MODEL_IDS["coarse"])
        if c2f_entry is None:
            c2f_entry = C2F_MODEL_ENTRIES.get(ACTIVE_MODEL_IDS["c2f"])
        if coarse_entry is None or c2f_entry is None:
            raise ValueError("No valid Coarse/C2F model pair is available")
        status = select_models(coarse_entry.id, c2f_entry.id)
        return (
            gr.update(value=coarse_entry.label),
            gr.update(value=c2f_entry.label),
            status,
        )

    # connect widgets
    coarse_model_choice.change(
        fn=select_models_event,
        inputs=[coarse_model_choice, c2f_model_choice],
        outputs=[coarse_model_choice, c2f_model_choice, model_selection_status],
    )
    c2f_model_choice.change(
        fn=select_models_event,
        inputs=[coarse_model_choice, c2f_model_choice],
        outputs=[coarse_model_choice, c2f_model_choice, model_selection_status],
    )

    vamp_button.click(
        fn=vamp_event,
        inputs=_inputs,
        outputs=[output_audio, audio_mask],
    )

    api_vamp_button = gr.Button("api vamp", visible=True)
    api_vamp_button.click(
        fn=api_vamp_event,
        inputs=_inputs,
        outputs=[output_audio, api_variant_archive],
        api_name="vamp"
    )

    api_model_info = gr.Textbox(visible=False)
    api_model_info_button = gr.Button("api model info", visible=False)
    api_model_info_button.click(
        fn=api_model_info_event,
        inputs=[],
        outputs=[api_model_info],
        api_name="model_info",
    )

    api_hint_preview = gr.Textbox(visible=False)
    api_hint_preview_button = gr.Button("api hint preview", visible=False)
    api_hint_preview_button.click(
        fn=api_hint_preview_event,
        inputs=_inputs,
        outputs=[api_hint_preview],
        api_name="hint_preview",
    )

    use_as_input_button.click(
        fn=lambda x: x,
        inputs=[output_audio],
        outputs=[input_audio]
    )

    save_button.click(
        fn=save_vamp_event,
        inputs=_inputs + [notes_text, output_audio],
        outputs=[thank_you, download_file]
    )


demo.launch(share=True, debug=True)
#swap comment to enable public gradio + LAN
#demo.launch(server_name='0.0.0.0', server_port=7860, share=True, debug=True)
demo.queue()
