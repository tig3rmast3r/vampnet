from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import argbind
import audiotools as at
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

import scripts.exp.train as train_mod
from lac.model.lac import LAC as DAC
from vampnet.modules.transformer import VampNet


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _mean_metrics(rows: list[dict]) -> dict[str, float]:
    if not rows:
        raise ValueError("No validation rows were produced")
    keys = sorted(rows[0].keys())
    return {
        key: float(np.mean([_to_float(row[key]) for row in rows]))
        for key in keys
    }


def _write_tensorboard(log_dir: Path, metrics: dict[str, float], step: int) -> None:
    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        for key, value in sorted(metrics.items()):
            writer.add_scalar(f"{key}/val", value, global_step=step)
    finally:
        writer.close()


def _copy_follow_args(args: dict, run_dir: Path, checkpoint: Path, device: str) -> dict:
    out = dict(args)
    out["save_path"] = str(run_dir)
    out["resume"] = False
    out["lh"] = False
    out["scheduler"] = "noam"
    out["nocompile"] = True
    out["validated_checkpoint"] = str(checkpoint)
    out["validation_device"] = device
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run train.py-compatible validation for one VampNet checkpoint."
    )
    parser.add_argument("--config", required=True, help="Reference training config to copy validation settings from.")
    parser.add_argument("--checkpoint", required=True, help="VampNet .pth checkpoint to validate.")
    parser.add_argument("--run-dir", required=True, help="Output TensorBoard/reference run directory.")
    parser.add_argument("--device", default="cuda", help="Torch device, usually cuda or cpu.")
    parser.add_argument("--step", type=int, default=0, help="TensorBoard global step for the reference metrics.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override validation batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override validation dataloader workers.")
    parser.add_argument("--val-examples", type=int, default=None, help="Override configured validation example count.")
    parsed = parser.parse_args()

    config = Path(parsed.config)
    checkpoint = Path(parsed.checkpoint)
    run_dir = Path(parsed.run_dir)
    log_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)

    args = argbind.load_args(str(config))
    args["args.debug"] = True
    val_examples = int(
        parsed.val_examples
        if parsed.val_examples is not None
        else args.get("val/AudioDataset.n_examples", args.get("val/TokenDataset.n_examples", 0)) or 0
    )
    eval_args = _copy_follow_args(args, run_dir, checkpoint, parsed.device)
    eval_args["val/AudioDataset.n_examples"] = val_examples
    eval_args["val/TokenDataset.n_examples"] = val_examples
    argbind.dump_args(eval_args, str(run_dir / "args.yml"))

    at.util.seed(int(args.get("seed", 0)))
    device = torch.device(parsed.device)
    codec = DAC.load(args["codec_ckpt"], map_location="cpu").to(device).eval()
    model = VampNet.load(location=checkpoint, map_location="cpu", strict=False)
    train_mod._set_all_dropouts(model, 0.0)
    model = model.to(device).eval()

    if str(args.get("data_mode", "audio")).lower() != "tokenized":
        raise ValueError("This utility currently expects tokenized validation, matching the follow run.")

    chunk_frames = train_mod._token_chunk_frames(args, codec)
    val_seed = int(args.get("val_seed", args.get("seed", 0) + 1001))
    val_data = train_mod._make_token_dataset(
        args=args,
        split="val",
        shuffle_seed=val_seed,
        chunk_frames=chunk_frames,
        n_codebooks=int(model.n_codebooks),
        n_examples=val_examples,
    )

    batch_size = int(parsed.batch_size or args.get("batch_size", 1))
    num_workers = int(parsed.num_workers if parsed.num_workers is not None else args.get("num_workers", 0))
    loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=val_data.collate,
        persistent_workers=num_workers > 0,
    )

    state = SimpleNamespace(
        model=model,
        codec=codec,
        criterion=nn.CrossEntropyLoss(label_smoothing=float(args.get("CrossEntropyLoss.label_smoothing", 0.0))),
        val_rng=torch.quasirandom.SobolEngine(1, scramble=True, seed=val_seed),
        val_seed=val_seed,
        data_mode="tokenized",
        val_data=val_data,
    )
    accel = SimpleNamespace(device=device, unwrap=lambda module: module, autocast=lambda *a, **k: torch.cuda.amp.autocast(enabled=False))

    rows: list[dict] = []
    state.val_rng = torch.quasirandom.SobolEngine(1, scramble=True, seed=state.val_seed)
    with train_mod._fork_seed(state.val_seed):
        for batch in tqdm(loader, desc="validate"):
            with torch.no_grad():
                rows.append(train_mod.val_loop(state, batch, accel))

    metrics = _mean_metrics(rows)
    _write_tensorboard(log_dir, metrics, parsed.step)

    manifest = {
        "config": str(config),
        "checkpoint": str(checkpoint),
        "run_dir": str(run_dir),
        "device": str(device),
        "step": int(parsed.step),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "val_seed": val_seed,
        "val_examples_configured": val_examples,
        "val_batches": len(rows),
        "token_chunk_frames": chunk_frames,
        "metrics": metrics,
    }
    (run_dir / "validation_metrics.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
