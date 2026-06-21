import os
import sys
import warnings
import numbers
import math
import random
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, cast
from dataclasses import dataclass

import argbind
import audiotools as at
import numpy as np
import torch
import time
import torch.nn as nn
from audiotools import AudioSignal
from audiotools.data import transforms as tfm
from einops import rearrange
from rich import pretty
from rich.traceback import install
from torch.utils.tensorboard.writer import SummaryWriter

import vampnet
from vampnet.modules.transformer import VampNet
from vampnet.util import codebook_unflatten, codebook_flatten
from vampnet import mask as pmask
# from dac.model.dac import DAC
from lac.model.lac import LAC as DAC
from vampnet.scheduler import (
    NoamScheduler,
    RLROPScheduler,
    WarmupFlatCosineScheduler,
)

from audiotools.ml.decorators import (
    timer, Tracker, when
)

import loralib as lora

import torch._dynamo
# Guarded for type checkers and runtime compatibility across torch versions.
if hasattr(torch._dynamo, "config"):
    torch._dynamo.config.verbose = True  # type: ignore[attr-defined]

# Enable cudnn autotuner to speed up training
# (can be altered by the funcs.seed function)
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))
# Uncomment to trade memory for speed.

# Install to make things look nice
warnings.filterwarnings("ignore", category=UserWarning)
pretty.install()
install()

# optim
Accelerator = cast(Any, argbind.bind(at.ml.Accelerator, without_prefix=True))


@argbind.bind("CrossEntropyLoss")
def CrossEntropyLoss(
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
):
    return nn.CrossEntropyLoss(
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )


@argbind.bind("AdamW")
def AdamW(
    lr: float = 1e-3,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    amsgrad: bool = False,
    maximize: bool = False,
    capturable: bool = False,
    differentiable: bool = False,
    foreach=None,
    fused=None,
):
    kwargs = {
        "lr": lr,
        "betas": betas,
        "eps": eps,
        "weight_decay": weight_decay,
        "amsgrad": amsgrad,
        "maximize": maximize,
        "capturable": capturable,
        "differentiable": differentiable,
    }
    if foreach is not None:
        kwargs["foreach"] = foreach
    if fused is not None:
        kwargs["fused"] = fused
    return kwargs


NoamScheduler = cast(Any, argbind.bind(vampnet.scheduler.NoamScheduler))
RLROPScheduler = cast(Any, argbind.bind(vampnet.scheduler.RLROPScheduler))
WarmupFlatCosineScheduler = cast(
    Any, argbind.bind(vampnet.scheduler.WarmupFlatCosineScheduler)
)

# transforms
filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
    "BaseTransform",
    "Compose",
    "Choose",
]

# model
VampNet = cast(Any, argbind.bind(VampNet))

# data
AudioLoader = cast(Any, argbind.bind(at.datasets.AudioLoader))
AudioDataset = cast(Any, argbind.bind(at.datasets.AudioDataset, "train", "val"))

IGNORE_INDEX = -100
MAX_RANDOM_SEED = 2**32 - 1


def _new_seed() -> int:
    return int.from_bytes(os.urandom(8), "little") % MAX_RANDOM_SEED


def _derive_seed(base_seed: int, *parts: object) -> int:
    payload = ":".join([str(int(base_seed)), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % MAX_RANDOM_SEED


def _resolve_scheduler_type(args) -> str:
    scheduler_type = str(args.get("scheduler", "noam")).lower()
    if scheduler_type not in {"noam", "cosine", "rlrop"}:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    return scheduler_type


def _resolve_lh_reference_lr(args, scheduler_type: str, d_model: Optional[float] = None) -> float:
    if scheduler_type == "cosine":
        return float(
            args.get(
                "WarmupFlatCosineScheduler.base_lr",
                args.get("CosineScheduler.base_lr", args.get("AdamW.lr", 1e-3)),
            )
        )
    if scheduler_type == "noam":
        factor = float(args.get("NoamScheduler.factor", 1.0))
        warmup = max(float(args.get("NoamScheduler.warmup", 4000)), 1.0)
        model_dim = float(
            d_model
            if d_model is not None
            else args.get("VampNet.embedding_dim", args.get("NoamScheduler.d_model", 512))
        )
        return factor * (model_dim ** -0.5) * (warmup ** -0.5)
    if scheduler_type == "rlrop":
        return float(args.get("RLROPScheduler.lr", args.get("AdamW.lr", 1e-3)))
    return float(args.get("AdamW.lr", 1e-3))


def _apply_lh_weight_decay(state) -> None:
    if not state.lh:
        return

    lr_t = float(state.optimizer.param_groups[0].get("lr", 0.0))
    wd_t = state.lh_wd_start * state.lh_reference_lr / max(lr_t, 1e-12)
    wd_t = min(max(wd_t, state.lh_wd_start), state.lh_wd_end)

    for group in state.optimizer.param_groups:
        group["weight_decay"] = wd_t


@contextmanager
def _fork_seed(seed: int):
    """Seed temporary stochastic work without perturbing the training RNG stream."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        at.util.seed(int(seed) % MAX_RANDOM_SEED)
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


class EpochSeededAudioDataset:
    def __init__(self, dataset: AudioDataset, index_seed_offset: int):
        self.dataset = dataset
        self.index_seed_offset = int(index_seed_offset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        seeded_idx = (int(idx) + self.index_seed_offset) % MAX_RANDOM_SEED
        return self.dataset[seeded_idx]

    def __getattr__(self, name):
        return getattr(self.dataset, name)


class TokenizedCodeDataset:
    """Dataset of precomputed codec token files (*.tokens.npz).

    Each token file is expected to contain:
    - codes: uint16/int array of shape (n_codebooks, n_frames)
    - valid_starts (optional): int array of candidate chunk starts in token frames
    """

    def __init__(
        self,
        sources: list[str],
        chunk_frames: int,
        n_codebooks: int,
        n_examples: int = 0,
        shuffle_state: int = 0,
        without_replacement: bool = True,
    ):
        self.sources = [Path(s).expanduser().resolve() for s in sources]
        self.chunk_frames = int(chunk_frames)
        self.n_codebooks = int(n_codebooks)
        self.without_replacement = bool(without_replacement)
        self.entries = self._discover_entries(self.sources)
        if len(self.entries) == 0:
            raise ValueError(
                "No token files found. Expected *.tokens.npz under sources="
                f"{[str(s) for s in self.sources]}"
            )

        state = np.random.RandomState(int(shuffle_state) % MAX_RANDOM_SEED)
        state.shuffle(self.entries)
        self.length = int(n_examples) if int(n_examples) > 0 else len(self.entries)

    @staticmethod
    def _discover_entries(sources: list[Path]) -> list[Path]:
        entries: list[Path] = []
        for source in sources:
            if source.is_file() and source.suffix == ".npz":
                entries.append(source)
                continue
            if source.is_dir():
                entries.extend(sorted(source.rglob("*.tokens.npz")))
        return entries

    def __len__(self):
        return self.length

    def _sample_start(self, idx: int, n_frames: int, valid_starts: np.ndarray) -> int:
        max_start = max(int(n_frames) - self.chunk_frames, 0)
        if max_start <= 0:
            return 0
        state = np.random.RandomState(int(idx) % MAX_RANDOM_SEED)
        if valid_starts.size > 0:
            start = int(valid_starts[state.randint(0, valid_starts.size)])
            return max(0, min(start, max_start))
        return int(state.randint(0, max_start + 1))

    @staticmethod
    def _pad_chunk(chunk: np.ndarray, chunk_frames: int) -> np.ndarray:
        if chunk.shape[1] == chunk_frames:
            return chunk
        if chunk.shape[1] == 0:
            return np.zeros((chunk.shape[0], chunk_frames), dtype=np.int64)
        pad_n = chunk_frames - chunk.shape[1]
        pad = np.repeat(chunk[:, -1:], pad_n, axis=1)
        return np.concatenate([chunk, pad], axis=1)

    def __getitem__(self, idx):
        state = np.random.RandomState(int(idx) % MAX_RANDOM_SEED)
        if self.without_replacement:
            entry = self.entries[int(idx) % len(self.entries)]
        else:
            entry = self.entries[int(state.randint(0, len(self.entries)))]

        with np.load(entry, allow_pickle=False) as payload:
            codes = payload["codes"]
            if codes.ndim != 2:
                raise ValueError(f"Invalid code shape for {entry}: {codes.shape}")
            valid_starts = payload["valid_starts"] if "valid_starts" in payload.files else np.empty((0,), dtype=np.int64)

        if codes.shape[0] < self.n_codebooks:
            raise ValueError(
                f"Token file has fewer codebooks than required: {entry} "
                f"(found={codes.shape[0]}, required={self.n_codebooks})"
            )

        codes = codes[: self.n_codebooks, :]
        start = self._sample_start(idx=int(idx), n_frames=int(codes.shape[1]), valid_starts=np.asarray(valid_starts))
        end = start + self.chunk_frames
        chunk = codes[:, start:end].astype(np.int64, copy=False)
        chunk = self._pad_chunk(chunk, self.chunk_frames)

        return {
            "codes": torch.from_numpy(chunk).long(),
            "start_token": torch.tensor(start, dtype=torch.long),
            "idx": torch.tensor(int(idx), dtype=torch.long),
        }

    @staticmethod
    def collate(list_of_dicts: list[dict]):
        codes = torch.stack([item["codes"] for item in list_of_dicts], dim=0)
        start_token = torch.stack([item["start_token"] for item in list_of_dicts], dim=0)
        idx = torch.stack([item["idx"] for item in list_of_dicts], dim=0)
        return {
            "codes": codes,
            "start_token": start_token,
            "idx": idx,
        }


class EpochSeededTokenDataset:
    def __init__(self, dataset: TokenizedCodeDataset, index_seed_offset: int):
        self.dataset = dataset
        self.index_seed_offset = int(index_seed_offset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        seeded_idx = (int(idx) + self.index_seed_offset) % MAX_RANDOM_SEED
        return self.dataset[seeded_idx]

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def _audio_item_count(dataset: AudioDataset) -> int:
    return sum(len(loader.audio_indices) for loader in dataset.loaders.values())


def _make_train_dataset(args, sample_rate: int, run_seed: int, epoch: int, epoch_examples: int):
    shuffle_seed = _derive_seed(run_seed, "train-shuffle", epoch)
    excerpt_seed = _derive_seed(run_seed, "train-excerpt", epoch)
    max_offset = max(MAX_RANDOM_SEED - max(int(epoch_examples), 1) - 1, 1)
    index_seed_offset = excerpt_seed % max_offset

    with argbind.scope(args, "train"):
        dataset = AudioDataset(
            AudioLoader(shuffle_state=shuffle_seed),
            sample_rate,
            transform=build_transform(),
        )
    if epoch_examples > 0:
        dataset.length = int(epoch_examples)
    return EpochSeededAudioDataset(dataset, index_seed_offset)


def _token_item_count(dataset: TokenizedCodeDataset) -> int:
    return len(dataset.entries)


def _dataset_item_count(dataset: Any) -> int:
    if hasattr(dataset, "loaders"):
        return _audio_item_count(dataset)
    if hasattr(dataset, "entries"):
        return _token_item_count(dataset)
    return len(dataset)


def _resolve_data_mode(args) -> str:
    mode = str(args.get("data_mode", "audio")).lower().strip()
    if mode not in {"audio", "tokenized"}:
        raise ValueError(f"Unsupported data_mode={mode}. Expected one of: audio, tokenized")
    return mode


def _token_chunk_frames(args, codec: DAC) -> int:
    duration_s = float(args.get("AudioDataset.duration", 10.0))
    token_rate = float(codec.sample_rate) / float(codec.hop_length)
    return max(int(round(duration_s * token_rate)), 1)


def _make_token_dataset(
    args,
    split: str,
    shuffle_seed: int,
    chunk_frames: int,
    n_codebooks: int,
    n_examples: int = 0,
):
    key = f"{split}/TokenDataset.sources"
    sources = args.get(key, None)
    if sources is None:
        raise ValueError(f"Missing {key} for tokenized mode")
    if isinstance(sources, str):
        sources = [sources]

    without_replacement = bool(args.get("TokenDataset.without_replacement", True))
    return TokenizedCodeDataset(
        sources=list(sources),
        chunk_frames=chunk_frames,
        n_codebooks=n_codebooks,
        n_examples=int(n_examples),
        shuffle_state=int(shuffle_seed),
        without_replacement=without_replacement,
    )


def _make_train_token_dataset(
    args,
    run_seed: int,
    epoch: int,
    epoch_examples: int,
    chunk_frames: int,
    n_codebooks: int,
):
    shuffle_seed = _derive_seed(run_seed, "train-shuffle", epoch)
    excerpt_seed = _derive_seed(run_seed, "train-excerpt", epoch)
    max_offset = max(MAX_RANDOM_SEED - max(int(epoch_examples), 1) - 1, 1)
    index_seed_offset = excerpt_seed % max_offset
    dataset = _make_token_dataset(
        args=args,
        split="train",
        shuffle_seed=shuffle_seed,
        chunk_frames=chunk_frames,
        n_codebooks=n_codebooks,
        n_examples=int(epoch_examples),
    )
    return EpochSeededTokenDataset(dataset, index_seed_offset)


def _extract_codes(state: "State", batch: dict, accel: Accelerator) -> torch.Tensor:
    vn = accel.unwrap(state.model)
    if state.data_mode == "tokenized":
        z = batch["codes"].to(accel.device, dtype=torch.long)
        return z[:, : vn.n_codebooks, :]

    signal = apply_transform(state.train_data.transform, batch)
    with torch.inference_mode():
        state.codec.to(accel.device)
        z = state.codec.encode(signal.samples, signal.sample_rate)["codes"]
    return z[:, : vn.n_codebooks, :]


def _set_all_dropouts(module: nn.Module, p: float) -> None:
    """Force dropout probability across known dropout fields.

    This is used to guarantee no-dropout training without maintaining
    alternate transformer source files.
    """
    p = float(p)
    for submodule in module.modules():
        if isinstance(submodule, nn.Dropout):
            submodule.p = p
            continue

        # Some modules (e.g. attention implementations) store dropout as a scalar.
        for attr_name in ("dropout", "dropout_p", "attn_dropout", "p_dropout"):
            if not hasattr(submodule, attr_name):
                continue
            value = getattr(submodule, attr_name)
            if isinstance(value, nn.Dropout):
                value.p = p
            elif isinstance(value, numbers.Real):
                setattr(submodule, attr_name, p)


@argbind.bind("train", "val", without_prefix=True)
def build_transform():
    transform = tfm.Compose(
        [
            tfm.VolumeNorm(("const", -24)),
            # tfm.PitchShift(),
            tfm.RescaleAudio(),
        ]
    )
    return transform


@torch.no_grad()
def apply_transform(transform_fn, batch):
    sig: AudioSignal = batch["signal"]
    kwargs = batch["transform_args"]

    sig: AudioSignal = transform_fn(sig.clone(), **kwargs)
    return sig


def build_datasets(args, sample_rate: int):
    with argbind.scope(args, "train"):
        train_data = AudioDataset(
            AudioLoader(), sample_rate, transform=build_transform()
        )
    with argbind.scope(args, "val"):
        val_data = AudioDataset(AudioLoader(), sample_rate, transform=build_transform())
    return train_data, val_data


def rand_float(shape, low, high, rng):
    return rng.draw(shape)[:, 0] * (high - low) + low


def flip_coin(shape, p, rng):
    return rng.draw(shape)[:, 0] < p


def num_params_hook(o, p):
    return o + f" {p/1e6:<.3f}M params."


def add_num_params_repr_hook(model):
    import numpy as np
    from functools import partial

    for n, m in model.named_modules():
        o = m.extra_repr()
        p = sum([np.prod(p.size()) for p in m.parameters()])

        setattr(m, "extra_repr", partial(num_params_hook, o=o, p=p))


def accuracy(
    preds: torch.Tensor,
    target: torch.Tensor,
    top_k: int = 1,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    # Flatten the predictions and targets to be of shape (batch_size * sequence_length, n_class)
    preds = rearrange(preds, "b p s -> (b s) p")
    target = rearrange(target, "b s -> (b s)")

    # return torchmetrics.functional.accuracy(preds, target, task='multiclass', top_k=topk, num_classes=preds.shape[-1], ignore_index=ignore_index)
    if ignore_index is not None:
        # Create a mask for the ignored index
        mask = target != ignore_index
        # Apply the mask to the target and predictions
        preds = preds[mask]
        target = target[mask]

    # Get the top-k predicted classes and their indices
    _, pred_indices = torch.topk(preds, k=top_k, dim=-1)

    # Determine if the true target is in the top-k predicted classes
    correct = torch.sum(torch.eq(pred_indices, target.unsqueeze(1)), dim=1)

    # Calculate the accuracy
    accuracy = torch.mean(correct.float())

    return accuracy

def _metrics(z_hat, r, target, flat_mask, output):
    for r_range in [(0, 0.5), (0.5, 1.0)]:
        unmasked_target = target.masked_fill(flat_mask.bool(), IGNORE_INDEX)
        masked_target = target.masked_fill(~flat_mask.bool(), IGNORE_INDEX)

        assert target.shape[0] == r.shape[0]
        # grab the indices of the r values that are in the range
        r_idx = (r >= r_range[0]) & (r < r_range[1])

        # grab the target and z_hat values that are in the range
        r_unmasked_target = unmasked_target[r_idx]
        r_masked_target = masked_target[r_idx]
        r_z_hat = z_hat[r_idx]

        for topk in (1, 25):
            s, e = r_range
            tag = f"accuracy-{s}-{e}/top{topk}"

            output[f"{tag}/unmasked"] = accuracy(
                preds=r_z_hat,
                target=r_unmasked_target,
                ignore_index=IGNORE_INDEX,
                top_k=topk,
            )
            output[f"{tag}/masked"] = accuracy(
                preds=r_z_hat,
                target=r_masked_target,
                ignore_index=IGNORE_INDEX,
                top_k=topk,
            )


@dataclass
class State:
    model: VampNet
    codec: DAC

    optimizer: AdamW
    scheduler: Any
    scheduler_type: str
    criterion: CrossEntropyLoss
    grad_clip_val: float

    train_data: Any
    val_data: Any
    sample_data: Any
    val_rng: torch.quasirandom.SobolEngine

    tracker: Tracker
    run_seed: int
    val_seed: int
    sample_seed: int
    epoch_examples: int
    epoch_steps: int
    data_mode: str
    token_chunk_frames: int
    lh: bool
    lh_reference_lr: float
    lh_wd_start: float
    lh_wd_end: float
    latest_grad_norm: Optional[float] = None  # Store the latest grad norm

@timer()
def train_loop(state: State, batch: dict, accel: Accelerator):
    state.model.train()
    batch = at.util.prepare_batch(batch, accel.device)

    output = {}
    vn = accel.unwrap(state.model)
    with accel.autocast():
        if state.data_mode == "tokenized":
            z = batch["codes"].to(accel.device, dtype=torch.long)
            z = z[:, : vn.n_codebooks, :]
        else:
            signal = apply_transform(state.train_data.transform, batch)
            with torch.inference_mode():
                state.codec.to(accel.device)
                z = state.codec.encode(signal.samples, signal.sample_rate)["codes"]
                z = z[:, : vn.n_codebooks, :]

        n_batch = z.shape[0]
        step_seed = _derive_seed(state.run_seed, "train-mask", state.tracker.step)
        with _fork_seed(step_seed):
            step_rng = torch.quasirandom.SobolEngine(
                1, scramble=True, seed=step_seed
            )
            r = step_rng.draw(n_batch)[:, 0].to(accel.device)
            mask = pmask.random(z, r)
        mask = pmask.codebook_unmask(mask, vn.n_conditioning_codebooks)
        z_mask, mask = pmask.apply_mask(z, mask, vn.mask_token)

        z_mask_latent = vn.embedding.from_codes(z_mask, state.codec)

        dtype = torch.bfloat16 if accel.amp else None
        with accel.autocast(dtype=dtype):
            z_hat = state.model(z_mask_latent)

        target = codebook_flatten(
            z[:, vn.n_conditioning_codebooks :, :],
        )

        flat_mask = codebook_flatten(
            mask[:, vn.n_conditioning_codebooks :, :],
        )

        # replace target with ignore index for masked tokens
        t_masked = target.masked_fill(~flat_mask.bool(), IGNORE_INDEX)
        output["loss"] = state.criterion(z_hat, t_masked)

        _metrics(
            r=r,
            z_hat=z_hat,
            target=target,
            flat_mask=flat_mask,
            output=output,
        )


    accel.backward(output["loss"])

    output["other/learning_rate"] = state.optimizer.param_groups[0]["lr"]
    output["other/batch_size"] = z.shape[0]


    accel.scaler.unscale_(state.optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(state.model.parameters(), max_norm=state.grad_clip_val) # outputs clipped

    #this is for custom rlrop when used in "rise" mode (factor>1), usefful when used with flash_attn v2 that is very prone to gradient explosion
    #will lower lr immediately after 10 grad explosions
    if state.scheduler_type == "rlrop" and state.scheduler.factor > 1:
        if grad_norm > state.grad_clip_val:
            if not hasattr(state, 'grad_exceed_count'):
                state.grad_exceed_count = 0
            state.grad_exceed_count += 1
            print(f"Gradient norm {grad_norm} exceeded clip value {state.grad_clip_val} and was clipped. Count: {state.grad_exceed_count}")

            # Check if the count has reached 10
            if state.grad_exceed_count >= 10:
                new_factor = 2.0 - state.scheduler.factor

                for param_group in state.optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] * new_factor, state.scheduler.min_lrs[0])
                state.grad_exceed_count = 0  # Reset the count
                print(f"Learning rate reduced by factor {new_factor} due to excessive grad norm.")

        if not hasattr(state, 'grad_norms'):
            state.grad_norms = []
        state.grad_norms.append(grad_norm.item())

    _apply_lh_weight_decay(state)

    output["other/grad_norm"] = grad_norm
    output["other/weight_decay"] = state.optimizer.param_groups[0]["weight_decay"]
    accel.step(state.optimizer)
    state.optimizer.zero_grad()

    # Step deterministic per-iteration schedulers here; RLROP is stepped on validation.
    if state.scheduler_type in {"noam", "cosine"}:
        state.scheduler.step()
    elif state.scheduler_type == "rlrop" and hasattr(state.scheduler, "step_train"):
        state.scheduler.step_train()

    accel.update()
    return {k: v for k, v in sorted(output.items())}


@timer()
@torch.no_grad()
def val_loop(state: State, batch: dict, accel: Accelerator):
    state.model.eval()
    state.codec.eval()
    batch = at.util.prepare_batch(batch, accel.device)
    vn = accel.unwrap(state.model)
    if state.data_mode == "tokenized":
        z = batch["codes"].to(accel.device, dtype=torch.long)
        z = z[:, : vn.n_codebooks, :]
    else:
        signal = apply_transform(state.val_data.transform, batch)
        z = state.codec.encode(signal.samples, signal.sample_rate)["codes"]
        z = z[:, : vn.n_codebooks, :]

    n_batch = z.shape[0]
    r = state.val_rng.draw(n_batch)[:, 0].to(accel.device)
    mask = pmask.random(z, r)
    mask = pmask.codebook_unmask(mask, vn.n_conditioning_codebooks)
    z_mask, mask = pmask.apply_mask(z, mask, vn.mask_token)

    z_mask_latent = vn.embedding.from_codes(z_mask, state.codec)

    z_hat = state.model(z_mask_latent)

    target = codebook_flatten(
        z[:, vn.n_conditioning_codebooks :, :],
    )

    flat_mask = codebook_flatten(
        mask[:, vn.n_conditioning_codebooks :, :]
    )

    output = {}
    # replace target with ignore index for masked tokens
    t_masked = target.masked_fill(~flat_mask.bool(), IGNORE_INDEX)
    output["loss"] = state.criterion(z_hat, t_masked)

    _metrics(
        r=r,
        z_hat=z_hat,
        target=target,
        flat_mask=flat_mask,
        output=output,
    )

    return output

def validate(state, val_dataloader, accel, include_grad_norm: bool = False):
    val_losses = []
    state.val_rng = torch.quasirandom.SobolEngine(
        1, scramble=True, seed=state.val_seed
    )
    with _fork_seed(state.val_seed):
        for batch in val_dataloader:
            output = val_loop(state, batch, accel)
            loss = output["loss"]
            if isinstance(loss, torch.Tensor):
                val_losses.append(loss.item())
            else:
                val_losses.append(loss)
    # Consolidate state dicts if using ZeroRedundancyOptimizer
    if hasattr(state.optimizer, "consolidate_state_dict"):
        state.optimizer.consolidate_state_dict()
    mean_val_loss = sum(val_losses) / len(val_losses)
    print(f"Mean Validation Loss: {mean_val_loss}")  # Print the mean validation loss
    mean_grad_norm = state.mean_grad_norm if hasattr(state, 'mean_grad_norm') else None
    print(f"Mean Grad Norm: {mean_grad_norm}")  # Print the mean grad norm
    if include_grad_norm:
        return {"loss": mean_val_loss, "grad_norm": mean_grad_norm}
    else:
        return {"loss": mean_val_loss}



def checkpoint(state, save_iters, save_path, fine_tune):
    if accel.local_rank != 0:
        state.tracker.print(f"ERROR:Skipping checkpoint on rank {accel.local_rank}")
        return

    metadata = {"logs": dict(state.tracker.history)}

    tags = ["latest"]
    state.tracker.print(f"Saving to {str(Path('.').absolute())}")

    if state.tracker.step in save_iters:
        tags.append(f"{state.tracker.step // 1000}k")

    if state.tracker.is_best("val", "loss"):
        state.tracker.print(f"Best model so far")
        tags.append("best")

    if fine_tune:
        for tag in tags:
            # save the lora model
            (Path(save_path) / tag).mkdir(parents=True, exist_ok=True)
            torch.save(
                lora.lora_state_dict(accel.unwrap(state.model)),
                f"{save_path}/{tag}/lora.pth"
            )

    for tag in tags:
        model_extra = {
            "optimizer.pth": state.optimizer.state_dict(),
            "scheduler.pth": state.scheduler.state_dict(),
            "tracker.pth": state.tracker.state_dict(),
            "metadata.pth": metadata,
            "training_state.pth": {
                "run_seed": state.run_seed,
                "val_seed": state.val_seed,
                "sample_seed": state.sample_seed,
                "scheduler_type": state.scheduler_type,
                "epoch_examples": state.epoch_examples,
                "epoch_steps": state.epoch_steps,
                "lh": state.lh,
                "lh_reference_lr": state.lh_reference_lr,
                "lh_wd_start": state.lh_wd_start,
                "lh_wd_end": state.lh_wd_end,
            },
        }

        accel.unwrap(state.model).metadata = metadata
        accel.unwrap(state.model).save_to_folder(
            f"{save_path}/{tag}", model_extra, package=False
        )

def save_sampled(state, z, writer):
    num_samples = z.shape[0]

    for i in range(num_samples):
        sampled = accel.unwrap(state.model).generate(
            codec=state.codec,
            time_steps=z.shape[-1],
            start_tokens=z[i : i + 1],
        )
        sampled.cpu().write_audio_to_tb(
            f"sampled/{i}",
            writer,
            step=state.tracker.step,
            plot_fn=None,
        )


def save_imputation(state, z, val_idx, writer):
    n_prefix = int(z.shape[-1] * 0.25)
    n_suffix = int(z.shape[-1] * 0.25)

    vn = accel.unwrap(state.model)

    mask = pmask.inpaint(z, n_prefix, n_suffix)
    mask = pmask.codebook_unmask(mask, vn.n_conditioning_codebooks)
    z_mask, mask = pmask.apply_mask(z, mask, vn.mask_token)

    imputed_noisy = vn.to_signal(z_mask, state.codec)
    imputed_true = vn.to_signal(z, state.codec)

    imputed = []
    for i in range(len(z)):
        imputed.append(
            vn.generate(
                codec=state.codec,
                time_steps=z.shape[-1],
                start_tokens=z[i][None, ...],
                mask=mask[i][None, ...],
            )
        )
    imputed = AudioSignal.batch(imputed)

    for i in range(len(val_idx)):
        imputed_noisy[i].cpu().write_audio_to_tb(
            f"inpainted_prompt/{i}",
            writer,
            step=state.tracker.step,
            plot_fn=None,
        )
        imputed[i].cpu().write_audio_to_tb(
            f"inpainted_middle/{i}",
            writer,
            step=state.tracker.step,
            plot_fn=None,
        )
        imputed_true[i].cpu().write_audio_to_tb(
            f"reconstructed/{i}",
            writer,
            step=state.tracker.step,
            plot_fn=None,
        )


@torch.no_grad()
def save_samples(state: State, val_idx: int, writer: SummaryWriter):
    state.model.eval()
    state.codec.eval()
    vn = accel.unwrap(state.model)

    sample_data = state.sample_data
    batch = [sample_data[i] for i in val_idx]
    batch = at.util.prepare_batch(sample_data.collate(batch), accel.device)

    if state.data_mode == "tokenized":
        z = batch["codes"].to(accel.device, dtype=torch.long)
        z = z[:, : vn.n_codebooks, :]
        signal = vn.to_signal(z, state.codec)
    else:
        signal = apply_transform(sample_data.transform, batch)
        z = state.codec.encode(signal.samples, signal.sample_rate)["codes"]
        z = z[:, : vn.n_codebooks, :]

    r = torch.linspace(0.1, 0.95, len(val_idx)).to(accel.device)


    mask = pmask.random(z, r)
    mask = pmask.codebook_unmask(mask, vn.n_conditioning_codebooks)
    z_mask, mask = pmask.apply_mask(z, mask, vn.mask_token)

    z_mask_latent = vn.embedding.from_codes(z_mask, state.codec)

    z_hat = state.model(z_mask_latent)

    z_pred = torch.softmax(z_hat, dim=1).argmax(dim=1)
    z_pred = codebook_unflatten(z_pred, n_c=vn.n_predict_codebooks)
    z_pred = torch.cat([z[:, : vn.n_conditioning_codebooks, :], z_pred], dim=1)

    generated = vn.to_signal(z_pred, state.codec)
    reconstructed = vn.to_signal(z, state.codec)
    masked = vn.to_signal(z_mask.squeeze(1), state.codec)

    for i in range(generated.batch_size):
        audio_dict = {
            "original": signal[i],
            "masked": masked[i],
            "generated": generated[i],
            "reconstructed": reconstructed[i],
        }
        for k, v in audio_dict.items():
            v.cpu().write_audio_to_tb(
                f"onestep/_{i}.r={r[i]:0.2f}/{k}",
                writer,
                step=state.tracker.step,
                plot_fn=None,
            )

    save_sampled(state=state, z=z, writer=writer)
    save_imputation(state=state, z=z, val_idx=val_idx, writer=writer)


@argbind.bind(without_prefix=True)
def load(
    args,
    accel: at.ml.Accelerator,
    tracker: Tracker,
    save_path: str,
    resume: bool = False,
    nocompile: bool = False,
    lh: bool = False,
    lh_wd_start: float = 0.0001,
    lh_wd_end: float = 1.0,
    tag: str = "latest",
    fine_tune_checkpoint: Optional[str] = None,
    grad_clip_val: float = 10.0, #increased from 5
) -> State:
    use_dropout = bool(args.get("dropout", False))
    scheduler_type = _resolve_scheduler_type(args)
    lh_wd_start = float(lh_wd_start)
    lh_wd_end = float(lh_wd_end)
    if lh:
        if lh_wd_start <= 0 or lh_wd_end <= 0:
            raise ValueError("lh_wd_start and lh_wd_end must be positive when lh is enabled")
        if lh_wd_end < lh_wd_start:
            raise ValueError("lh_wd_end must be greater than or equal to lh_wd_start")
        args["AdamW.weight_decay"] = lh_wd_start

    codec = DAC.load(args["codec_ckpt"], map_location="cpu")
    codec = codec.to(accel.device).eval()

    model, v_extra = None, {}

    if args["fine_tune"]:
        assert fine_tune_checkpoint is not None, "Must provide a fine-tune checkpoint"
        model = VampNet.load(location=Path(fine_tune_checkpoint), map_location="cpu")

    if resume:
        kwargs = {
            "folder": f"{save_path}/{tag}",
            "map_location": "cpu",
            "package": False,
        }
        tracker.print(f"Loading checkpoint from {kwargs['folder']}")
        if (Path(kwargs["folder"]) / "vampnet").exists():
            model, v_extra = VampNet.load_from_folder(**kwargs)
        else:
            raise ValueError(
                f"Could not find a VampNet checkpoint in {kwargs['folder']}"
            )

    if model is None:
        model = VampNet()

    # Default policy: dropout disabled unless explicitly requested via --dropout.
    configured_dropout = float(args.get("VampNet.dropout", 0.0))
    if use_dropout:
        # If user enables dropout but config is 0, use legacy dropout default.
        target_dropout = configured_dropout if configured_dropout > 0 else 0.1
        args["VampNet.dropout"] = target_dropout
        _set_all_dropouts(model, target_dropout)
        print(f"dropout enabled: forcing all dropout rates to {target_dropout}")
    else:
        args["VampNet.dropout"] = 0.0
        _set_all_dropouts(model, 0.0)
        print("dropout disabled (default): forcing all dropout rates to 0.0")

    if nocompile:
        print(f"torch.compile DISABLED")
    else:
        model = torch.compile(model)

    model = accel.prepare_model(model)

    # assert accel.unwrap(model).n_codebooks == codec.quantizer.n_codebooks
    assert (
        accel.unwrap(model).vocab_size == codec.quantizer.quantizers[0].codebook_size
    )

    if accel.world_size > 1:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            torch.optim.AdamW,
            **AdamW(),
        )
        print(f"OPTIMIZER LR is {optimizer.param_groups[0]['lr']}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), **AdamW())

    lh_reference_lr = _resolve_lh_reference_lr(
        args,
        scheduler_type=scheduler_type,
        d_model=float(accel.unwrap(model).embedding_dim),
    )

    if lh:
        print(
            "lh mode enabled: AdamW.weight_decay is ignored and dynamic WD is "
            f"computed from scheduler LR (reference_lr={lh_reference_lr}, "
            f"wd_start={lh_wd_start}, wd_end={lh_wd_end})."
        )

    # Scheduler Selection
    if scheduler_type == "rlrop":
        scheduler = RLROPScheduler(
            optimizer,
            lr=args.get("RLROPScheduler.lr", args.get("AdamW.lr", 1e-3)),
            factor=args.get("RLROPScheduler.factor", 0.5),
            patience=args.get("RLROPScheduler.patience", 10),
            min_lr=args.get("RLROPScheduler.min_lr", 1e-6),
            threshold=args.get("RLROPScheduler.threshold", 0.0005),
            threshold_mode=args.get("RLROPScheduler.threshold_mode", "rel"),
            warmup_steps=args.get("RLROPScheduler.warmup_steps", 0),
            eps=1e-8,
            mode='min',
        )
    elif scheduler_type == "cosine":
        scheduler = WarmupFlatCosineScheduler(
            optimizer,
            base_lr=args.get(
                "WarmupFlatCosineScheduler.base_lr",
                args.get("CosineScheduler.base_lr", args.get("AdamW.lr", 4e-4)),
            ),
            min_lr=args.get(
                "WarmupFlatCosineScheduler.min_lr",
                args.get("CosineScheduler.min_lr", 5e-5),
            ),
            warmup_steps=args.get(
                "WarmupFlatCosineScheduler.warmup_steps",
                args.get("CosineScheduler.warmup_steps", 1000),
            ),
            flat_steps=args.get(
                "WarmupFlatCosineScheduler.flat_steps",
                args.get("CosineScheduler.flat_steps", 0),
            ),
            total_steps=args.get(
                "WarmupFlatCosineScheduler.total_steps",
                args.get("CosineScheduler.total_steps", args.get("num_iters", 1)),
            ),
        )
    else:
        scheduler = NoamScheduler(
            optimizer, d_model=accel.unwrap(model).embedding_dim, factor=args["NoamScheduler.factor"], warmup=args["NoamScheduler.warmup"]
        )

    training_state = {}
    if resume:
        if "optimizer.pth" in v_extra:
            optimizer.load_state_dict(v_extra["optimizer.pth"])
            if lh:
                for group in optimizer.param_groups:
                    group["weight_decay"] = lh_wd_start
        if "scheduler.pth" in v_extra:
            scheduler.load_state_dict(v_extra["scheduler.pth"])
        if "tracker.pth" in v_extra:
            tracker.load_state_dict(v_extra["tracker.pth"])
        if "training_state.pth" in v_extra:
            training_state = v_extra["training_state.pth"]
    else:
        if scheduler_type == "rlrop" and getattr(scheduler, "warmup_steps", 0) <= 0:
            scheduler.step(metrics=0)
    if not resume and scheduler_type != "rlrop":
        scheduler.step()

    if lh:
        class _InitialLHState:
            pass

        initial_lh_state = _InitialLHState()
        initial_lh_state.lh = True
        initial_lh_state.optimizer = optimizer
        initial_lh_state.lh_reference_lr = lh_reference_lr
        initial_lh_state.lh_wd_start = lh_wd_start
        initial_lh_state.lh_wd_end = lh_wd_end
        _apply_lh_weight_decay(initial_lh_state)

    criterion = CrossEntropyLoss()

    sample_rate = codec.sample_rate
    data_mode = _resolve_data_mode(args)

    configured_run_seed = int(args.get("run_seed", -1))
    run_seed = int(training_state.get("run_seed", configured_run_seed))
    if run_seed < 0:
        run_seed = _new_seed()
    val_seed = int(training_state.get("val_seed", args.get("val_seed", args.get("seed", 0) + 1001)))
    sample_seed = int(training_state.get("sample_seed", args.get("sample_seed", args.get("seed", 0) + 2002)))

    # log a model summary w/ num params
    if accel.local_rank == 0:
        add_num_params_repr_hook(accel.unwrap(model))
        with open(f"{save_path}/model.txt", "w") as f:
            f.write(repr(accel.unwrap(model)))

    # load the datasets
    token_chunk_frames = _token_chunk_frames(args, codec) if data_mode == "tokenized" else 0
    if data_mode == "tokenized":
        count_data = _make_token_dataset(
            args=args,
            split="train",
            shuffle_seed=0,
            chunk_frames=token_chunk_frames,
            n_codebooks=int(accel.unwrap(model).n_codebooks),
            n_examples=0,
        )
        detected_train_examples = _token_item_count(count_data)
    else:
        with argbind.scope(args, "train"):
            count_data = AudioDataset(
                AudioLoader(shuffle_state=0),
                sample_rate,
                transform=build_transform(),
            )
        detected_train_examples = _audio_item_count(count_data)

    epoch_examples = int(args.get("train_epoch_examples", 0) or 0)
    if epoch_examples <= 0:
        epoch_examples = detected_train_examples
    batch_size = int(args.get("batch_size", 1))
    epoch_steps = max(math.ceil(epoch_examples / batch_size), 1)
    current_epoch = tracker.step // epoch_steps

    if data_mode == "tokenized":
        train_data = _make_train_token_dataset(
            args=args,
            run_seed=run_seed,
            epoch=current_epoch,
            epoch_examples=epoch_examples,
            chunk_frames=token_chunk_frames,
            n_codebooks=int(accel.unwrap(model).n_codebooks),
        )
        val_examples = int(args.get("val/AudioDataset.n_examples", args.get("val/TokenDataset.n_examples", 0)) or 0)
        val_data = _make_token_dataset(
            args=args,
            split="val",
            shuffle_seed=val_seed,
            chunk_frames=token_chunk_frames,
            n_codebooks=int(accel.unwrap(model).n_codebooks),
            n_examples=val_examples,
        )
        if args.get("sample/TokenDataset.sources", None) is not None:
            sample_examples = int(
                args.get(
                    "sample/AudioDataset.n_examples",
                    args.get("sample/TokenDataset.n_examples", 0),
                )
                or 0
            )
            sample_data = _make_token_dataset(
                args=args,
                split="sample",
                shuffle_seed=sample_seed,
                chunk_frames=token_chunk_frames,
                n_codebooks=int(accel.unwrap(model).n_codebooks),
                n_examples=sample_examples,
            )
        else:
            sample_data = val_data
    else:
        train_data = _make_train_dataset(
            args,
            sample_rate=sample_rate,
            run_seed=run_seed,
            epoch=current_epoch,
            epoch_examples=epoch_examples,
        )
        with argbind.scope(args, "val"):
            val_data = AudioDataset(
                AudioLoader(shuffle_state=val_seed),
                sample_rate,
                transform=build_transform(),
            )

        if args.get("sample/AudioLoader.sources", None) is not None:
            with argbind.scope(args, "sample"):
                sample_data = AudioDataset(
                    AudioLoader(shuffle_state=sample_seed),
                    sample_rate,
                    transform=build_transform(),
                )
        else:
            sample_data = val_data

    return State(
        tracker=tracker,
        model=model,
        codec=codec,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_type=scheduler_type,
        criterion=criterion,
        train_data=train_data,
        val_data=val_data,
        sample_data=sample_data,
        val_rng=torch.quasirandom.SobolEngine(1, scramble=True, seed=val_seed),
        grad_clip_val=grad_clip_val,
        run_seed=run_seed,
        val_seed=val_seed,
        sample_seed=sample_seed,
        epoch_examples=epoch_examples,
        epoch_steps=epoch_steps,
        data_mode=data_mode,
        token_chunk_frames=token_chunk_frames,
        lh=bool(lh),
        lh_reference_lr=lh_reference_lr,
        lh_wd_start=lh_wd_start,
        lh_wd_end=lh_wd_end,
    )




@argbind.bind(without_prefix=True)
def train(
    args,
    accel: at.ml.Accelerator,
    seed: int = 0,
    codec_ckpt: str = None,
    save_path: str = "ckpt",
    num_iters: int = int(1000e6),
    save_iters: list = [10000, 50000, 100000, 300000, 500000],
    sample_freq: int = 10000,
    val_freq: int = 1000,
    batch_size: int = 12,
    val_idx: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    num_workers: int = 10,
    fine_tune: bool = False,
    dropout: bool = False,
    data_mode: str = "audio",
    scheduler: str = "noam",
    run_seed: int = -1,
    val_seed: int = 1001,
    sample_seed: int = 2002,
    train_epoch_examples: int = 0,
    num_epochs: int = 0,
    save_epoch_freq: int = 0,
    sample_epoch_freq: int = 0,
    val_epoch_freq: int = 0,
):
    assert codec_ckpt is not None, "codec_ckpt is required"
    args["data_mode"] = data_mode

    seed = seed + accel.local_rank
    at.util.seed(seed)
    writer = None

    if accel.local_rank == 0:
        writer = SummaryWriter(log_dir=f"{save_path}/logs/")
        argbind.dump_args(args, f"{save_path}/args.yml")

    tracker = Tracker(
        writer=writer, log_file=f"{save_path}/log.txt", rank=accel.local_rank
    )

    # load the codec model
    state: State = load(
        args=args,
        accel=accel,
        tracker=tracker,
        save_path=save_path,
        resume=args.get("resume", False),
        nocompile=args.get("nocompile", False),
        lh=args.get("lh", False),
        lh_wd_start=args.get("lh_wd_start", 0.0001),
        lh_wd_end=args.get("lh_wd_end", 1.0),
    )
    print("initialized state.")

    sample_sources_key = (
        "sample/TokenDataset.sources"
        if state.data_mode == "tokenized"
        else "sample/AudioLoader.sources"
    )
    if args.get(sample_sources_key, None) is not None:
        sample_count = _dataset_item_count(state.sample_data)
        max_sample_idx = max([int(index) for index in val_idx], default=-1)
        if sample_count <= max_sample_idx:
            raise ValueError(
                "sample validation dataset does not contain enough items: "
                f"count={sample_count}, required_index={max_sample_idx}, "
                f"sources={args.get(sample_sources_key)}"
            )
        if accel.local_rank == 0 and sample_count != len(val_idx):
            print(
                "WARNING: sample validation dataset item count differs from val_idx count "
                f"(sample_items={sample_count}, val_idx={len(val_idx)}). "
                "Only the configured val_idx entries will be sampled."
            )

    if num_epochs > 0:
        num_iters = state.epoch_steps * num_epochs
        args["num_iters"] = num_iters
    if save_epoch_freq > 0 and num_epochs > 0:
        save_iters = [
            state.epoch_steps * epoch
            for epoch in range(save_epoch_freq, num_epochs + 1, save_epoch_freq)
        ]
        if num_iters not in save_iters:
            save_iters.append(num_iters)
    if sample_epoch_freq > 0:
        sample_freq = state.epoch_steps * sample_epoch_freq
    if val_epoch_freq > 0:
        val_freq = state.epoch_steps * val_epoch_freq

    if state.scheduler_type == "cosine":
        if int(args.get("WarmupFlatCosineScheduler.flat_steps", 0) or 0) <= 0:
            state.scheduler.flat_steps = state.epoch_steps
        if int(args.get("WarmupFlatCosineScheduler.total_steps", 0) or 0) <= 0:
            state.scheduler.total_steps = num_iters

    val_dataloader = accel.prepare_dataloader(
        state.val_data,
        start_idx=0,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=state.val_data.collate,
        persistent_workers=num_workers > 0,
    )
    print(
        "initialized validation dataloader. "
        f"train_epoch_examples={state.epoch_examples}, "
        f"train_epoch_steps={state.epoch_steps}, run_seed={state.run_seed}."
    )

    if fine_tune:
        lora.mark_only_lora_as_trainable(state.model)
        print("marked only lora as trainable.")

    # Wrap the functions so that they neatly track in TensorBoard + progress bars
    # and only run when specific conditions are met.
    global train_loop, val_loop, validate, save_samples, checkpoint

    train_loop = tracker.log("train", "value", history=False)(
        tracker.track("train", num_iters, completed=state.tracker.step)(train_loop)
    )
    val_loop = tracker.track("val", len(val_dataloader))(val_loop)
    validate = tracker.log("val", "mean")(validate)

    save_samples = when(lambda: accel.local_rank == 0)(save_samples)
    checkpoint = when(lambda: accel.local_rank == 0)(checkpoint)

    def build_train_dataloader_for_step():
        epoch = tracker.step // state.epoch_steps
        step_in_epoch = tracker.step % state.epoch_steps
        if state.data_mode == "tokenized":
            state.train_data = _make_train_token_dataset(
                args=args,
                run_seed=state.run_seed,
                epoch=epoch,
                epoch_examples=state.epoch_examples,
                chunk_frames=state.token_chunk_frames,
                n_codebooks=int(accel.unwrap(state.model).n_codebooks),
            )
        else:
            state.train_data = _make_train_dataset(
                args,
                sample_rate=state.train_data.sample_rate,
                run_seed=state.run_seed,
                epoch=epoch,
                epoch_examples=state.epoch_examples,
            )
        print(
            f"initialized train epoch {epoch + 1} "
            f"(start_step={step_in_epoch}, epoch_steps={state.epoch_steps})."
        )
        return accel.prepare_dataloader(
            state.train_data,
            start_idx=step_in_epoch * batch_size,
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=state.train_data.collate,
        )

    print("starting training loop.")
    with tracker.live:
        while tracker.step < num_iters:
            train_dataloader = build_train_dataloader_for_step()
            for batch in train_dataloader:
                current_step = tracker.step
                train_output = train_loop(state, batch, accel)
                state.latest_grad_norm = train_output.get("other/grad_norm", None)  # Update the latest grad norm

                tracker.step = current_step + 1
                last_iter = (
                    tracker.step >= num_iters if num_iters is not None else False
                )

                if sample_freq > 0 and (tracker.step % sample_freq == 0 or last_iter):
                    with _fork_seed(state.sample_seed):
                        save_samples(state, val_idx, writer)

                if val_freq > 0 and (tracker.step % val_freq == 0 or last_iter):
                    # Calculate mean grad norm for this training phase
                    if state.scheduler_type == "rlrop":
                        if hasattr(state, 'grad_norms') and state.grad_norms:
                            state.mean_grad_norm = sum(state.grad_norms) / len(state.grad_norms)
                            state.grad_norms = []  # Reset for the next validation phase
                        else:
                            state.mean_grad_norm = None

                    val_output = validate(
                        state,
                        val_dataloader,
                        accel,
                        include_grad_norm=state.scheduler_type == "rlrop",
                    )
                    print(f"Validation Output: {val_output}")  # Print validation output
                    if state.scheduler_type == "rlrop":
                        print(f"Scheduler state before step: {state.scheduler.state_dict()}")
                        state.scheduler.step(val_output["loss"], val_output.get("grad_norm"))
                        print(f"Scheduler state after step: {state.scheduler.state_dict()}")

                    checkpoint(
                        state=state,
                        save_iters=save_iters,
                        save_path=save_path,
                        fine_tune=fine_tune
                    )

                    # Reset validation progress bar, print summary since last validation.
                    tracker.done("val", f"Iteration {tracker.step}")

                if last_iter or tracker.step % state.epoch_steps == 0:
                    break


if __name__ == "__main__":
    args = argbind.parse_args()
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0
    with argbind.scope(args):
        with Accelerator() as accel:
            if accel.local_rank != 0:
                sys.tracebacklimit = 0
            train(args, accel)
