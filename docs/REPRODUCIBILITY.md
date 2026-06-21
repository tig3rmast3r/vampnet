# Reproducibility

This fork treats the `vampnet10` Conda environment as the canonical Linux setup.
Conda dependencies are pinned in `env/environment.yml`; pip dependencies are
split into `env/source-requirements.txt` and `env/pip-requirements.txt`. The
local package is installed with `pip install -e . --no-deps` so dependency
resolution does not silently drift through packaging metadata.

## Why Conda plus pinned forks

VampNet depends on audio and CUDA packages whose behavior can change across
machines even when the Python code is unchanged. The environment pins:

- Python 3.10 and PyTorch 2.3.1 with CUDA 12.1 runtime from Conda.
- FFmpeg inside the Conda env, so audio loading does not use `/usr/bin/ffmpeg`.
- Source dependencies through `tig3rmast3r` forks pinned to exact commits.

The source requirements are installed with `--no-deps` because some old package
metadata still points at upstream Git URLs for code that is mirrored in this
fork. The following `pip-requirements.txt` step installs the required runtime
dependencies explicitly.

The host machine still needs a compatible NVIDIA driver. A system-wide CUDA
toolkit is not required for normal use because the Conda env provides the CUDA
runtime used by PyTorch.

## Locked install

```bash
git clone https://github.com/tig3rmast3r/vampnet.git
cd vampnet

conda-lock install -n vampnet10 env/conda-lock.yml
conda activate vampnet10

pip install --no-deps -r env/source-requirements.txt
pip install -r env/pip-requirements.txt
pip install -e . --no-deps
```

## Models

Download the pretrained checkpoints from
[`zenodo.org/record/8136629`](https://zenodo.org/record/8136629), then place
them in this layout:

```text
models/
  vampnet/
    codec.pth
    coarse.pth
    c2f.pth
  wavebeat.pth
```

## Maintainers: update the lock

`env/environment.yml` is the human-readable Conda recipe. After changing it,
regenerate `env/conda-lock.yml`:

```bash
conda-lock -f env/environment.yml -p linux-64
```

## Optional FlashAttention

FlashAttention is not part of the base install because it is a compiled CUDA
extension and is much less portable than the rest of the stack. It is only
needed for configs with `VampNet.flash_attn: true`.

```bash
pip install -r env/flash-attn.txt --no-build-isolation
```

## Verification

Check that Python imports resolve inside the environment:

```bash
python - <<'PY'
import torch, madmom, lac, audiotools, vampnet
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("madmom", madmom.__version__)
PY
```

Check that the codec checkpoint loads:

```bash
python - <<'PY'
from lac.model.lac import LAC
codec = LAC.load("models/vampnet/codec.pth", map_location="cpu")
print(codec.sample_rate, codec.hop_length, codec.quantizer.n_codebooks)
PY
```

Check that FFmpeg comes from the Conda env, not the system:

```bash
which ffmpeg
ffmpeg -version
```

The path should be under the active Conda environment, for example
`.../envs/vampnet10/bin/ffmpeg`.

Finally, check the main CLIs parse:

```bash
python scripts/exp/train.py --help
python scripts/utils/pretokenize_dataset_codec.py --help
```
