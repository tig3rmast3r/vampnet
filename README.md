# VampNet (ismir2023 fork)

This is a maintained fork of VampNet focused on reproducible Linux training,
tokenized datasets, and checkpoint validation.

The upstream README from the original fork point is available here:
[`hugofloresgarcia/vampnet@72e2675/README.md`](https://github.com/hugofloresgarcia/vampnet/blob/72e2675790091fe28ecfd8391303a46b25a703db/README.md).

## Reproducible Install

The canonical environment is `vampnet10`: Python 3.10.14, PyTorch 2.3.1,
CUDA 12.1 runtime from Conda, in-env FFmpeg 4.3, pinned audio tooling, and
pinned source forks for fragile dependencies such as `madmom`, `lac`,
`audiotools`, and `wavebeat`.

```bash
git clone https://github.com/tig3rmast3r/vampnet.git
cd vampnet

conda-lock install -n vampnet10 env/conda-lock.yml
conda activate vampnet10

pip install --no-deps -r env/source-requirements.txt
pip install -r env/pip-requirements.txt
pip install -e . --no-deps
```

`env/environment.yml` is the human-readable Conda recipe used to regenerate
the lock. For details and verification commands, see
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

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

The model weights are licensed
[`CC BY-NC-SA 4.0`](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en).
Models fine-tuned from those weights inherit the same license constraints.

# Usage

## Launching the Gradio Interface

You can launch a Gradio UI to experiment with VampNet.

```bash
python app.py --args.load conf/interface.yml --Interface.device cuda
```

# Training / Fine-tuning

## Newer PyTorch versions and Blackwell GPUs

A modified `train_blackwell.py` is provided for newer PyTorch versions. It is
intended for PyTorch 2.7+, the minimum required for Blackwell GPU compatibility.
I encountered severe gradient explosions when training with AMP on Blackwell
GPUs. However, I have not tested this extensively enough to determine whether
the issue is related to Blackwell itself or to the newer PyTorch stack. Since I
do not own a Blackwell GPU, I cannot investigate it further.

## Pre-Tokenizer (optional)

For larger datasets, pre-tokenizing audio into `.tokens.npz` files can make
training faster and use less RAM, because the codec encoding and audio decoding
work is done once before training instead of repeatedly inside each training
run.

```bash
python scripts/utils/pretokenize_dataset_codec.py \
  --input-roots /path/to/DATASET /path/to/DATASET_no_kick \
  --output-root /path/to/TOK_DATASET \
  --codec-ckpt models/vampnet/codec.pth
```

Then point your training config to the tokenized folders and use
`data_mode: tokenized`. See the script help for all options:

```bash
python scripts/utils/pretokenize_dataset_codec.py --help
```

## Training a model

To train a model, run the following script:

```bash
python scripts/exp/train.py --args.load conf/vampnet.yml --save_path /path/to/checkpoints [--resume] [--nocompile] [--lh] [--dropout]
```
You can resume training with `--resume`.

On Windows, use `--nocompile` unless you know your PyTorch/backend setup supports `torch.compile`.
Windows CPU support exists in newer PyTorch versions, but this fork's canonical setup is Linux `vampnet10`.

You can enable dynamic LH weight decay with `lh: true`. In this mode `AdamW.weight_decay`
is ignored and the real applied weight decay is derived from the scheduler LR, clamped between `lh_wd_start` and `lh_wd_end`.

Background: [Decoupling weight decay](https://fabian-sp.github.io/posts/2024/02/decoupling/).

To control transformer dropout from CLI without swapping source files:

- default behavior is dropout OFF
- `--dropout` enables dropout during training
- `--VampNet.dropout <value>` sets dropout probability when enabled (if omitted and `--dropout` is set, default is `0.1`)

Use `--amp` to enable BF16 precision, reducing VRAM use and accelerating
training on modern GPUs.

Use `scheduler: noam`, `scheduler: cosine`, or `scheduler: rlrop`.

RLROP uses dedicated keys:

- `RLROPScheduler.lr`
- `RLROPScheduler.factor`
- `RLROPScheduler.patience`
- `RLROPScheduler.threshold`
- `RLROPScheduler.threshold_mode`
- `RLROPScheduler.min_lr`
- `RLROPScheduler.warmup_steps`

This fork includes FlashAttention v2 support for compatible single-GPU configs.
FlashAttention is optional; install `env/flash-attn.txt` only when using configs
with `VampNet.flash_attn: true`.

for multi-gpu training, use torchrun:

```bash
torchrun --nproc_per_node gpu scripts/exp/train.py --args.load conf/vampnet.yml --save_path path/to/ckpt
```

You can edit `conf/vampnet.yml` to change the dataset paths or any training hyperparameters. 

For coarse2fine models, you can use `conf/c2f.yml` as a starting configuration. 

See `python scripts/exp/train.py -h` for a list of options.

## Debugging training

To debug training, it's easier to debug with 1 gpu and 0 workers

```bash
CUDA_VISIBLE_DEVICES=0 python -m pdb scripts/exp/train.py --args.load conf/vampnet.yml --save_path /path/to/checkpoints --num_workers 0
```

## Fine-tuning
To fine-tune a model, use the script in `scripts/exp/fine_tune.py` to generate 3 configuration files: `c2f.yml`, `coarse.yml`, and `interface.yml`. 
The first two are used to fine-tune the coarse and fine models, respectively. The last one is used to launch the gradio interface.

```bash
python scripts/exp/fine_tune.py "/path/to/audio1.mp3 /path/to/audio2/ /path/to/audio3.wav" <fine_tune_name>
```

This will create a folder under `conf/<fine_tune_name>/` with the 3 configuration files.

The save_paths will be set to `runs/<fine_tune_name>/coarse` and `runs/<fine_tune_name>/c2f`. 

launch the coarse job: 
```bash
python scripts/exp/train.py --args.load conf/generated/<fine_tune_name>/coarse.yml 
```

this will save the coarse model to `runs/<fine_tune_name>/coarse/ckpt/best/`.

launch the c2f job: 
```bash
python  scripts/exp/train.py --args.load conf/generated/<fine_tune_name>/c2f.yml 
```

launch the interface: 
```bash
python  app.py --args.load conf/generated/<fine_tune_name>/interface.yml 
```

## Fine-tuning ALT Method (for large datasets)
Upstream method is not usable for fine-tuning large dataset (error command line too long) so here's an alternative

Case 1 (no validation samples folder)
```bash
python scripts/finetune/ftcfg.py <samplefolder> <fine_tune_name>
```
example
```bash
python scripts/finetune/ftcfg.py /dataset/mytrainsamples mymodel
```
Case 2 (with validation sample folder)
```bash
python scripts/finetune/ftcfgval.py <samplefolder> <fine_tune_name> <valfolder>
```
example
```bash
python scripts/finetune/ftcfgval.py /dataset/mytrainsamples mymodel /dataset/myvalsamples
```

Case 1 (no validation samples folder)
```bash
python scripts/finetune/ftloracfg.py <samplefolder> <val_epochs_freq> <sample(save)_epochs_freq)> <1st_epochs_checkpoint> <2nd_epochs_checkpoint> <3rd_epochs_checkpoints> <4th_epochs_checkpoint> <5th_epochs_checkpoint>
```
example:
```bash
python scripts/finetune/ftloracfg.py /dataset/mytrainsamples 25 50 100 200 300 400 500
```
Case 2 (with validation samples folder)
```bash
python scripts/finetune/ftloracfgval.py <samplefolder> <validation_folder> <val_epochs_freq> <sample(save)_epochs_freq)> <1st_epochs_checkpoint> <2nd_epochs_checkpoint> <3rd_epochs_checkpoints> <4th_epochs_checkpoint> <5th_epochs_checkpoint>
```
example:
```bash
python scripts/finetune/ftloracfgval.py /dataset/mytrainsamples /dataset/myvalsamples 25 50 100 200 300 400 500
```

## Fork ChangeLog

### 2024
- Little modifications to let this work on Windows (and ubuntu WSL), you need to append cp1252 if you plan to train, see below
- Alternate method to configure files for fine-tuning
- Added cp1252_To_Append.py, you have to append those lines to your cp1252.py file under python_path\Lib\Encodings in order to avoid charmap errors during fine-tuning/training (Windows Only)
- Added options -nocompile and -lh on train.py
- new option for ReduceLROnPlateauScheduler as alternative to NoamScheduler
- Flash Attention v2 integration
- Gradio-Export script, which copies generated WAV files from `gradio-outputs` into `gradio-export` with timestamped filenames
- new script rename_special_chars, will rename audio chunks names to remove not ASCII chars that will lead to errors during training
- several other scripts (now under scripts/utils/legacy)

### June 2026

- [new] Blackwell/Torch 2.7+ training entrypoint (`train_blackwell.py`)
- [fix] Dropout correctly disabled when set to 0 (saves VRAM)
- [fix] Seeds are now step/epoch based and do not generate already-used patterns when resuming
- [fix] Typical filter flag now applies correctly
- [change] RLROP now has its own dedicated configuration variables
- [new] Cosine scheduler
- [new] Pretokenizer script and support for training with tokenized audio files
- [new] Conda-lock reproducible install with pinned fork dependencies
- [change] Improved `rename_special_chars` script
- [change] Improved `gradio-export` script
- [change] `--lh` behavior applied directly to AdamW (faster)
- [new] Extra TensorBoard graphs for `--lh`
- [new] Left-click logger audio player, CSV click annotations, and `move_files_from_csv` script
- [new] Duration- and audio-matching dataset deduplication script
- [new] Bulk folder conversion to mono FLAC
- [new] `clean_audio_tree` utility
- [new] Checkpoint validation script
