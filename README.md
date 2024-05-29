# VampNet

This repository contains recipes for training generative music models on top of the Descript Audio Codec.

## try `unloop`
you can try vampnet in a co-creative looper called unloop. see this link: https://github.com/hugofloresgarcia/unloop

# Setting up

**Requires Python 3.9**. 

you'll need a Python 3.9 environment to run VampNet. This is due to a [known issue with madmom](https://github.com/hugofloresgarcia/vampnet/issues/15). 

(for example, using conda)
```bash
conda create -n vampnet python=3.9
conda activate vampnet
```


install VampNet

```bash
git clone https://github.com/hugofloresgarcia/vampnet.git
pip install -e ./vampnet
```

## A note on argbind
This repository relies on [argbind](https://github.com/pseeth/argbind) to manage CLIs and config files. 
Config files are stored in the `conf/` folder. 

## Getting the Pretrained Models

### Licensing for Pretrained Models: 
The weights for the models are licensed [`CC BY-NC-SA 4.0`](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.ml). Likewise, any VampNet models fine-tuned on the pretrained models are also licensed [`CC BY-NC-SA 4.0`](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.ml).

Download the pretrained models from [this link](https://zenodo.org/record/8136629). Then, extract the models to the `models/` folder. 


# Usage

## Launching the Gradio Interface
You can launch a gradio UI to play with vampnet. 

```bash
python app.py --args.load conf/interface.yml --Interface.device cuda
```

# Training / Fine-tuning 

## Training a model

To train a model, run the following script: 

```bash
python scripts/exp/train.py --args.load conf/vampnet.yml --save_path /path/to/checkpoints [--resume] [--nocompile] [--lh] [--rlrop]
```
You can resume a training with --resume
For Windows you must enable -nocompile option that will disable torch.compile that is not available (around 10% slower, Windows too is around 5% than linux).
You can experiment with an alternate version of adamw optimizer called AdamW-lh using -lh option. (slower but lower grad_norm)
source: https://fabian-sp.github.io/posts/2024/02/decoupling/

Very rude implementation of ReduceLROnPlateauScheduler with command --rlrop
When using this scheduler:
NoamScheduler.factor in vampnet.yml will be treated as rlrop factor (reduce next lr by factorm typical usage 0.5<->0.8)
NoamScheduler.warmup in vampnet.yml will be treated as rlrop patience (number of bad vals before lr reduction, typical usage 10)
Also, starting lr will be honored by AdamW.lr in vampnet.yml (it has no sense when using noam scheduler as it will be immediately overwritten by Noam calculations)
Other paramaters are fixed but you can change them editing vampnet/scheduler.py, (lines 32 and 57), mainly the other relevant value is threshold (0.001 by default), unless you want it to do a completely different task.


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
Standard method is not usable for fine-tuning large dataset (error command line too long) so here's an alternative
Put your samples folder into train/ eg:vampnet/train/mysamples/
Put your (optional) validation samples in another folder on train/ eg:vampnet/train/myvalsamples/

Case 1 (no validation samples folder)
```bash
python scripts/finetune/ftcfg.py <samplefolder> <fine_tune_name>
```
example
```bash
python scripts/finetune/ftcfg.py mysamples mymodel
```
Case 2 (with validation sample folder)
```bash
python scripts/finetune/ftcfgval.py <samplefolder> <fine_tune_name> <valfolder>
```
example
```bash
python scripts/finetune/ftcfgval.py mysamples mymodel myvalsamples
```
After that you can modify your lora.yml according to your desired epochs using the following script
NOTE: configure conf/lora/lora.yml batch_size/num_workers according to you free gpu RAM.
i suggest 5 for 24GB , 3 for 16GB, 2 for 12GB, 1 for 8GB, if you run out of vram it will continue but slower
use same value for both batch_size/workers

Case 1 (no validation samples folder)
```bash
python scripts/finetune/ftloracfg.py <samplefolder> <val_epochs_freq> <sample(save)_epochs_freq)> <1st_epochs_checkpoint> <2nd_epochs_checkpoint> <3rd_epochs_checkpoints> <4th_epochs_checkpoint> <5th_epochs_checkpoint>
```
example:
```bash
python scripts/finetune/ftloracfg.py mysamples 25 50 100 200 300 400 500
```
Case 2 (with validation samples folder)
```bash
python scripts/finetune/ftloracfgval.py <samplefolder> <validation_folder> <val_epochs_freq> <sample(save)_epochs_freq)> <1st_epochs_checkpoint> <2nd_epochs_checkpoint> <3rd_epochs_checkpoints> <4th_epochs_checkpoint> <5th_epochs_checkpoint>
```
example:
```bash
python scripts/finetune/ftloracfgval.py mysamples myvalsamples 25 50 100 200 300 400 500
```

## Fork_info
- Little modifications to let this work on Windows (and ubuntu WSL)
Note: in order to use python 3.11.x (Windows only) install madmom from git source
- Alternate method to configure files for fine-tuning
- New script createchunks.py, more info inside the script
- Added cp1252_To_Append.py, you have to append those lines to your cp1252.py file under python_path\Lib\Encodings in order to avoid charmap errors during fine-tuning/training (Windows Only)
- Added Gradio-Export script, this simple script will save your gradio-outputs wav files into gradio-export folder removing all folders and adding a timestamp prefix to names, the script supports also double click, no need to launch from command line
- new script check_normalization, will check an entire folder (with optional normalization)
- new script check_LUFS, will output LUFS for a folder
- new script check cuda, check if cuda is working
- new script train_calculator, will calculate total training based on settings, original model is around 2500 total train value (based on his pdf on ArXiv)
- new script rename_special_chars, will rename audio chunks names to remove not ASCII chars that will lead to errors during training
- new script convert_m4a_to_wav
- new script convert_to_mono
- new Powershell script demucs_folder, will separate tracks for an entire folder using facebookresearch's demucs
- new script merge_demucs, will merge back demucsed files (you may want to merge back after having removed something eg.Drums)
- new script parse_and_export_linux_log, will convert log.txt from training into csv (works only for linux logs)
- new folder scripts/compare, temporary folder to fix bad audio results for training in linux (used in bash setup)
- new bash script quickinstall.sh, use this to quickly configure a fresh ubuntu22.04 Cuda container, more info inside
- new pth_reader to see pth content
- new noam_recalculator, will recalculate noam.factor and step to maintain a similar decay curve when changing the batch size (moving to another server) and modifies tracker.pth and scheduler.pth accordingly
- new pth_editor, you can manually change Noam.Factor Noam.warmup batch_size current_step and lr into tracker.pth, scheduler.pth and optimizer.pth, to make cyclic decay or other stuff
- Added options -nocompile and -lh on train.py
- new loselesscompress script to quickly convert dataset from wav to flac and viceversa. helps uploading time and training time too.
- new option for ReduceLROnPlateauScheduler as alternative for NoamScheduler
