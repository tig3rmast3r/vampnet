#!/bin/bash

# pth download not included, you will have to download manually
# BEFORE launching this script:
# cd workspace
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /workspace/Miniconda3-latest-Linux-x86_64.sh
# bash /workspace/Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda
# echo 'export PATH="/workspace/miniconda/bin:$PATH"' >> ~/.bashrc
# source ~/.bashrc
# conda init bash
# source ~/.bashrc
# conda create -n vampnet python=3.10.14 -y
# conda activate vampnet
# wget https://raw.githubusercontent.com/tig3rmast3r/vampnet/ismir/scripts/utils/quickinstall.sh
# chmod +x quickinstall.sh
# ./quickinstall.sh

git clone https://github.com/tig3rmast3r/vampnet
#best python/pytorch combination for torch.compile to work with no errors
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install cython
apt install build-essential -y
git clone https://github.com/CPJKU/madmom
cd madmom
pip install .
cd ../
pip install -e ./vampnet
wget https://raw.githubusercontent.com/tig3rmast3r/vampnet/ismir/scripts/utils/compare/working
wget https://raw.githubusercontent.com/tig3rmast3r/vampnet/ismir/scripts/utils/compare/compare.py
conda list >> new
python compare.py
pip install -r requirements_to_update.txt
