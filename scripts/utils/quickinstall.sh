#!/bin/bash

# pth download not included, you will have to download manually
# BEFORE launching this script:
# cd workspace
# wget https://raw.githubusercontent.com/tig3rmast3r/vampnet/ismir/scripts/utils/quickinstall.sh
# chmod +x quickinstall.sh
# ./quickinstall.sh


wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /workspace/Miniconda3-latest-Linux-x86_64.sh
bash /workspace/Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda
export PATH="/workspace/miniconda/bin:$PATH"
conda create -n vampnet python=3.10.14 -y
export PATH="/workspace/miniconda/envs/vampnet/bin:$PATH"
git clone https://github.com/hugofloresgarcia/vampnet.git
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
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
conda init bash
source ~/.bashrc