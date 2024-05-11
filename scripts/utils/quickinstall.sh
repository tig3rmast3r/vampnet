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
# conda create -n vampnet python=3.10.14
# conda activate vampnet
# wget https://github.com/tig3rmast3r/vampnet/tree/ismir/scripts/utils/quickinstall.sh
# chmod +x quickinstall.sh
# ./quickinstall.sh

git clone https://github.com/hugofloresgarcia/vampnet.git
#this install latest pytorch, 2.3.0 at the time of writing, force it if there are errors with newer versions
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
echo |
pip install cython
apt install build-essential
echo |
git clone https://github.com/CPJKU/madmom
cd madmom
pip install .
cd ../
pip install -e ./vampnet
wget https://github.com/tig3rmast3r/vampnet/tree/ismir/scripts/utils/compare/working
wget https://github.com/tig3rmast3r/vampnet/tree/ismir/scripts/utils/compare/compare.py
conda list >> new
python compare.py
pip install -r requirements_to_update.txt

