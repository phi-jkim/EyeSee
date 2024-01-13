#!/bin/bash
#SBATCH --partition=tig
#SBATCH --qos=tig-main
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
export HOME=/data/scratch/$USER

source /data/scratch/jinhakim/miniconda3/etc/profile.d/conda.sh
conda activate ml_train
pip3 install -r requirements.txt
python3 train_annotate.py
echo "Job finished" 