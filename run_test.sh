#!/bin/bash
#SBATCH -J testadapters
#SBATCH -p p02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o test-%j.out

export HF_HUB_OFFLINE=1
source /abr/coss36/miniforge3/etc/profile.d/conda.sh
conda activate ai
cd /abr/coss36/four-leaf-AI
python test_adapters.py
