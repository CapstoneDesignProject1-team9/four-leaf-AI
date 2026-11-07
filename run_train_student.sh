#!/bin/bash
#SBATCH -p p02
#SBATCH --gres=gpu:1
#SBATCH -J train_student
#SBATCH -o train_student_%j.log
cd $SLURM_SUBMIT_DIR
source ~/.bashrc
conda activate ai
export HF_HUB_OFFLINE=1
python train/finetune.py \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --data data/train_mix_6000.jsonl \
  --adapter-dir ./adapter_student_llama8b
