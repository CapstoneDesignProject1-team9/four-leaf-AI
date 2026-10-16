cat > run.sh << 'EOF'
#!/bin/bash
#SBATCH -J finetune
#SBATCH -p p02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o slurm-%j.out

# 제출: sbatch run.sh        확인: squeue -u coss36        로그: tail -f slurm-작업번호.out
# 컴퓨트 노드는 인터넷이 막혀 있으므로 이미 캐시된 모델만 사용합니다(오프라인).
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton_cache_$USER

source /abr/coss36/miniforge3/etc/profile.d/conda.sh
conda activate ai
cd /abr/coss36/four-leaf-AI

nvidia-smi
python train/finetune.py "$@"
EOF
ls -l run.sh