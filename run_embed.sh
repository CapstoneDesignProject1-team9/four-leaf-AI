#!/bin/bash
#SBATCH -J embed
#SBATCH -p p02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o embed-%j.out

# GPU(A6000)로 공지·에브리타임 강의평·강의계획서 벡터DB를 만든다.
# 제출: sbatch run_embed.sh      확인: squeue -u $USER      로그: tail -f embed-작업번호.out
#
# 사전 준비 (인터넷이 되는 로그인 노드에서 한 번만):
#   conda create -n embed python=3.12 -y && conda activate embed
#   pip install -r knowledge/embeddings/requirements-embedding.txt
#   BGE-M3 모델을 .cache/embedding_models 에 미리 내려받는다 (컴퓨트 노드는 인터넷이 막혀 있다).
# 입력 청크 파일(knowledge/processed/...)이 없는 코퍼스는 건너뛴다.
#
# 환경이 다르면 아래 변수를 바꾼다. 예: CONDA_ENV=embed BATCH_SIZE=32 sbatch run_embed.sh
CONDA_ENV="${CONDA_ENV:-embed}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

export HF_HUB_OFFLINE=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR" || exit 1

nvidia-smi

NOTICES=knowledge/processed/notices/knu_notice_chunks.jsonl
SYLLABI=knowledge/processed/syllabi/knu_syllabus_chunks.jsonl
REVIEWS=knowledge/processed/everytime_lectures/review_chunks.jsonl

if [ -f "$NOTICES" ]; then
  python knowledge/embeddings/embed_notices.py index --device cuda --offline --batch-size "$BATCH_SIZE"
else
  echo "[skip] notices: $NOTICES not found"
fi

if [ -f "$REVIEWS" ]; then
  python knowledge/embeddings/embed_everytime_reviews.py index --device cuda --offline --batch-size "$BATCH_SIZE"
else
  echo "[skip] everytime reviews: $REVIEWS not found"
fi

if [ -f "$SYLLABI" ]; then
  python knowledge/embeddings/embed_notices.py index --corpus syllabi --device cuda --offline --batch-size "$BATCH_SIZE"
else
  echo "[skip] syllabi: $SYLLABI not found"
fi
