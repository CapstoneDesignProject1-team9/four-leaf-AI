# A6000에서 Base + RAG 평가 생성

이 절차는 Docker와 vLLM 없이 기존 Chroma/BGE-M3 검색 코드와 Transformers의 Llama 3.1 8B를 함께 사용합니다. `generate_answers.py`의 `transformers-rag` 백엔드는 평가 배치용이며 FastAPI 채팅 서버를 띄우지 않습니다.

## 1. Mac에서 RAG 파일을 로그인 노드로 복사

벡터 DB와 임베딩 모델 캐시가 필요합니다. 현재 로컬 프로젝트에서 벡터 DB는 `four-leaf-AI/knowledge/vectorstores/`, BGE-M3 캐시는 `four-leaf-AI/.cache/embedding_models/models--BAAI--bge-m3/`에 있습니다. SSH 접속 가능한 실제 로그인 호스트 또는 VS Code Remote SSH 파일 전송을 사용합니다. `ABRM02`가 Mac에서 DNS로 해석되지 않으면 그 이름을 `rsync` 호스트로 사용하지 않습니다.

```bash
cd /Users/tak/Desktop/CDP1-team9-fourleaf
rsync -avh --partial four-leaf-AI/knowledge/vectorstores/ coss36@<LOGIN_HOST>:~/four-leaf-AI/knowledge/vectorstores/
rsync -avh --partial four-leaf-AI/.cache/embedding_models/models--BAAI--bge-m3/ coss36@<LOGIN_HOST>:~/four-leaf-AI/.cache/embedding_models/models--BAAI--bge-m3/
```

또한 최신 `eval/generate_answers.py`를 서버의 같은 경로에 복사합니다. HF의 Llama 3.1 8B 캐시와 어댑터는 이미 서버에 있으므로 다시 복사할 필요가 없습니다.

## 2. 서버에서 데이터·의존성 확인

로그인 노드에서:

```bash
cd ~/four-leaf-AI
du -sh knowledge/vectorstores .cache/embedding_models/models--BAAI--bge-m3
python -c "import chromadb, sentence_transformers, langchain_openai; print('RAG dependencies available')"
```

가져온 벡터 DB는 아래 경로에 있어야 합니다.

```text
knowledge/vectorstores/knu_notice_bge_m3
knowledge/vectorstores/knu_syllabi_bge_m3
knowledge/vectorstores/everytime_bge_m3
```

BGE-M3 캐시는 아래 설정의 디렉터리 구조를 유지합니다.

```text
.cache/embedding_models/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181
```

## 3. GPU 작업 셸 할당

로그인 노드에서 노드를 고정하지 않고 `p02` GPU를 요청합니다. 계산 노드에 들어간 다음 `nvidia-smi`로 실제 배정된 GPU를 확인합니다.

```bash
srun -p p02 --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=12:00:00 --pty bash -l
hostname
nvidia-smi
```

## 4. 검색과 단일 문항 생성 확인

GPU 노드에서:

```bash
conda activate ai
cd ~/four-leaf-AI
export HF_HOME=/abr/coss36/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODEL_DIR="$(find "$HF_HOME/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n1)"
export EMBEDDING_MODEL_CACHE_DIR=./.cache/embedding_models
export EMBEDDING_DEVICE=cuda

python - <<'PY'
import json
source = json.load(open("eval/student_eval_dataset.json", encoding="utf-8"))
json.dump(source[:1], open("eval/student_rag_smoke.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
PY

python eval/generate_answers.py --backend transformers-rag --dataset eval/student_rag_smoke.json --mode rag
```

`student_rag_smoke.json`에서 응답과 contexts가 저장됐는지 확인합니다. 검색 결과가 없거나 Chroma/BGE 초기화 오류가 나면 전체 생성으로 넘어가지 말고 그 오류부터 해결합니다.

## 5. 학생용 30문항 생성

```bash
cp eval/student_eval_dataset.json eval/student_eval_dataset.before_a6000_rag.json
python eval/generate_answers.py --backend transformers-rag --dataset eval/student_eval_dataset.json --mode rag
```

이 실행은 각 문항의 `Base Model + RAG` 응답, 검색 문맥, latency를 저장합니다. 교수자용 데이터에는 교수자 RAG 조건이 포함되지 않습니다.
