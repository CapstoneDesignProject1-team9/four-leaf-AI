# 공지·강의계획서 임베딩 및 검색

아래 파일 두 개를 기존 프로젝트에 넣으세요.

```text
knowledge/
  embeddings/
    embed_notices.py
    requirements-embedding.txt
  processed/notices/
    knu_notice_chunks.jsonl
  processed/syllabi/
    knu_syllabus_chunks.jsonl
```

## 설치 및 실행

Python 3.10~3.12 권장. 프로젝트 루트에서 실행합니다.
프로젝트의 기존 ChromaDB 0.5.11은 이 Windows 환경에서 C++ 빌드 도구를 요구했습니다.
이 단계는 Windows용 패키지가 있는 ChromaDB 1.0.20을 사용하므로 반드시 전용 가상환경을 사용하세요.

```powershell
python -m venv .venv-embedding
.\.venv-embedding\Scripts\python.exe -m pip install -r knowledge/embeddings/requirements-embedding.txt
.\.venv-embedding\Scripts\python.exe knowledge/embeddings/embed_notices.py check
.\.venv-embedding\Scripts\python.exe knowledge/embeddings/embed_notices.py index
.\.venv-embedding\Scripts\python.exe knowledge/embeddings/embed_notices.py search "수강정정 신청 방법은?"
```

처음에는 약 2.3GB의 모델 가중치와 관련 파일을 다운로드합니다. 다운로드 시간과 최초 로딩 시간은 검색 시간과 구분하세요.
DB는 `knowledge/vectorstores/knu_bge_m3`, 모델 캐시는 `.cache/embedding_models`에 생성됩니다.
해당 폴더 및 가상환경은 Git에 올리지 않도록 프로젝트의 `.gitignore`에 제외하세요.

## 강의계획서 색인

공지와 같은 BGE-M3 모델을 사용하되 입력 파일, ChromaDB 경로, 컬렉션은 별도로 유지합니다.
저장소 루트에서 전처리 → 기존 청커 → 임베딩 순서로 실행하세요.

```powershell
python knowledge/preprocessing/preprocess_syllabi.py
python knowledge/preprocessing/chunk_notices.py `
  --input knowledge/processed/syllabi/knu_syllabus_documents.jsonl `
  --output knowledge/processed/syllabi/knu_syllabus_chunks.jsonl
python knowledge/embeddings/embed_notices.py check --corpus syllabi
python knowledge/embeddings/embed_notices.py index --corpus syllabi
python knowledge/embeddings/embed_notices.py search "자연어처리개론 강의 내용" --corpus syllabi
```

기존 출력 파일을 다시 만들 때는 전처리기와 청커에 `--overwrite`를 추가하세요.
강의계획서 DB는 `knowledge/vectorstores/knu_syllabi_bge_m3`, 컬렉션은
`knu_syllabi_bge_m3_v1`입니다. 공지 DB인 `knu_bge_m3`와 분리되어 있습니다.

## 동작

- BGE-M3의 1024차원 dense 벡터, 정규화, cosine 거리를 사용합니다.
- CPU에서 기본 8개씩 처리합니다. `--batch-size 4` 등으로 메모리 사용을 조절할 수 있습니다.
- 동일 명령으로 재실행하면 저장된 청크는 재사용합니다. 바뀐 메타데이터만 갱신할 때도 재임베딩하지 않습니다.
- 모델 revision을 첫 실행에서 기록하며, 이후 저장·검색은 동일 revision을 사용합니다.
- 파일은 전체 스냅샷으로 취급합니다. 성공 후 입력에 없는 예전 청크는 비활성화해 검색에서 제외하며, 물리적으로 삭제하지 않습니다.
- 전체 파일 검증이 먼저 수행됩니다. 작업 중단 시 완료된 배치는 보존되고, 완료 전까지 검색은 차단됩니다.
- `index` 실행 중에는 크롤링/청킹 결과 파일을 교체하거나 같은 DB에 다른 `index`를 실행하지 마세요.
- 입력이 한도를 초과하면 내용을 자동으로 잘라 저장하지 않고 오류로 알려줍니다.
- `search`는 질문·관련 청크·출처를 확인하는 검색 테스트입니다. 아직 LLM 답변을 생성하지 않습니다.
- 검색 거리는 정답 확률이 아닙니다. 상위 결과에 정답 공지가 들어오는지 원문과 비교하세요.

다른 입력/DB를 사용하려면 각 실행에 `--input 경로`, `--db 경로`를 전달하세요.
네트워크 없이 캐시된 모델만 사용하려면 `--offline`을 추가하세요.

## 실제 질문 테스트

수집된 공지에 답이 있는 질문부터 비교하세요.

```powershell
python knowledge/embeddings/embed_notices.py search "수강정정 신청 방법" --top-k 5
python knowledge/embeddings/embed_notices.py search "다문화가정 장학금 신청 안내" --top-k 5
python knowledge/embeddings/embed_notices.py search "연구실 안전교육 이수 방법" --top-k 5
```

이 질문들은 테스트 예시이며 정량적 정확도 평가 세트는 아닙니다.

## 앱 검색

앱은 공지와 강의계획서 컬렉션을 각각 검색한 뒤 cosine 거리 기준 통합 상위 5개를 답변에 사용합니다.
강의계획서 DB가 아직 생성되지 않았으면 공지 검색으로 동작하고, DB가 준비되면 자동으로 함께 검색합니다.
임베딩 모델과 revision은 두 코퍼스에서 같게 유지하세요. 이 DB를 구버전 ChromaDB 0.5.11로 열지 마세요.
루트 `requirements.txt`는 수정하지 않았습니다.
