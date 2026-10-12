# 에타 리뷰 임베딩

`embed_everytime_reviews.py`는 공지용 `embed_notices.py`의 모델·배치·재개 기능을 재사용하는 별도 진입점입니다. 두 Python 파일이 모두 필요합니다. 공지용 실행 코드는 수정하지 않았습니다.

## 설치 및 실행 (프로젝트 루트)

```powershell
.\.venv312\Scripts\python.exe -m pip install -r knowledge/embeddings/requirements-embedding.txt
.\.venv312\Scripts\python.exe knowledge/embeddings/embed_everytime_reviews.py check
.\.venv312\Scripts\python.exe knowledge/embeddings/embed_everytime_reviews.py index --batch-size 8
```

- check: 청크 형식·중복 ID·본문 해시·에타 출처를 검사합니다. 모델 로딩·다운로드·임베딩·DB 쓰기는 하지 않습니다. 토큰 길이 검사는 아닙니다.
- index: BGE-M3를 사용해 정규화된 1024차원 벡터를 배치 저장합니다. 실제 토큰 길이를 검사해 모델 한도를 넘으면 자르지 않고 중단합니다. 이 경우 청크 크기를 줄여 다시 청킹한 후 index를 재실행하세요.
- 기존 공지와 같은 revision의 모델 캐시를 재사용합니다. 캐시가 없으면 최초 모델 다운로드가 발생할 수 있습니다. 다운로드를 원하지 않으면 --offline을 붙이세요.
- 기본 CPU입니다. CUDA가 준비된 환경에서만 --device cuda를 사용하세요. 메모리 부족 시 --batch-size 4 또는 2로 줄이세요.
- 로컬 모델 계산이며 LLM API는 호출하지 않습니다. 시간은 CPU/GPU 환경에 따라 달라집니다.

## 경로와 격리

- 입력: knowledge/processed/everytime_lectures/review_chunks.jsonl
- DB: knowledge/vectorstores/everytime_bge_m3/
- 컬렉션: everytime_reviews_bge_m3_v1
- 모델 캐시: .cache/embedding_models/

공지·강의계획서 DB와 분리되어 있으며 앱의 채팅 RAG에서 강의평 질의에 사용합니다. 벡터 DB·모델 캐시는 Git에서 제외됩니다.

## 홈페이지 연결

- 기본 경로·컬렉션은 위와 같으며 `CHROMA_EVERYTIME_PERSIST_DIR`, `CHROMA_EVERYTIME_COLLECTION_NAME`으로 변경할 수 있습니다.
- DB가 없으면 기존 공지·강의계획서 기능을 유지합니다. DB가 있으나 모델 revision·차원·정규화·색인 완료 상태가 호환되지 않으면 시작 오류로 알립니다.
- `자료구조 강의평 알려줘`, `자료구조 수업 난이도는?`처럼 강의평·후기·난이도 키워드를 사용합니다. 명시적인 강의평 질의가 기존 강의계획서 키워드보다 우선합니다.
- 저장된 과목명·교수명이 질문에 포함되면 정확 일치 메타데이터 필터를 적용합니다. 별칭·오타 자동 교정은 하지 않습니다. 이름이 인식되지 않으면 넓게 검색하므로 답변의 과목·교수를 반드시 확인하세요.
- 한 리뷰의 중복 청크를 줄이고 최대 RAG_TOP_K개 후기를 전달합니다. 검색 후기만으로 전체 평균이나 객관적 사실을 단정하지 않도록 안내합니다.
- 기본 설정에서는 `.env.dev` 변경이 필요 없습니다. 개발 Compose의 프로젝트 마운트를 사용하는 경우 `docker compose --env-file .env.dev -f docker-compose.dev.yml restart ai`로 코드·DB를 다시 읽습니다.
- 환경변수 변경은 컨테이너 재생성이 필요합니다. 재임베딩이나 모델 재다운로드는 필요하지 않습니다.

## 검색 테스트

```powershell
.\.venv312\Scripts\python.exe knowledge/embeddings/embed_everytime_reviews.py search "과제가 많고 시험이 어려운 수업" --course-name "자료구조"
```

--professor "이호경" 같은 정확 일치 필터도 사용할 수 있습니다. 본문만 임베딩하므로 특정 과목·교수 검색에는 필터를 함께 쓰세요. 결과는 청크 단위이며 한 리뷰의 여러 청크가 나올 수 있습니다. 요약·통계 단계에서는 document_id로 구분해야 합니다.

## 재실행과 주의사항

- Ctrl+C 이후 같은 index 명령을 실행하면 저장 완료된 동일 본문 벡터는 재사용합니다.
- 추천 수 등 메타데이터만 바뀌면 벡터를 재계산하지 않고 메타데이터만 갱신합니다.
- 입력은 전체 스냅샷입니다. 전체 성공 후 입력에서 사라진 청크는 inactive 처리합니다. 일부 샘플 파일을 같은 DB에 색인하면 나머지가 비활성화되므로 전체 청크 파일을 사용하세요.
- index 중에는 입력 파일을 바꾸거나 해당 DB의 다른 쓰기·검색 작업을 실행하지 마세요. 중단된 색인은 index를 완료하기 전 검색할 수 없습니다.
- 프로세스가 강제 종료되면 knowledge/vectorstores/everytime_bge_m3.index.lock이 남을 수 있습니다. 다른 index 프로세스가 종료됐는지 확인한 뒤에만 잠금 파일을 제거하고 재개하세요.
- 이번 구현 시 모델 다운로드나 실제 임베딩은 실행하지 않았습니다. check 성공만으로 모델·DB 동작 검증이 완료되는 것은 아닙니다.
