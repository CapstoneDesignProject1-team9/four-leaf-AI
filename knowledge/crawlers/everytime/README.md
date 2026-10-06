# 에브리타임 강의평 크롤러

공지사항 크롤러와 독립적으로 실행합니다. 경북대 과목명으로 강의 후보를 찾은 뒤 본인 계정으로 열람 가능한 강의평을 수집합니다. 비밀번호는 수동 입력하며 로그인 프로필이나 리뷰 원문을 Git에 올리지 않습니다.

## 프로젝트 루트에서 설치

아래 명령은 Python 환경이 활성화되어 있다는 전제입니다. 기존 프로젝트 환경을 사용하거나 별도 가상환경을 만드세요. 크롤러 의존성은 서비스 전체 의존성과 분리했습니다.

```powershell
python -m pip install -r knowledge/crawlers/everytime/requirements.txt
$env:PLAYWRIGHT_BROWSERS_PATH = "$PWD\knowledge\crawlers\everytime\.playwright-browsers"
python -m playwright install chromium
```

## 1. 강의 목록 수집

```powershell
python knowledge/crawlers/everytime/find_everytime_lectures.py --limit 0
```

동봉한 과목 목록에서 과목명으로 검색합니다. 과목 코드는 로컬 목록 선택용이며 검색어로 사용하지 않습니다. 프로그램의 브라우저에서 직접 로그인하고 안내에 따라 Enter를 누르세요.

## 2. 전체 강의평 수집

```powershell
python knowledge/crawlers/everytime/crawl_everytime_reviews.py --candidate-json knowledge/raw/everytime_lectures/lecture_candidates.json --all-courses --max-pages 100
```

자료구조만 시험하려면 --all-courses 대신 --course-name "자료구조"를 사용합니다.

- 처리 기록을 읽고 기존 처리 강의를 건너뜁니다. 중단 후 같은 명령으로 재개합니다.
- --retry-failed: 확인 필요 강의도 재시도합니다.
- --refresh: 기존 처리 강의도 다시 방문해 리뷰와 추천 수를 갱신합니다.
- 첫 리뷰는 기본 3초 대기합니다. 느리면 --review-wait-seconds 10으로 늘리세요.
- 리뷰가 안 보이면 확인 필요로 기록합니다. 리뷰 0개를 확정한 것은 아닙니다.
- 최대 스크롤 횟수 도달은 미완료로 기록합니다. 추가 로딩이 없어 종료해도 사이트 전체 수집을 보장하지 않습니다.
- 접근 차단·인증 오류는 중단하며 우회하지 않습니다.

## 저장 위치

기본 경로는 실행 위치가 아니라 프로젝트 경로 기준입니다. 명시적으로 전달한 상대 경로는 현재 작업 폴더 기준입니다.

- knowledge/raw/everytime_lectures/lecture_candidates.json: 강의 목록
- knowledge/raw/everytime_lectures/reviews.jsonl: 리뷰별 원본
- knowledge/raw/everytime_lectures/reviews_attempts.jsonl: 진행 기록
- knowledge/raw/everytime_lectures/reviews_by_course.json: 과목 → 교수별 조회
- knowledge/raw/everytime_lectures/reviews_by_professor.json: 교수 → 과목별 조회
- .cache/everytime-browser-profile/: 로그인 프로필 (공유 금지)

두 조회 JSON에는 수집한 리뷰 기준 평균 별점과 집계 수가 포함됩니다. 추천 수는 별점 평균에 가중하지 않습니다. 교수명 그룹은 동명이인을 구별하지 않습니다.

## 기존 수집 결과 이어받기

기존 실행 폴더의 results에서 lecture_candidates.json, reviews.jsonl, reviews_attempts.jsonl을 새 원본 폴더로 함께 복사해야 이전 수집 기록을 이어받습니다. 원본 파일은 이번 코드 통합에 포함하지 않았습니다. 기존 폴더와 로그인 프로필은 그대로 보존되어 있습니다.

두 조회 JSON만 다시 생성:

```powershell
python knowledge/crawlers/everytime/crawl_everytime_reviews.py --export-only
```

전처리·청킹·임베딩 코드는 아직 이 크롤러에 포함되어 있지 않습니다. 향후 전처리 결과는 knowledge/processed/everytime_lectures/로 구분합니다. 검색·임베딩 원본은 reviews.jsonl 한 벌만 사용하세요.

## 사이트 접속 없는 테스트

```powershell
python -m unittest discover -s knowledge/crawlers/everytime -p "test_*.py"
```

Playwright와 로컬 Chromium 설치가 필요합니다. 테스트는 임시 파일과 로컬 HTML을 사용하며 에타에 접속하지 않습니다.
