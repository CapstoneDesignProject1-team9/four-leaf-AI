"""학생용 합성 Q&A 생성: 공지사항 + 강의계획서에서 8개 범주 + '자료 없음(거절)' 예시, 총 TOTAL_TARGET개.

범주: 교육과정·전공 / 졸업·학적 / 수강·학사일정 / 수업·학생 운영 / 장학·지원 / 외부행사 / 교직과정 / 대학원·진학
거절: 자료가 없는 질문에 "공지사항·강의계획서에 자료가 없어 모른다"고 답하는 예시 (ABSTENTION_TARGET개)
(에브리타임 강의평은 사용하지 않는다.)

흐름
 1. 공지 원본(knowledge/raw/*notices*.jsonl)을 post_id 단위로 합친다. (본문·첨부·OCR이 레코드로 따로 있음)
 2. 공지를 Claude(haiku)로 범주 분류한다. 결과는 output/notice_categories.jsonl에 저장해 재사용한다.
 3. 강의계획서(knowledge/raw/*syllabus*.jsonl)는 '교육과정·전공'과 '수업·학생 운영' 두 관점으로 쓴다.
 4. 범주별 만들 수 있는 양을 계산해 TOTAL_TARGET에서 거절 예시를 뺀 만큼을 범주별로 나눈다.
    (양이 적은 범주는 가능한 만큼만, 남는 몫은 양이 많은 범주가 채운다.)
 5. Claude(sonnet)가 Q&A를 만들고 Claude(haiku)가 검수해 통과한 것만 저장한다.
 6. 거절 예시는 두 가지다.
    - 문서 부재형: 문서에 나오지 않는 세부 정보를 묻는 질문 → 문서에 없다고 답하고 확인 가능한 내용만 안내
    - 범위 밖형: 학과 공지·강의계획서로 답할 수 없는 주제(실시간 정보, 개인 정보, 타 학교 등) → 자료가 없어 모른다고 답함

실행 (프로젝트 루트에서)
    python scripts/generate_qa_student.py --plan   # 범주별 가능량·목표만 확인 (분류 API 호출은 함)
    python scripts/generate_qa_student.py --test   # 범주별 1묶음씩 소량 시험
    python scripts/generate_qa_student.py          # 본 실행 (중단 후 다시 실행하면 이어서 진행)

선택: output/exclude_post_ids.txt에 post_id를 한 줄씩 적으면 해당 공지는 학습 데이터에서 제외한다.
      (평가 문항의 근거 공지를 넣어 두면 학습·평가 데이터 중복을 막을 수 있다.)
"""

import argparse
import asyncio
import hashlib
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "knowledge" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# ⚙️ 핵심 설정
# ---------------------------------------------------------
TEST_MODE = False  # True: 범주별 1묶음씩만 빠르게 테스트 (--test 옵션과 같음)
TOTAL_TARGET = 3000  # 전체 목표 개수. 거절 예시도 여기에 포함된다 (--total 옵션으로 변경)
ABSTENTION_TARGET = 200  # 그중 '자료 없음(거절)' 예시 개수 (--abstain 옵션으로 변경)
CONCURRENCY = 5  # 동시 처리 워커 수 (Q&A 생성)
CLASSIFY_CONCURRENCY = 10  # 동시 처리 워커 수 (공지 분류)
PASS_RATE_ESTIMATE = 0.85  # 검수 통과율 가정 (이전 500개 생성에서 약 88%)

MIN_TEXT_CHARS = 200  # 이보다 짧은 공지는 쓰지 않음
MAX_NOTICE_CHARS = 6000  # 공지 하나에서 모델에 전달하는 최대 글자 수
MAX_SYLLABUS_CHARS = 9000  # 강의계획서 하나에서 전달하는 최대 글자 수
ALLOW_OCR_FLAGGED = False  # True면 OCR 검토 권장 표시가 붙은 문서도 사용
TEST_CLASSIFY_LIMIT = 60  # 테스트 모드에서 분류할 공지 수
SYLLABUS_PAIRS_PER_VIEW = 4  # 강의계획서 한 관점당 질문 수
ABSTAIN_DOC_SAMPLES = 130  # 문서 부재형 예시를 만들 문서 수
ABSTAIN_DOC_PAIRS = 2  # 문서 하나당 문서 부재형 질문 수
ABSTAIN_SCOPE_PAIRS = 5  # 범위 밖 주제 하나당 질문 수

CATEGORIES = [
    "교육과정·전공",
    "졸업·학적",
    "수강·학사일정",
    "수업·학생 운영",
    "장학·지원",
    "외부행사",
    "교직과정",
    "대학원·진학",
]
NONE_LABEL = "무관"
ABSTAIN_LABEL = "자료없음"
BUCKETS = CATEGORIES + [ABSTAIN_LABEL]  # 결과 카운트·진행 관리 단위

CATEGORY_GUIDE = {
    "교육과정·전공": "교육과정 개편, 전공 이수 체계, 교과목 개설·구분 변경, 복수·부전공, 트랙, 선후수 체계",
    "졸업·학적": "졸업요건, 졸업 신청·대상자, 학위수여, 휴·복학, 편입·전과·전입, 학적 변동, 학점 인정",
    "수강·학사일정": "수강신청·정정·취소, 계절수업, 시험·성적, 학사일정",
    "수업·학생 운영": "출석 인정, 실습·현장실습, 수업 운영, 학과 행정, 상담·조교, 시설·설문, 학과 주최 행사·특강",
    "장학·지원": "교내외 장학금(외부 재단·기관의 장학생 선발 포함), 학자금·등록금, 생활비·근로, 학생 지원비와 지원 사업",
    "외부행사": "공모전, 봉사활동, 채용·인턴, 외부 기관의 대회·교육·설명회·행사",
    "교직과정": "교직 이수 신청·예정자 선발, 교직 적성·인성검사, 교육실습, 교원자격, 교직과정 이수 현황",
    "대학원·진학": "본교 대학원 입학·진학설명회·오픈랩, 학·석사 연계과정, 대학원 교과목, 대학원 진학 안내",
}

# 분류 기준이 겹치는 경우를 정하는 규칙 (분류 프롬프트에 들어간다)
CLASSIFY_TIEBREAK_RULES = """겹칠 때의 판단 기준:
- 공모전, 대회, 봉사활동, 채용, 인턴, 외부 설명회·교육은 상금이나 장학금이 걸려 있어도 '외부행사'입니다.
- 외부 재단·기업·지자체의 '장학생 선발', '학자금 지원', '생활비·근로 지원'이 핵심이면 '장학·지원'입니다.
- 수강신청·정정·취소, 계절수업, 시험·성적, 학사일정은 '수강·학사일정'입니다.
- 교과구분 변경, 복수·부전공, 선후수 규정, 교육과정 개편은 '교육과정·전공'입니다.
- 졸업요건·졸업 신청·학위수여·휴복학·편입은 '졸업·학적'입니다.
- 출석 인정, 현장실습 참여 신청, 조교, 시설, 설문, 학과가 여는 특강·행사는 '수업·학생 운영'입니다.
- 교직 이수, 교직 적성·인성검사, 교육실습, 교원자격 관련은 '교직과정'입니다.
- 본교 대학원의 입학·진학설명회·오픈랩·학석사연계는 '대학원·진학'입니다. 다른 기관의 대학원·교육 홍보는 '외부행사'입니다.
- 외부 기관이 여는 행사는 '외부행사', 학과·학교가 학생에게 안내하는 행정은 위 학사 범주로 분류합니다."""

SYLLABUS_VIEWS = [
    (
        "교육과정·전공",
        "교과목 정보와 교육과정 (개설 학기, 학점, 수강 학년, 강의 내용, 선수과목, 담당 교수)",
    ),
    (
        "수업·학생 운영",
        "수업 운영과 평가 (출석, 평가 비율, 과제, 시험, 수업 방식, 강의 시간·강의실)",
    ),
]

# 학과 공지사항·강의계획서로는 답할 수 없는 주제 (범위 밖 거절 예시용)
OUT_OF_SCOPE_TOPICS = [
    "오늘·내일의 날씨, 미세먼지, 교통 혼잡 같은 실시간 정보",
    "학생식당·구내식당의 오늘 메뉴와 가격",
    "셔틀버스, 시내버스 시간표와 실시간 운행 정보",
    "도서관 좌석, 열람실 혼잡도 같은 실시간 현황",
    "다른 대학교의 학사 규정, 수강신청, 장학금",
    "2030학년도처럼 아직 공지되지 않은 먼 미래의 학사일정과 수강신청 일정",
    "교수님의 개인 휴대전화 번호, 집 주소, 사생활",
    "질문한 학생 본인의 성적, 학점, 수강신청 결과, 장학금 지급 여부 조회",
    "특정 학생의 개인정보, 학번, 연락처",
    "기숙사 호실 배정 결과나 룸메이트 정보",
    "취업 합격 가능성 예측, 기업별 연봉과 면접 후기",
    "대학원 입학 합격 커트라인과 합격 가능성",
    "다른 단과대학(의대, 경영대 등) 학과의 교육과정과 졸업요건",
    "연애, 가족, 건강, 심리 상담 같은 개인 고민",
    "과제나 코딩 숙제를 대신 풀어 달라는 요청과 시험 정답",
    "특정 수업의 내년 개설 여부와 담당 교수 배정 확정 정보",
    "학교 근처 맛집, 카페, 자취방 시세",
    "학과 사무실 직원의 근무 일정과 개인 일정",
    "다음 학기 등록금의 정확한 금액 확정 정보",
    "주식, 부동산, 가상화폐 같은 투자 상담",
    "병역, 비자, 해외 체류 같은 법률·행정 상담",
    "동아리 회비, 학생회 내부 회의 내용",
    "특정 교수님의 연구실 위치와 상담 가능 시간 확정 정보",
    "방학 중 학교 건물 출입 가능 시간과 정확한 개방 일정",
    "학교 밖 학원, 인터넷 강의 추천과 후기",
]

VALIDATED_PATH = OUTPUT_DIR / "synthetic_qa_dataset.jsonl"
REJECTED_PATH = OUTPUT_DIR / "rejected_qa_dataset.jsonl"
PROGRESS_PATH = OUTPUT_DIR / "progress.txt"
CLASSIFY_CACHE_PATH = OUTPUT_DIR / "notice_categories.jsonl"
EXCLUDE_PATH = OUTPUT_DIR / "exclude_post_ids.txt"
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 범주별 누적·시도 카운트, 범주별 현재 라운드 절대 목표
passed_counts = {c: 0 for c in BUCKETS}
attempted = {c: 0 for c in BUCKETS}
quotas = {c: 0 for c in BUCKETS}
seen_questions: set[str] = set()


def configure_paths(test_mode: bool) -> None:
    """테스트 모드면 결과 파일 이름 앞에 test_를 붙여 본 실행 결과와 섞이지 않게 한다."""
    global VALIDATED_PATH, REJECTED_PATH
    prefix = "test_" if test_mode else ""
    VALIDATED_PATH = OUTPUT_DIR / f"{prefix}synthetic_qa_dataset.jsonl"
    REJECTED_PATH = OUTPUT_DIR / f"{prefix}rejected_qa_dataset.jsonl"


class QAPair(BaseModel):
    instruction: str = Field(description="학생이 질문하는 프롬프트")
    output: str = Field(description="AI 튜터의 상세하고 친절한 답변")


class QADataset(BaseModel):
    pairs: list[QAPair]


class JudgeResult(BaseModel):
    is_pass: bool = Field(alias="pass", description="통과 여부 (true/false)")
    reason: str = Field(description="판단 이유 한 줄")


class JudgeBatchResult(BaseModel):
    results: list[JudgeResult]


class ClassifyResult(BaseModel):
    category: str = Field(description="범주 이름 하나 (정해진 목록 중)")


async def call_with_retry(chain, payload, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await chain.ainvoke(payload)
        except Exception as e:
            msg = str(e).lower()
            if "credit balance" in msg:
                logger.error("API 크레딧 소진! 충전 후 다시 실행하세요.")
                sys.exit(0)
            if any(k in msg for k in ["429", "529", "503", "overloaded", "rate_limit"]):
                wait = min(5 * (2**attempt), 60)
                logger.warning(f"API 과부하. {wait}초 대기 후 재시도...")
                await asyncio.sleep(wait)
            else:
                raise e
    raise RuntimeError("API 재시도 횟수 초과")


# ---------------------------------------------------------
# 원본 읽기
# ---------------------------------------------------------
def normalize_question(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.lower())


def get_meta(record: dict) -> dict:
    """공지·강의계획서 레코드는 제목·날짜 등이 metadata 안에 있다."""
    meta = record.get("metadata")
    return meta if isinstance(meta, dict) else record


def load_exclude_ids() -> set[str]:
    if not EXCLUDE_PATH.exists():
        return set()
    with open(EXCLUDE_PATH, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_notice_posts(exclude_ids: set[str]) -> list[dict]:
    """공지 레코드를 post_id 단위로 합친다. (본문·첨부·OCR이 같은 공지의 레코드)"""
    posts: dict[str, dict] = {}
    for fp in sorted(DATA_DIR.glob("**/*.jsonl")):
        if "notices" not in fp.name.lower():
            continue
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                meta = get_meta(rec)
                post_id = str(meta.get("post_id") or rec.get("document_id") or "").strip()
                text = str(rec.get("text") or rec.get("content") or "").strip()
                if not post_id or not text:
                    continue
                post = posts.setdefault(
                    post_id,
                    {
                        "post_id": post_id,
                        "title": str(meta.get("title") or "").strip(),
                        "date": str(meta.get("date") or "").strip(),
                        "docs": [],
                        "source": fp.name,
                    },
                )
                post["docs"].append(
                    {
                        "text": text,
                        "flags": str(meta.get("quality_flags") or ""),
                        "type": str(meta.get("source_type") or ""),
                        "hash": rec.get("content_hash") or hashlib.md5(text.encode()).hexdigest(),
                    }
                )

    order = {"body": 0, "attachment": 1, "image_ocr": 2}
    result = []
    for post in posts.values():
        if post["post_id"] in exclude_ids:
            continue
        docs = post["docs"]
        if not ALLOW_OCR_FLAGGED:
            docs = [d for d in docs if "ocr_review_recommended" not in d["flags"]]
        seen_hash, parts = set(), []
        for d in sorted(docs, key=lambda d: order.get(d["type"], 3)):
            if d["hash"] in seen_hash:
                continue
            seen_hash.add(d["hash"])
            parts.append(d["text"])
        body = "\n---\n".join(parts)[:MAX_NOTICE_CHARS]
        if len(body) < MIN_TEXT_CHARS or not post["title"]:
            continue
        post["body"] = body
        result.append(post)
    return result


def load_syllabi() -> list[dict]:
    syllabi = []
    for fp in sorted(DATA_DIR.glob("**/*.jsonl")):
        if "syllabus" not in fp.name.lower():
            continue
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                meta = get_meta(rec)
                text = str(rec.get("text") or rec.get("content") or "").strip()
                if len(text) < MIN_TEXT_CHARS:
                    continue
                syllabi.append(
                    {
                        "post_id": str(meta.get("post_id") or rec.get("document_id") or ""),
                        "name": str(meta.get("course_name") or meta.get("title") or ""),
                        "code": str(meta.get("course_code") or ""),
                        "instructor": str(meta.get("instructor") or ""),
                        "year": str(meta.get("year") or ""),
                        "semester": str(meta.get("semester") or ""),
                        "body": text[:MAX_SYLLABUS_CHARS],
                        "source": fp.name,
                    }
                )
    return syllabi


# ---------------------------------------------------------
# 공지 범주 분류
# ---------------------------------------------------------
def load_classify_cache() -> dict[str, str]:
    cache: dict[str, str] = {}
    if CLASSIFY_CACHE_PATH.exists():
        with open(CLASSIFY_CACHE_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    cache[row["post_id"]] = row["category"]
                except (json.JSONDecodeError, KeyError):
                    pass
    return cache


async def classify_posts(posts: list[dict]) -> dict[str, str]:
    cache = load_classify_cache()
    todo = [p for p in posts if p["post_id"] not in cache]
    logger.info(f"공지 분류: 전체 {len(posts)}개 중 새로 분류할 공지 {len(todo)}개")
    if not todo:
        return cache

    guide = "\n".join(f"- {c}: {g}" for c, g in CATEGORY_GUIDE.items())
    llm = ChatAnthropic(model="claude-haiku-5-5", max_tokens=200)
    parser = JsonOutputParser(pydantic_object=ClassifyResult)
    system_prompt = (
        "당신은 대학 학과 공지사항을 분류하는 분류기입니다.\n"
        "공지의 주된 내용에 맞는 범주 하나를 고르세요.\n"
        + guide
        + f"\n- {NONE_LABEL}: 위 어디에도 맞지 않음 (광고, 일반 홍보 등)\n"
        + CLASSIFY_TIEBREAK_RULES
        + "\n범주 이름은 위 목록에 있는 그대로만 사용하세요.\n{format_instructions}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "제목: {title}\n게시일: {date}\n본문 일부:\n{body}")]
    )
    chain = prompt | llm | parser
    sem = asyncio.Semaphore(CLASSIFY_CONCURRENCY)
    valid = set(CATEGORIES) | {NONE_LABEL}

    async def one(post: dict) -> None:
        async with sem:
            try:
                res = await call_with_retry(
                    chain,
                    {
                        "title": post["title"],
                        "date": post["date"],
                        "body": post["body"][:700],
                        "format_instructions": parser.get_format_instructions(),
                    },
                )
                category = str(res.get("category", "")).strip()
            except Exception as e:
                logger.warning(f"분류 실패({post['post_id']}): {e}")
                return
            if category not in valid:
                category = NONE_LABEL
            cache[post["post_id"]] = category
            with open(CLASSIFY_CACHE_PATH, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"post_id": post["post_id"], "category": category}, ensure_ascii=False
                    )
                    + "\n"
                )

    await asyncio.gather(*(one(p) for p in todo))
    return cache


# ---------------------------------------------------------
# 묶음(생성 단위) 만들기와 범주별 목표 나누기
# ---------------------------------------------------------
def notice_pairs(body_len: int) -> int:
    if body_len < 800:
        return 3
    if body_len < 2000:
        return 4
    return 5


def build_items(
    posts: list[dict], categories: dict[str, str], syllabi: list[dict]
) -> dict[str, list]:
    items: dict[str, list] = {c: [] for c in BUCKETS}
    for p in posts:
        cat = categories.get(p["post_id"])
        if cat not in CATEGORIES:
            continue
        head = f"[공지사항 | 제목: {p['title']} | 게시일: {p['date']} | 분류: {cat}]\n"
        items[cat].append(
            {
                "key": f"notice::{p['post_id']}::{cat}",
                "kind": "공지사항",
                "text": head + p["body"],
                "source": p["source"],
                "post_id": p["post_id"],
                "date": p["date"],
                "n_pairs": notice_pairs(len(p["body"])),
                "focus": CATEGORY_GUIDE[cat],
            }
        )
    for s in syllabi:
        head = (
            f"[강의계획서 | 과목명: {s['name']} | 강좌번호: {s['code']} | 교수: {s['instructor']}"
            f" | 개설: {s['year']}학년도 {s['semester']}]\n"
        )
        for cat, focus in SYLLABUS_VIEWS:
            items[cat].append(
                {
                    "key": f"syllabus::{s['post_id']}::{cat}",
                    "kind": "강의계획서",
                    "text": head + s["body"],
                    "source": s["source"],
                    "post_id": s["post_id"],
                    # 강의계획서 원본에는 post_id가 없어 정답 문서를 강좌번호로 찾는다(build_rag_sft.py).
                    "course_code": s["code"],
                    "date": f"{s['year']}-09-01" if s["year"] else "",
                    "n_pairs": SYLLABUS_PAIRS_PER_VIEW,
                    "focus": focus,
                }
            )
    for cat in CATEGORIES:  # 최신 문서부터 사용
        random.Random(42).shuffle(items[cat])
        items[cat].sort(key=lambda i: i["date"], reverse=True)

    items[ABSTAIN_LABEL] = build_abstain_items(items)
    return items


def build_abstain_items(items: dict[str, list]) -> list[dict]:
    """거절 예시용 묶음: 문서 부재형과 범위 밖형을 번갈아 배치한다."""
    docs = [i for c in CATEGORIES for i in items[c]]
    random.Random(7).shuffle(docs)
    doc_items = []
    for d in docs[:ABSTAIN_DOC_SAMPLES]:
        doc_items.append(
            {
                "key": f"abstain_doc::{d['key']}",
                "kind": "문서부재",
                "text": d["text"],
                "source": d["kind"],
                "post_id": d["post_id"],
                "course_code": d.get("course_code", ""),
                "date": d["date"],
                "n_pairs": ABSTAIN_DOC_PAIRS,
                "focus": "",
            }
        )
    scope_items = []
    for idx, topic in enumerate(OUT_OF_SCOPE_TOPICS):
        scope_items.append(
            {
                "key": f"abstain_scope::{idx}",
                "kind": "범위밖",
                "text": topic,
                "source": "범위 밖",
                "post_id": "",
                "date": "",
                "n_pairs": ABSTAIN_SCOPE_PAIRS,
                "focus": "",
            }
        )
    mixed = []
    for i in range(max(len(doc_items), len(scope_items))):
        if i < len(scope_items):
            mixed.append(scope_items[i])
        if i < len(doc_items):
            mixed.append(doc_items[i])
    return mixed


def compute_quotas(capacity: dict[str, int], total: int) -> dict[str, int]:
    """양이 적은 범주는 가능한 만큼만, 남는 몫은 나머지 범주가 균등하게 나눠 갖는다."""
    result = {c: 0 for c in CATEGORIES}
    active = [c for c in CATEGORIES if capacity[c] > 0]
    remaining = total
    while active:
        share = remaining / len(active)
        small = [c for c in active if capacity[c] <= share]
        if not small:
            base = int(share)
            for c in active:
                result[c] = min(base, capacity[c])
            leftover = remaining - sum(result[c] for c in active)
            for c in active:
                if leftover <= 0:
                    break
                if result[c] < capacity[c]:
                    result[c] += 1
                    leftover -= 1
            break
        for c in small:
            result[c] = capacity[c]
            remaining -= capacity[c]
            active.remove(c)
    return result


def source_done(cat: str) -> bool:
    if TEST_MODE:
        return attempted[cat] >= 1
    return passed_counts[cat] >= quotas[cat]


# ---------------------------------------------------------
# Q&A 생성과 검수
# ---------------------------------------------------------
async def generate_synthetic_data_async(item: dict, category: str, num_pairs: int) -> list[dict]:
    llm = ChatAnthropic(model="claude-sonnet-5-5", max_tokens=8192)
    parser = JsonOutputParser(pydantic_object=QADataset)
    system_prompt = """당신은 대학 학사 안내 AI 튜터를 위한 고품질 Instruction 데이터셋 생성 전문가입니다.
아래 [참고 문서]는 '{category}' 범주({focus})의 {kind}입니다.
경북대학교 컴퓨터학부 학생이 챗봇에 물어볼 법한 질문 {num_pairs}개와 AI 튜터의 답변을 만드세요.

규칙:
1. 답변은 반드시 [참고 문서]에 있는 내용만 사용합니다. 문서에 없는 날짜, 금액, 조건, 링크는 만들지 마세요.
2. 문서가 특정 연도, 학기, 기간에 해당하면 질문에 그 연도·학기를 자연스럽게 포함하고(예: "2026학년도 2학기 ..."), 답변에도 해당 기간과 마감일을 정확히 적으세요. 머리말의 게시일을 참고하세요.
3. 질문 유형을 서로 다르게 하세요. (예: 대상, 일정·마감, 신청 방법, 제출 서류, 조건·기준, 유의사항)
4. 문서의 문장을 그대로 베끼지 말고 학생의 말투로 질문하고, 답변은 친절한 해요체로 2~5문장으로 쓰세요.
5. 질문만 읽고도 무엇에 대한 질문인지 알 수 있어야 합니다. "이 공지", "이 수업"처럼 문서를 가리키는 표현은 쓰지 마세요. 강의계획서는 과목명을 질문에 넣으세요.
6. 문서에 물어볼 만한 내용이 부족하면 질문 개수를 줄여도 됩니다.
{format_instructions}"""
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "[참고 문서]:\n{context}")]
    )
    chain = prompt | llm | parser

    try:
        res = await call_with_retry(
            chain,
            {
                "context": item["text"],
                "category": category,
                "focus": item["focus"],
                "kind": item["kind"],
                "num_pairs": num_pairs,
                "format_instructions": parser.get_format_instructions(),
            },
        )
        return res.get("pairs", [])
    except Exception as e:
        logger.error(f"생성 중 오류: {e}")
        return []


async def generate_abstention_async(item: dict, num_pairs: int) -> list[dict]:
    """자료가 없는 질문에 '공지사항·강의계획서에 자료가 없어 모른다'고 답하는 예시를 만든다."""
    llm = ChatAnthropic(model="claude-sonnet-5-5", max_tokens=8192)
    parser = JsonOutputParser(pydantic_object=QADataset)
    if item["kind"] == "문서부재":
        system_prompt = """당신은 대학 학사 안내 AI 튜터를 위한 Instruction 데이터셋 생성 전문가입니다.
아래 [참고 문서]와 같은 주제이지만, 문서에는 **나와 있지 않은** 세부 정보를 묻는 학생 질문 {num_pairs}개와 AI 튜터의 답변을 만드세요.
(예: 문서에 일정은 있는데 금액이 없다면 금액을 묻기, 신청 방법만 있다면 면제 조건을 묻기)

규칙:
1. 질문은 [참고 문서]를 아무리 읽어도 답을 찾을 수 없어야 합니다. 문서에 있는 내용을 묻지 마세요.
2. 답변은 먼저 "제가 가진 공지사항과 강의계획서 자료에는 ○○ 내용이 없어서 정확히 알려드리기 어려워요."처럼 자료에 없다고 분명히 말하세요. 날짜, 금액, 조건을 추측해서 말하지 마세요.
3. 문서에서 확인되는 관련 사실이 있으면 1~2문장으로 알려 주고, 정확한 내용은 학과 사무실이나 해당 담당 부서에 문의하라고 안내하세요.
4. 질문에 연도·학기·공지 제목이나 과목명을 넣어 무엇에 대한 질문인지 알 수 있게 하세요. "이 공지", "이 수업" 같은 표현은 쓰지 마세요.
5. 답변은 친절한 해요체로 2~4문장으로 쓰세요.
{format_instructions}"""
        context = item["text"]
        human = "[참고 문서]:\n{context}"
    else:
        system_prompt = """당신은 대학 학사 안내 AI 튜터를 위한 Instruction 데이터셋 생성 전문가입니다.
이 AI 튜터는 경북대학교 컴퓨터학부의 공지사항과 강의계획서만 근거로 답합니다.
아래 [주제]는 그 자료로는 답할 수 없는 영역입니다. 이 주제에 대해 학생이 챗봇에 물어볼 법한 질문 {num_pairs}개와 AI 튜터의 답변을 만드세요.

규칙:
1. 질문은 서로 다른 표현과 상황으로 만드세요. 학생의 자연스러운 말투를 쓰세요.
2. 답변은 반드시 "죄송해요. 제가 가진 학과 공지사항과 강의계획서 자료에는 해당 내용이 없어서 정확히 알려드리기 어려워요."처럼 자료에 없어 모른다고 분명히 말하는 문장으로 시작하세요.
3. 이어서 어디에서 확인하면 좋은지 일반적인 방법만 한두 문장으로 안내하세요. (예: 학교 홈페이지, 학과 사무실, 해당 담당 부서) 구체적인 사실, 숫자, 연락처를 만들어 내지 마세요.
4. 질문에 답을 추측해서 말하지 마세요.
5. 답변은 친절한 해요체로 2~3문장으로 쓰세요.
{format_instructions}"""
        context = item["text"]
        human = "[주제]:\n{context}"

    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human)])
    chain = prompt | llm | parser
    try:
        res = await call_with_retry(
            chain,
            {
                "context": context,
                "num_pairs": num_pairs,
                "format_instructions": parser.get_format_instructions(),
            },
        )
        return res.get("pairs", [])
    except Exception as e:
        logger.error(f"거절 예시 생성 중 오류: {e}")
        return []


async def validate_qa_batch_async(qa_pairs: list[dict], context_text: str) -> list[dict]:
    llm = ChatAnthropic(model="claude-haiku-5-5", max_tokens=4096)
    parser = JsonOutputParser(pydantic_object=JudgeBatchResult)
    system_prompt = """당신은 Q&A 데이터의 품질을 검증하는 엄격한 평가자입니다.
아래 [참고 문서]만을 근거로 각 Q&A를 평가하세요.
1. 답변이 참고 문서에 실제로 있는 사실에 기반하는가? (없는 내용을 지어냈다면 실패)
2. 질문과 답변이 논리적으로 맞는가?
3. 왜곡되거나 오해의 소지가 있는 정보가 없는가?
4. 답변의 날짜, 기간, 연도, 학기가 문서와 일치하는가?
5. 질문만 읽고 무엇을 묻는지 알 수 있는가? ("이 공지", "이 수업"처럼 문서를 가리키는 표현이 있으면 실패)
입력과 같은 순서와 개수로 평가 결과를 반환하세요.
{format_instructions}"""
    human_prompt = "[참고 문서]:\n{context}\n\n[평가할 Q&A 리스트]:\n{pairs}"
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
    chain = prompt | llm | parser

    try:
        res = await call_with_retry(
            chain,
            {
                "context": context_text,
                "pairs": json.dumps(qa_pairs, ensure_ascii=False, indent=2),
                "format_instructions": parser.get_format_instructions(),
            },
        )
        return res.get("results", [])
    except Exception as e:
        logger.warning(f"검증 중 오류: {e}")
        return [None] * len(qa_pairs)


async def validate_abstention_batch_async(qa_pairs: list[dict], item: dict) -> list[dict]:
    llm = ChatAnthropic(model="claude-haiku-5-5", max_tokens=4096)
    parser = JsonOutputParser(pydantic_object=JudgeBatchResult)
    if item["kind"] == "문서부재":
        system_prompt = """당신은 '자료 없음' 거절 Q&A의 품질을 검증하는 엄격한 평가자입니다.
아래 [참고 문서]를 기준으로 각 Q&A를 평가하세요. 모두 만족해야 통과입니다.
1. 질문이 묻는 정보가 [참고 문서]에 실제로 없는가? (문서에 답이 있으면 실패)
2. 답변이 자료에 없어서 정확히 알려드리기 어렵다고 분명히 말하는가?
3. 답변이 날짜, 금액, 조건 등을 추측하거나 지어내지 않았는가? (문서에서 확인되는 관련 사실만 언급해야 함)
4. "이 공지", "이 수업"처럼 문서를 가리키는 표현이 질문에 없는가?
입력과 같은 순서와 개수로 평가 결과를 반환하세요.
{format_instructions}"""
        human_prompt = "[참고 문서]:\n{context}\n\n[평가할 Q&A 리스트]:\n{pairs}"
    else:
        system_prompt = """당신은 '자료 없음' 거절 Q&A의 품질을 검증하는 엄격한 평가자입니다.
이 AI 튜터는 대학 학과의 공지사항과 강의계획서만 근거로 답합니다. 아래 [주제]는 그 자료로 답할 수 없는 영역입니다.
각 Q&A를 평가하세요. 모두 만족해야 통과입니다.
1. 질문이 [주제]와 관련이 있고, 학과 공지사항이나 강의계획서로는 답할 수 없는 내용인가?
2. 답변이 자료에 없어 모른다고 분명히 말하며 시작하는가?
3. 답변이 구체적인 사실, 숫자, 연락처를 지어내거나 질문에 대한 답을 추측하지 않았는가?
4. 어디서 확인하면 좋은지 안내가 일반적인 수준인가?
입력과 같은 순서와 개수로 평가 결과를 반환하세요.
{format_instructions}"""
        human_prompt = "[주제]:\n{context}\n\n[평가할 Q&A 리스트]:\n{pairs}"
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
    chain = prompt | llm | parser
    try:
        res = await call_with_retry(
            chain,
            {
                "context": item["text"],
                "pairs": json.dumps(qa_pairs, ensure_ascii=False, indent=2),
                "format_instructions": parser.get_format_instructions(),
            },
        )
        return res.get("results", [])
    except Exception as e:
        logger.warning(f"거절 예시 검증 중 오류: {e}")
        return [None] * len(qa_pairs)


async def process_batch(item: dict, cat: str, done_keys: set) -> None:
    record_key = item["key"]

    if not TEST_MODE and record_key in done_keys:
        return
    if source_done(cat):
        return

    attempted[cat] += 1
    num_pairs = 1 if TEST_MODE else item["n_pairs"]
    is_abstain = cat == ABSTAIN_LABEL

    logger.info(
        f"[{cat}] API 요청 중... ({item['kind']}, 현재 {passed_counts[cat]}/{quotas[cat]}, "
        f"전체 {sum(passed_counts.values())})"
    )

    if is_abstain:
        pairs = await generate_abstention_async(item, num_pairs=num_pairs)
    else:
        pairs = await generate_synthetic_data_async(item, cat, num_pairs=num_pairs)
    if not pairs:
        return

    if is_abstain:
        results = await validate_abstention_batch_async(pairs, item)
    else:
        results = await validate_qa_batch_async(pairs, item["text"])
    if len(results) != len(pairs) or any(r is None for r in results):
        logger.warning(f"[{cat}] 검증 실패 및 누락 - 묶음 스킵")
        return

    for pair, res in zip(pairs, results):
        instruction, output = (
            (pair.get("instruction") or "").strip(),
            (pair.get("output") or "").strip(),
        )
        if not instruction or not output:
            continue

        is_pass = res.get("pass", res.get("is_pass", False))
        out_data = {
            "category": cat,
            "source": item["source"] if is_abstain else item["kind"],
            "post_id": item["post_id"],
            "course_code": item.get("course_code", ""),
            "date": item["date"],
            "instruction": instruction,
            "output": output,
        }

        if is_pass:
            qkey = normalize_question(instruction)
            if qkey in seen_questions:
                continue  # 같은 질문은 한 번만 저장
            # 본 실행 중이면서 할당량을 다 채웠으면 파일에 안 씀
            if not TEST_MODE and passed_counts[cat] >= quotas[cat]:
                continue
            seen_questions.add(qkey)
            with open(VALIDATED_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(out_data, ensure_ascii=False) + "\n")
            passed_counts[cat] += 1
            if TEST_MODE:
                logger.info(f"테스트 통과]: {instruction[:30]}...")
        else:
            out_data["reject_reason"] = res.get("reason", "알 수 없음")
            with open(REJECTED_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(out_data, ensure_ascii=False) + "\n")

    if not TEST_MODE:
        with open(PROGRESS_PATH, "a", encoding="utf-8") as f:
            f.write(record_key + "\n")
        done_keys.add(record_key)


def pending_items(items: dict[str, list], done_keys: set) -> dict[str, list]:
    return {c: [i for i in items[c] if i["key"] not in done_keys] for c in BUCKETS}


async def run_round(items: dict[str, list], done_keys: set) -> None:
    """남은 문서로 범주를 번갈아 가며 Q&A를 만든다."""

    def interleave():
        queues = {c: iter(items[c]) for c in BUCKETS if items[c]}
        while queues:
            for cat in list(queues):
                if source_done(cat):
                    del queues[cat]
                    continue
                try:
                    item = next(queues[cat])
                except StopIteration:
                    del queues[cat]
                    continue
                yield item, cat

    stream = interleave()

    async def worker():
        for item, cat in stream:
            await process_batch(item, cat, done_keys)

    n_workers = 1 if TEST_MODE else CONCURRENCY
    await asyncio.gather(*(worker() for _ in range(n_workers)))


def load_existing_results() -> None:
    """이어서 실행할 때 이미 저장된 개수와 질문을 읽는다."""
    if TEST_MODE or not VALIDATED_PATH.exists():
        return
    with open(VALIDATED_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cat = row.get("category")
            if cat in passed_counts:
                passed_counts[cat] += 1
            seen_questions.add(normalize_question(row.get("instruction", "")))


def plan_quotas(items: dict[str, list]) -> dict[str, int]:
    """이번 라운드에 범주별로 더 만들 개수를 계산한다. (거절 예시 먼저, 나머지를 범주별로 균등 분배)"""
    pair_capacity = {
        c: int(sum(i["n_pairs"] for i in items[c]) * PASS_RATE_ESTIMATE) for c in BUCKETS
    }
    abstain_left = max(ABSTENTION_TARGET - passed_counts[ABSTAIN_LABEL], 0)
    abstain_extra = min(abstain_left, pair_capacity[ABSTAIN_LABEL])
    total_left = max(TOTAL_TARGET - sum(passed_counts.values()), 0)
    category_extra = compute_quotas(pair_capacity, max(total_left - abstain_extra, 0))

    extra = {**category_extra, ABSTAIN_LABEL: abstain_extra}
    logger.info("범주별 현황 (문서 수 / 예상 생성 가능량 / 이번 목표 추가량)")
    for c in BUCKETS:
        logger.info(
            f"  {c:10s} 문서 {len(items[c]):4d} | 가능 {pair_capacity[c]:4d} | 목표 +{extra[c]:4d}"
        )
    if sum(pair_capacity[c] for c in CATEGORIES) < total_left - abstain_extra:
        logger.warning(
            f"원본 양이 부족해 목표보다 적게 생성될 수 있습니다 (범주 가능량 합계 {sum(pair_capacity[c] for c in CATEGORIES)}개)"
        )
    if abstain_left > pair_capacity[ABSTAIN_LABEL]:
        logger.warning("거절 예시를 목표만큼 만들 수 없습니다. ABSTAIN_DOC_SAMPLES 등을 늘리세요.")
    return extra


async def main():
    global TEST_MODE, TOTAL_TARGET, ABSTENTION_TARGET
    parser = argparse.ArgumentParser(description="학생용 합성 Q&A 생성 (공지사항 + 강의계획서)")
    parser.add_argument("--test", action="store_true", help="범주별 1묶음씩 소량 시험")
    parser.add_argument("--plan", action="store_true", help="범주별 가능량과 목표만 출력하고 종료")
    parser.add_argument("--total", type=int, default=None, help="전체 목표 개수 (거절 예시 포함)")
    parser.add_argument("--abstain", type=int, default=None, help="자료 없음(거절) 예시 개수")
    args = parser.parse_args()
    if args.test:
        TEST_MODE = True
    if args.total:
        TOTAL_TARGET = args.total
    if args.abstain is not None:
        ABSTENTION_TARGET = args.abstain
    configure_paths(TEST_MODE)

    done_keys: set[str] = set()
    if not TEST_MODE and PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            done_keys = set(f.read().splitlines())
    load_existing_results()

    posts = load_notice_posts(load_exclude_ids())
    syllabi = load_syllabi()
    logger.info(
        f"사용 가능한 공지 {len(posts)}개(post_id 기준, OCR 경고·짧은 글 제외), 강의계획서 {len(syllabi)}개"
    )
    if not posts and not syllabi:
        logger.error(
            f"원본이 없습니다. {DATA_DIR} 아래에 *notices*.jsonl, *syllabus*.jsonl이 있는지 확인하세요"
        )
        return

    if TEST_MODE:
        posts = random.Random(42).sample(posts, min(TEST_CLASSIFY_LIMIT, len(posts)))
    categories = await classify_posts(posts)
    counts = defaultdict(int)
    for p in posts:
        counts[categories.get(p["post_id"], "미분류")] += 1
    logger.info(f"공지 분류 결과: {dict(counts)}")

    all_items = build_items(posts, categories, syllabi)

    if args.plan:
        plan_quotas(pending_items(all_items, done_keys))
        return

    if TEST_MODE:
        logger.info("▶️ 테스트 모드: 범주별 1묶음씩만 생성합니다.")
        await run_round(pending_items(all_items, done_keys), done_keys)
        logger.info(f"종료! 누적: {passed_counts}")
        return

    logger.info(
        f"▶️ 본 실행: 전체 목표 {TOTAL_TARGET}개 (거절 예시 {ABSTENTION_TARGET}개 포함) | 누적: {passed_counts}"
    )
    for round_no in range(1, 4):  # 통과율이 낮아 모자라면 남은 문서로 최대 3라운드까지 보충
        total_left = TOTAL_TARGET - sum(passed_counts.values())
        if total_left <= 0:
            break
        pending = pending_items(all_items, done_keys)
        if not any(pending[c] for c in BUCKETS):
            logger.warning("남은 문서가 없습니다.")
            break
        extra = plan_quotas(pending)
        for c in BUCKETS:
            quotas[c] = passed_counts[c] + extra[c]
        before = sum(passed_counts.values())
        logger.info(f"라운드 {round_no} 시작 (남은 목표 {total_left}개)")
        await run_round(pending, done_keys)
        if sum(passed_counts.values()) == before:
            break

    logger.info(f"종료! 최종 누적: {passed_counts} (합계 {sum(passed_counts.values())}개)")


if __name__ == "__main__":
    asyncio.run(main())
