"""교수자 대시보드 시연용 학생 질문을 chat_logs에 넣거나 지운다.

시드 행은 model_name='seed'로 표시되어 실제 기록과 구분된다. 생성 시각은 최근 30일에 고르게 흩어 둔다.

사용법 (four-leaf-AI/에서, DB_* 환경 변수 또는 .env 설정 필요):
    python scripts/seed_chat_logs.py          # 시드 질문 추가(기존 시드는 지우고 다시 넣음)
    python scripts/seed_chat_logs.py --clear  # 시드 질문만 삭제
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.storage.chat_logs import _connect, initialize_chat_log_store  # noqa: E402

SEED_QUESTIONS = {
    "컴퓨터구조": [
        "캐시 직접 매핑이랑 세트 연관 매핑 차이가 뭐야?",
        "캐시 히트율 계산하는 방법 알려줘",
        "캐시 미스 종류(compulsory, conflict, capacity) 설명해줘",
        "write-back이랑 write-through 캐시 차이점이 뭐예요?",
        "2-way 세트 연관 캐시에서 주소를 태그, 인덱스, 오프셋으로 나누는 법 알려줘",
        "파이프라인 데이터 해저드는 포워딩으로 어떻게 해결해?",
        "파이프라인에서 stall이 언제 생기는지 모르겠어",
        "구조적 해저드랑 제어 해저드 차이 알려줘",
        "load-use 해저드는 포워딩으로도 왜 해결이 안 돼?",
        "MIPS 명령어 lw sw 형식이 헷갈려요",
        "MIPS 단일 사이클 데이터패스에서 제어 신호가 어떻게 정해져?",
        "분기 예측이 왜 필요한지 설명해줘",
        "TLB가 가상주소를 물리주소로 바꾸는 과정 알려줘",
        "CPI랑 클럭 속도로 성능 계산하는 문제 어떻게 풀어?",
    ],
    "운영체제": [
        "데드락 발생 4가지 조건이 뭐야?",
        "세마포어랑 뮤텍스 차이점 알려줘",
        "LRU 페이지 교체 알고리즘 구현 방법이 궁금해요",
        "프로세스랑 스레드 차이가 뭐야?",
        "라운드 로빈 스케줄링에서 타임 퀀텀은 어떻게 정해?",
    ],
    "자료구조": [
        "AVL 트리 회전 LL LR 차이 설명해줘",
        "해시 충돌 해결 방법에는 뭐가 있어?",
        "힙 정렬 시간복잡도가 왜 nlogn이야?",
    ],
    "학사": [
        "2026학년도 2학기 휴학 신청 기간 알려줘",
        "수강정정 기간에는 뭘 할 수 있어?",
        "국가장학금 2차 신청은 언제야?",
        "졸업요건 확인서는 언제까지 제출해야 해?",
        "복수전공 신청 자격이 어떻게 돼?",
    ],
}


def clear_seed() -> int:
    with _connect() as connection:
        return connection.execute("DELETE FROM chat_logs WHERE model_name = 'seed'").rowcount


def insert_seed() -> int:
    questions = [q for qs in SEED_QUESTIONS.values() for q in qs]
    with _connect() as connection:
        for i, question in enumerate(questions):
            hours_ago = (i * 30 * 24) // len(questions)
            connection.execute(
                """
                INSERT INTO chat_logs(session_id, question, answer, model_name, status,
                                      created_at, answered_at)
                VALUES (%s, %s, '(시연용 예시 데이터)', 'seed', 'answered',
                        now() - make_interval(hours => %s), now() - make_interval(hours => %s))
                """,
                (f"seed_{i % 7}", question, hours_ago, hours_ago),
            )
    return len(questions)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--clear", action="store_true", help="시드 질문만 삭제")
    args = parser.parse_args()
    initialize_chat_log_store()
    removed = clear_seed()
    if args.clear:
        print(f"시드 질문 {removed}건을 삭제했습니다.")
        return
    print(f"기존 시드 {removed}건을 지우고 {insert_seed()}건을 넣었습니다.")


if __name__ == "__main__":
    main()
