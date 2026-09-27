import json
import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

INPUT_FILE = "synthetic_qa_dataset.jsonl"
OUTPUT_FILE = "validated_qa_dataset.jsonl"
REJECTED_FILE = "rejected_qa_dataset.jsonl"

# 검증 시 원본 문서 대조용 (합성 때 썼던 문서와 동일하게 맞춰야 함)
source_document = """
[2026학년도 2학기 수강신청 안내]
1. 수강신청 기간: 2026년 8월 10일(월) ~ 8월 14일(금)
2. 대상: 재학생 및 복학생 전체
3. 최대 신청 학점: 19학점 (직전 학기 평점 3.75 이상은 22학점까지 가능)
"""

JUDGE_PROMPT_TEMPLATE = """
당신은 AI가 생성한 Q&A 데이터의 품질을 검증하는 평가자입니다.
아래 [원본 문서]의 내용만을 근거로 [질문]에 대해 [답변]이 정확하고 충실하게 작성되었는지 평가하세요.

[원본 문서]
{document}

[질문]
{instruction}

[답변]
{output}

평가 기준:
1. 답변 내용이 원본 문서에 실제로 있는 내용인가? (문서에 없는 내용을 지어냈다면 실패)
2. 질문과 답변이 논리적으로 맞는 짝인가?
3. 답변이 사실과 다르게 왜곡되지 않았는가?

반드시 아래 JSON 형식으로만 응답하세요:
{{"pass": true 또는 false, "reason": "판단 이유 한 줄"}}
"""


def parse_json_response(text: str):
    """마크다운 펜스가 섞여 있어도 JSON만 추출"""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.removeprefix("json").strip()
    return json.loads(text)


def check_format(item: dict) -> bool:
    """기본 형식 검증: 키 존재 여부, 빈 값 여부"""
    if "instruction" not in item or "output" not in item:
        return False
    if not item["instruction"].strip() or not item["output"].strip():
        return False
    return True


def check_grounding(item: dict, document: str) -> tuple[bool, str]:
    """LLM-as-a-Judge: 원본 문서 근거 기반으로 답변이 타당한지 검증"""
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        document=document,
        instruction=item["instruction"],
        output=item["output"],
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=judge_prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    try:
        result = parse_json_response(response.text)
        return result.get("pass", False), result.get("reason", "")
    except Exception as e:
        return False, f"판정 파싱 실패: {e}"


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"입력 파일이 없습니다: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    passed, rejected = [], []

    for idx, item in enumerate(items):
        if not check_format(item):
            rejected.append({**item, "reject_reason": "형식 오류 (필드 누락/빈 값)"})
            print(f"[{idx}] 형식 오류로 제외")
            continue

        is_pass, reason = check_grounding(item, source_document)
        if is_pass:
            passed.append(item)
            print(f"[{idx}] 통과 — {reason}")
        else:
            rejected.append({**item, "reject_reason": reason})
            print(f"[{idx}] 근거 부족으로 제외 — {reason}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in passed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(REJECTED_FILE, "w", encoding="utf-8") as f:
        for item in rejected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n총 {len(items)}개 중 통과 {len(passed)}개, 제외 {len(rejected)}개")
    print(f"통과 데이터: {OUTPUT_FILE}")
    print(f"제외 데이터(사유 포함): {REJECTED_FILE}")


if __name__ == "__main__":
    main()