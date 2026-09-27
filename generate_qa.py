import json
import os
from google import genai
from google.genai import types

# 1. Gemini API 클라이언트 설정 (환경변수 또는 API 키)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 2. 크롤링 데이터 대신 사용할 임시 더미 데이터 (2번 항목)
sample_document = """
[2026학년도 2학기 수강신청 안내]
1. 수강신청 기간: 2026년 8월 10일(월) ~ 8월 14일(금)
2. 대상: 재학생 및 복학생 전체
3. 최대 신청 학점: 19학점 (직전 학기 평점 3.75 이상은 22학점까지 가능)
"""

# 3. 프롬프트 정의 (1번 항목 반영)
prompt = f"""
다음 제공된 학사 안내 문서를 바탕으로, 학생들이 질문할 법한 [질문(instruction)]과 [답변(output)] 쌍 3개를 생성해줘.

[문서 내용]
{sample_document}

[출력 형식]
반드시 다른 설명 없이 아래와 같은 JSON 배열 형식으로만 응답해줘:
[
  {{"instruction": "수강신청 기간이 언제인가요?", "output": "2026학년도 2학기 수강신청 기간은 8월 10일(월)부터 8월 14일(금)까지입니다."}},
  ...
]
"""

# 4. LLM API 호출 (3번 항목)
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type="application/json"
    ),
)

# 5. JSONL 파일로 저장 (5번 항목)
output_filename = "synthetic_qa_dataset.jsonl"
try:
    # LLM 응답 텍스트를 파이썬 리스트로 변환
    qa_list = json.loads(response.text.strip())
    
    with open(output_filename, "w", encoding="utf-8") as f:
        for item in qa_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"성공적으로 {len(qa_list)}개의 Q&A 데이터를 {output_filename}에 저장했습니다!")

except Exception as e:
    print("오류 발생:", e)