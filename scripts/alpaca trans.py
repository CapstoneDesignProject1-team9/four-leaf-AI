import json
import os

# scripts 폴더의 상위 폴더(four-leaf-AI 루트)를 기준으로 data 폴더 위치 계산
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(BASE_DIR, "output", "professor_qa_dataset.jsonl")  # 실제 파일명으로 수정
OUTPUT_FILE = os.path.join(BASE_DIR, "output", "professor_alpaca_dataset.json")

alpaca_dataset = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        data = json.loads(line)
        
        # -------------------------------------------------------------
        # 2. 키 매핑 (기존 데이터셋의 필드명에 맞게 가져옵니다)
        # -------------------------------------------------------------
        # 기존 데이터가 이미 "instruction", "output"으로 되어있다면 그대로 가져오고,
        # 혹시 "question", "answer" 등으로 되어있다면 알맞게 읽어옵니다.
        instruction_text = data.get("instruction") or data.get("question") or ""
        input_text = data.get("input") or ""  # 없으면 빈 문자열("")
        output_text = data.get("output") or data.get("answer") or ""

        alpaca_item = {
            "instruction": instruction_text,
            "input": input_text,
            "output": output_text
        }
        
        alpaca_dataset.append(alpaca_item)

# 3. Alpaca 포맷 JSON 파일로 저장
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(alpaca_dataset, f, ensure_ascii=False, indent=2)

print(f"✅ 총 {len(alpaca_dataset)}개의 데이터가 '{OUTPUT_FILE}'로 성공적으로 변환되었습니다!")