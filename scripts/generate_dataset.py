import os
import sys
import json
import logging
from typing import List
from dotenv import load_dotenv

load_dotenv()

# 프로젝트 루트 폴더(four-leaf-AI)를 파이썬 경로에 추가하여 'app' 모듈을 찾을 수 있도록 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QAPair(BaseModel):
    instruction: str = Field(description="학생이 대학 생활, 학사, 진로 등에 대해 질문하는 프롬프트")
    output: str = Field(description="AI 튜터의 상세하고 친절한 답변")

class QADataset(BaseModel):
    pairs: List[QAPair] = Field(description="생성된 Q&A 쌍의 리스트")

def generate_synthetic_data(context_text: str, num_pairs: int = 5) -> List[dict]:
    """
    주어진 텍스트 컨텍스트를 바탕으로 sLLM 파인튜닝용 
    Instruction-Output 쌍을 생성합니다. (HyperCLOVA X 활용)
    """
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
    parser = JsonOutputParser(pydantic_object=QADataset)
    
    system_prompt = """당신은 대학교 학사 규정, 공지사항, 진로 가이드 등을 기반으로
인공지능 모델 파인튜닝을 위한 고품질 Instruction 데이터셋을 생성하는 전문가입니다.

제공된 [참고 문서]를 읽고, 실제 대학생이 할 법한 질문({num_pairs}개)과 
그에 대한 친절하고 정확한 AI 튜터의 답변을 생성해주세요.
답변은 반드시 [참고 문서]의 내용만을 기반으로 해야 하며, 친절한 해요체를 사용하세요.

형식 지침에 맞게 정확한 JSON을 출력해주세요.
{format_instructions}
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "[참고 문서]:\n{context}")
    ])
    
    chain = prompt | llm | parser
    
    try:
        logger.info(f"데이터 생성 요청 중... (목표: {num_pairs}개)")
        result = chain.invoke({
            "context": context_text,
            "num_pairs": num_pairs,
            "format_instructions": parser.get_format_instructions()
        })
        
        # pydantic 객체 파싱 결과에서 리스트 추출
        pairs = result.get("pairs", [])
        return pairs
        
    except Exception as e:
        logger.error(f"데이터 생성 중 오류 발생: {e}")
        return []

def main():
    # 저장해둔 경북대학교 학사일정 텍스트 파일 읽기
    input_txt_path = os.path.join("data", "knu_schedule.txt")
    output_path = os.path.join("data", "synthetic_dataset.jsonl")
    
    try:
        with open(input_txt_path, 'r', encoding='utf-8') as f:
            context_text = f.read()
    except FileNotFoundError:
        logger.error(f"{input_txt_path} 파일을 찾을 수 없습니다.")
        return
        
    # 데이터 생성 (예: 10쌍 생성)
    qa_pairs = generate_synthetic_data(context_text, num_pairs=10)
    
    # JSONL 형태로 저장 (파인튜닝 포맷)
    if qa_pairs:
        with open(output_path, 'a', encoding='utf-8') as f:
            for pair in qa_pairs:
                json_line = json.dumps({
                    "instruction": pair["instruction"],
                    "output": pair["output"]
                }, ensure_ascii=False)
                f.write(json_line + '\n')
                
        logger.info(f"성공적으로 {len(qa_pairs)}개의 데이터를 {output_path}에 저장했습니다.")
    else:
        logger.warning("데이터 생성에 실패했습니다.")

if __name__ == "__main__":
    main()
