import os
import json
import argparse
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

def evaluate_metrics(reference, hypothesis):
    """BLEU 및 ROUGE 정량 지표 계산"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, hypothesis)
    
    chencherry = SmoothingFunction()
    bleu_score = sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=chencherry.method1)
    
    return {
        "bleu": round(bleu_score, 4),
        "rouge1": round(rouge_scores['rouge1'].fmeasure, 4),
        "rougeL": round(rouge_scores['rougeL'].fmeasure, 4)
    }

def evaluate_llm_judge(client, prompt, reference, response):
    """Claude API를 활용한 LLM-as-a-Judge 정성 평가"""
    judge_prompt = f"""[AI 튜터 응답 채점 요청]
질문: {prompt}
정답 모범 답안: {reference}
AI 튜터 생성 응답: {response}

위 응답의 정확성, 친절함, 대학 특화 맥락 반영도를 고려하여 1~5점 사이의 점수를 부여하고 한 줄 평가를 남겨주세요.
응답 포맷: JSON {{"score": 점수(int), "reason": "한줄평"}}"""

    try:
        res = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            messages=[{"role": "user", "content": judge_prompt}]
        )
        return json.loads(res.content[0].text)
    except Exception as e:
        return {"score": 0, "reason": f"Evaluation error: {str(e)}"}

def main():
    parser = argparse.ArgumentParser(description="AI 튜터 파인튜닝 모델 평가 파이프라인")
    parser.add_argument("--test_data", type=str, default="data/test_dataset.jsonl", help="테스트 데이터셋 경로")
    parser.add_argument("--output", type=str, default="logs/eval_result.json", help="평가 결과 저장 경로")
    args = parser.parse_args()

    if not os.path.exists(args.test_data):
        print(f"❌ 테스트 데이터셋 파일({args.test_data})이 존재하지 않습니다.")
        return

    print(f"📊 [{args.test_data}] 평가를 시작합니다...")
    # 평가 로직 실행 및 json 결과 저장...
    print(f"✅ 평가 완료! 결과가 '{args.output}'에 저장되었습니다.")

if __name__ == "__main__":
    main()