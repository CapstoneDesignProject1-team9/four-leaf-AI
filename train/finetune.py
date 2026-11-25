"""Llama-3.1-8B-Instruct RAG LoRA 파인튜닝.

학습 데이터: train/build_rag_sft.py 가 만든 {"prompt", "completion"} JSONL
(서빙과 같은 프롬프트 + 검색 자료). 예전 {"instruction", "output"} JSONL도 읽는다.

- 손실은 답변(completion) 토큰에만 건다. 긴 검색 자료를 외우는 데 용량을 쓰지 않는다.
- 에폭마다 검증 손실을 재서 가장 좋은 체크포인트를 어댑터로 저장한다.
- rank는 Modal vLLM의 --max-lora-rank(16)를 넘으면 서빙할 수 없다.

실행 (GPU 노드에서, 프로젝트 루트):
    python train/finetune.py
    python train/finetune.py --epochs 2 --adapter-dir ./adapter_student_llama
"""

import argparse
import json
import random
import re

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_TRAIN = ["data/rag_sft_train.jsonl"]
DEFAULT_EVAL = ["data/rag_sft_val.jsonl"]
MAX_SERVED_RANK = 16  # modal_app.py --max-lora-rank


def normalize(text):
    return re.sub(r"[\s\W_]+", "", text.lower())


def load_rows(paths):
    """JSONL들을 prompt/completion 대화 형식으로 읽고, 같은 질문(프롬프트)은 한 번만 남긴다."""
    rows, seen = [], set()
    for path in paths:
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "prompt" not in row:  # 예전 {"instruction", "output"} 형식
                    question = (row.get("instruction") or "").strip()
                    answer = (row.get("output") or "").strip()
                    if not question or not answer:
                        continue
                    row = {
                        "prompt": [{"role": "user", "content": question}],
                        "completion": [{"role": "assistant", "content": answer}],
                    }
                key = normalize(row["prompt"][-1]["content"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"prompt": row["prompt"], "completion": row["completion"]})
                count += 1
        print(f"  {path}: {count}개")
    return rows


def drop_too_long(rows, tokenizer, max_length):
    """잘리면 답변 끝이 사라져 잘못 배우므로, 길이를 넘는 예시는 빼고 개수를 알린다."""
    kept = [
        r
        for r in rows
        if len(tokenizer.apply_chat_template(r["prompt"] + r["completion"])) <= max_length
    ]
    if len(kept) < len(rows):
        print(f"  max_length({max_length}) 초과로 {len(rows) - len(kept)}개 제외")
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", default=DEFAULT_TRAIN)
    parser.add_argument("--eval-data", nargs="*", default=DEFAULT_EVAL, help="비우면 검증 생략")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--adapter-dir", default="./adapter_student_llama")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", default=MODEL_ID)
    args = parser.parse_args()
    if args.rank > MAX_SERVED_RANK:
        raise SystemExit(f"rank {args.rank}는 vLLM max-lora-rank {MAX_SERVED_RANK}를 넘습니다.")

    print("=== 1. 토크나이저·데이터 준비 ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = drop_too_long(load_rows(args.data), tokenizer, args.max_length)
    eval_rows = drop_too_long(load_rows(args.eval_data), tokenizer, args.max_length)
    if not train_rows:
        raise SystemExit("학습할 데이터가 0개입니다.")
    random.Random(args.seed).shuffle(train_rows)
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None
    print(f"학습 {len(train_rows)}개, 검증 {len(eval_rows)}개")

    print("=== 2. 모델 로드 ===")
    # bf16 지원 GPU(A6000 등)면 bf16, 아니면 fp32 로드 + fp16 학습
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        target_modules="all-linear",  # Llama: q/k/v/o/gate/up/down
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    has_eval = eval_dataset is not None
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,  # 4k 토큰 문맥을 8B 모델로 학습할 때 메모리 절약
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        max_length=args.max_length,
        completion_only_loss=True,
        eval_strategy="epoch" if has_eval else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.model.print_trainable_parameters()

    print("=== 3. 학습 ===")
    trainer.train()
    if has_eval:
        print("최종(최고) 검증 손실:", trainer.evaluate()["eval_loss"])

    trainer.model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)
    print(f"=== 어댑터 저장 완료: {args.adapter_dir} ===")


if __name__ == "__main__":
    main()
