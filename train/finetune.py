"""Phi-3-mini LoRA 파인튜닝.

학습 데이터: scripts/generate_qa_from_notices.py 가 만든 {"instruction", "output"} JSONL

실행 (GPU 노드에서, 프로젝트 루트):
    python train/finetune.py
    python train/finetune.py --epochs 5
    python train/finetune.py --data data/notice_qa_dataset.jsonl data/validated_qa_dataset.jsonl
"""

import argparse
import json
import os
import random
import re

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

DEFAULT_DATA = ["data/notice_qa_dataset.jsonl"]
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
PHI_MODULES = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
LLAMA_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def normalize(text):
    return re.sub(r"[\s\W_]+", "", text.lower())


def load_pairs(paths):
    """JSONL 파일들을 읽어 중복 질문을 제거한 (instruction, output) 목록을 돌려줍니다."""
    pairs = []
    seen = set()
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"학습 데이터가 없습니다: {path}")
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                question = (row.get("instruction") or "").strip()
                answer = (row.get("output") or "").strip()
                key = normalize(question)
                if not question or not answer or key in seen:
                    continue
                seen.add(key)
                pairs.append((question, answer))
                count += 1
        print(f"  {path}: {count}쌍")
    return pairs


def to_messages(pair):
    # SFTTrainer가 모델의 chat template(Phi-3)을 자동 적용합니다.
    question, answer = pair
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--adapter-dir", default="./final_adapter")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", default=MODEL_ID)
    args = parser.parse_args()

    print("=== 1. 모델 설정 및 토크나이저 로드 중 ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # bf16 지원 GPU(A6000 등)면 bf16, 아니면 fp32 로드 + fp16 학습
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"bf16 사용: {use_bf16}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map="auto",
    )

    print("=== 2. LoRA 어댑터 설정 중 ===")
    # Phi-3는 q/k/v가 qkv_proj로, gate/up이 gate_up_proj로 합쳐져 있음
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=PHI_MODULES if "phi" in args.model_id.lower() else LLAMA_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("=== 3. 데이터셋 준비 중 ===")
    pairs = load_pairs(args.data)
    if not pairs:
        raise SystemExit("학습할 데이터가 0쌍입니다.")
    random.Random(args.seed).shuffle(pairs)
    train_dataset = Dataset.from_list([to_messages(p) for p in pairs])
    print(f"총 학습 데이터: {len(train_dataset)}쌍")
    print("샘플:", train_dataset[0]["messages"])

    steps_per_epoch = max(1, len(train_dataset) // (args.batch_size * args.grad_accum))
    print(f"예상 스텝: 에폭당 약 {steps_per_epoch}, 전체 약 {int(steps_per_epoch * args.epochs)}")

    print("=== 4. SFTConfig 및 SFTTrainer 설정 중 ===")
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        max_length=args.max_length,  # 구버전 trl이면 max_seq_length
        bf16=use_bf16,
        fp16=not use_bf16,
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        peft_config=peft_config,
        processing_class=tokenizer,  # 구버전 trl이면 tokenizer=tokenizer
    )
    trainer.model.print_trainable_parameters()

    print("=== 5. 파인튜닝 학습 시작 ===")
    trainer.train()

    trainer.model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)
    print(f"=== 학습 완료 및 어댑터 저장 성공! ({args.adapter_dir}) ===")


if __name__ == "__main__":
    main()
