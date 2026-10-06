"""
A100 파인튜닝(SFT) 스크립트 - QLoRA 기반

Hugging Face `TRL`의 `SFTTrainer`와 `peft`를 이용해 대학 특화 데이터셋을 파인튜닝합니다.
<<<<<<< HEAD
빠른 속도와 메모리 최적화를 위해 Unsloth 사용을 권장합니다.
(Unsloth가 설치되어 있지 않은 환경에서는 일반 transformers 방식으로 우회 가능하도록 구조화할 수 있으나,
여기서는 Unsloth 기반의 최신 템플릿을 제공합니다.)
"""

import os

=======
Unsloth 기반 최신 템플릿 및 argparse/Val split 고도화 버전입니다.
"""

import os
import argparse
>>>>>>> ca402a8 (feat: finetune.py 스크립트 고도화)
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# Unsloth 라이브러리가 A100 등 Ampere 아키텍처 이상에서 가장 효율적입니다.
try:
    from unsloth import FastLanguageModel
<<<<<<< HEAD

    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print(
        "WARNING: Unsloth가 설치되지 않았습니다. 일반 Hugging Face 프레임워크로 구동하려면 추가 설정이 필요합니다."
    )
=======
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("WARNING: Unsloth가 설치되지 않았습니다. Unsloth 환경에서 구동해주세요. (pip install unsloth)")


def formatting_prompts_func(examples):
    """
    JSONL 데이터셋의 (instruction, input, output)을 모델 학습용 프롬프트 포맷으로 변환
    """
    instructions = examples["instruction"]
    inputs = examples.get("input", [""] * len(instructions))
    outputs = examples["output"]
    texts = []

    alpaca_prompt_with_input = """아래는 작업을 설명하는 명령어와 맥락입니다. 요청을 적절하게 완료하는 응답을 작성하세요.

### 명령어:
{}

### 맥락:
{}

### 응답:
{}"""

    alpaca_prompt_no_input = """아래는 작업을 설명하는 명령어입니다. 요청을 적절하게 완료하는 응답을 작성하세요.

### 명령어:
{}

### 응답:
{}"""

    for instruction, input_text, output in zip(instructions, inputs, outputs):
        if input_text and str(input_text).strip():
            text = alpaca_prompt_with_input.format(instruction, input_text, output)
        else:
            text = alpaca_prompt_no_input.format(instruction, output)
        texts.append(text)

    return {"text": texts}
>>>>>>> ca402a8 (feat: finetune.py 스크립트 고도화)


def main():
    if not UNSLOTH_AVAILABLE:
        print("Unsloth 환경에서 실행해 주세요. (pip install unsloth)")
        return

<<<<<<< HEAD
    # 1. 모델 로드 설정
    max_seq_length = 2048  # 데이터 길이에 맞게 조절
    model_name = (
        "Bllossom/llama-3.1-Korean-Bllossom-8B"  # 한국어가 잘 지원되는 오픈소스 Llama 3 기반
    )

    print(f"[{model_name}] 모델 로딩 중...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,  # A100 지원 (bf16)
        load_in_4bit=True,  # QLoRA (4bit 양자화 로드)
    )

    # 2. LoRA(PEFT) 어댑터 설정
    # 모델의 전체 가중치를 학습하는 대신, 일부 추가 가중치(LoRA)만 학습하여 VRAM 최적화
=======
    parser = argparse.ArgumentParser(description="A100 QLoRA LLM 파인튜닝 스크립트")
    parser.add_argument("--data_path", type=str, default="data/synthetic_dataset.jsonl", help="학습 데이터 경로")
    parser.add_argument("--model_name", type=str, default="Bllossom/llama-3.1-Korean-Bllossom-8B", help="베이스 모델 ID")
    parser.add_argument("--output_dir", type=str, default="logs/lora_model", help="LoRA 저장 경로")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="최대 시퀀스 길이")
    parser.add_argument("--batch_size", type=int, default=2, help="배치 사이즈")
    parser.add_argument("--epochs", type=int, default=3, help="학습 에포크 수")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="학습률")
    args = parser.parse_args()

    # 1. 모델 로드 설정
    print(f"[{args.model_name}] 모델 로딩 중...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,  # A100 최적화 (bf16)
        load_in_4bit=True,     # QLoRA (4bit 양자화 로드)
    )

    # 2. LoRA(PEFT) 어댑터 설정
>>>>>>> ca402a8 (feat: finetune.py 스크립트 고도화)
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA Rank
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # 최적화된 체크포인팅
    )

<<<<<<< HEAD
    # 3. 데이터셋 준비
    # scripts/generate_dataset.py 에서 만든 JSONL 사용
    dataset_path = "data/synthetic_dataset.jsonl"
    if not os.path.exists(dataset_path):
        print(f"데이터셋을 찾을 수 없습니다: {dataset_path}")
        print("먼저 scripts/generate_dataset.py 를 실행하세요.")
        return

    dataset = load_dataset("json", data_files={"train": dataset_path}, split="train")

    # 채팅 템플릿 포맷 함수 (Llama-3 인스트럭션 형식 등 모델에 맞게 수정 가능)
    alpaca_prompt = """아래는 작업을 설명하는 명령어입니다. 요청을 적절하게 완료하는 응답을 작성하세요.

### 명령어:
{}

### 응답:
{}"""

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        outputs = examples["output"]
        texts = []
        for instruction, output in zip(instructions, outputs):
            text = alpaca_prompt.format(instruction, output)
            texts.append(text)
        return {
            "text": texts,
        }

    # 맵핑 수행
    dataset = dataset.map(formatting_prompts_func, batched=True)
=======
    # 3. 데이터셋 준비 및 Train/Val Split (90% / 10%)
    if not os.path.exists(args.data_path):
        print(f"❌ 데이터셋을 찾을 수 없습니다: {args.data_path}")
        print("먼저 scripts/generate_dataset.py 를 실행하여 데이터셋을 생성해 주세요.")
        return

    raw_dataset = load_dataset("json", data_files={"train": args.data_path}, split="train")
    formatted_dataset = raw_dataset.map(formatting_prompts_func, batched=True)
    
    # Validation Split
    dataset_split = formatted_dataset.train_test_split(test_size=0.1, seed=3407)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    print(f"📊 데이터 분할 완료 - 학습용: {len(train_dataset)}개, 검증용: {len(eval_dataset)}개")
>>>>>>> ca402a8 (feat: finetune.py 스크립트 고도화)

    # 4. 학습 파라미터 (Trainer) 설정
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
<<<<<<< HEAD
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # 긴 문맥 여러개 합치기 옵션
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=3,  # 학습 횟수
            learning_rate=2e-4,
            fp16=False,
            bf16=True,  # A100용
            logging_steps=1,
=======
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=False,
            bf16=True,  # A100 지원
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=10,
>>>>>>> ca402a8 (feat: finetune.py 스크립트 고도화)
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="logs/outputs",
<<<<<<< HEAD
=======
            report_to="none",
>>>>>>> ca402a8 (feat: finetune.py 스크립트 고도화)
        ),
    )

    # 5. 파인튜닝 실행
<<<<<<< HEAD
    print("학습을 시작합니다...")
=======
    print("🔥 파인튜닝 학습을 시작합니다...")
>>>>>>> ca402a8 (feat: finetune.py 스크립트 고도화)
    trainer_stats = trainer.train()
    print(trainer_stats)

    # 6. 모델 저장 (LoRA 가중치만 저장됨)
<<<<<<< HEAD
    save_path = "logs/lora_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"LoRA 모델이 성공적으로 저장되었습니다: {save_path}")

    # (옵션) 나중에 vLLM 서빙을 원한다면 16bit 모델 병합(Merge) 저장도 가능합니다.
    # model.save_pretrained_merged("logs/merged_model", tokenizer, save_method = "merged_16bit")


if __name__ == "__main__":
    main()
=======
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ LoRA 모델이 성공적으로 저장되었습니다: {args.output_dir}")


if __name__ == "__main__":
    main()
>>>>>>> ca402a8 (feat: finetune.py 스크립트 고도화)
