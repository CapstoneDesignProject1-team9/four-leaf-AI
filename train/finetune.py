"""
A100 파인튜닝(SFT) 스크립트 - QLoRA 기반

Hugging Face `TRL`의 `SFTTrainer`와 `peft`를 이용해 대학 특화 데이터셋을 파인튜닝합니다.
빠른 속도와 메모리 최적화를 위해 Unsloth 사용을 권장합니다.
(Unsloth가 설치되어 있지 않은 환경에서는 일반 transformers 방식으로 우회 가능하도록 구조화할 수 있으나,
여기서는 Unsloth 기반의 최신 템플릿을 제공합니다.)
"""

import os

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# Unsloth 라이브러리가 A100 등 Ampere 아키텍처 이상에서 가장 효율적입니다.
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("WARNING: Unsloth가 설치되지 않았습니다. 일반 Hugging Face 프레임워크로 구동하려면 추가 설정이 필요합니다.")

def main():
    if not UNSLOTH_AVAILABLE:
        print("Unsloth 환경에서 실행해 주세요. (pip install unsloth)")
        return

    # 1. 모델 로드 설정
    max_seq_length = 2048 # 데이터 길이에 맞게 조절
    model_name = "Bllossom/llama-3.1-Korean-Bllossom-8B" # 한국어가 잘 지원되는 오픈소스 Llama 3 기반

    print(f"[{model_name}] 모델 로딩 중...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,  # A100 지원 (bf16)
        load_in_4bit=True,     # QLoRA (4bit 양자화 로드)
    )

    # 2. LoRA(PEFT) 어댑터 설정
    # 모델의 전체 가중치를 학습하는 대신, 일부 추가 가중치(LoRA)만 학습하여 VRAM 최적화
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, # LoRA Rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth", # 최적화된 체크포인팅
    )

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
        outputs      = examples["output"]
        texts = []
        for instruction, output in zip(instructions, outputs):
            text = alpaca_prompt.format(instruction, output)
            texts.append(text)
        return { "text" : texts, }

    # 맵핑 수행
    dataset = dataset.map(formatting_prompts_func, batched = True)

    # 4. 학습 파라미터 (Trainer) 설정
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False, # 긴 문맥 여러개 합치기 옵션
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=3,       # 학습 횟수
            learning_rate=2e-4,
            fp16=False,
            bf16=True,                # A100용
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="logs/outputs",
        ),
    )

    # 5. 파인튜닝 실행
    print("학습을 시작합니다...")
    trainer_stats = trainer.train()
    print(trainer_stats)

    # 6. 모델 저장 (LoRA 가중치만 저장됨)
    save_path = "logs/lora_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"LoRA 모델이 성공적으로 저장되었습니다: {save_path}")

    # (옵션) 나중에 vLLM 서빙을 원한다면 16bit 모델 병합(Merge) 저장도 가능합니다.
    # model.save_pretrained_merged("logs/merged_model", tokenizer, save_method = "merged_16bit")

if __name__ == "__main__":
    main()
