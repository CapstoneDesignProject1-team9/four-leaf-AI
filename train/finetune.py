"""
A100 파인튜닝(SFT) 스크립트 - QLoRA 기반 (안정성 강화 버전)
"""

import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# Unsloth 안전 로드
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("WARNING: Unsloth가 설치되지 않았거나 현재 환경과 호환되지 않습니다.")

def main():
    if not UNSLOTH_AVAILABLE:
        print("Error: Unsloth 모듈을 로드할 수 없습니다. 환경을 확인해주세요.")
        return

    # 1. 모델 로드 설정
    max_seq_length = 2048  
    model_name = "Bllossom/llama-3.1-Korean-Bllossom-8B"  

    print(f"[{model_name}] 모델 로딩 중...")
    
    # 패키지 충돌 방지를 위해 4bit 로드 및 bfloat16 설정
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,  
        load_in_4bit=True,  
    )

    # 2. LoRA(PEFT) 어댑터 설정
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  
    )

    # 3. 데이터셋 준비
    dataset_path = "data/synthetic_dataset.jsonl"
    if not os.path.exists(dataset_path):
        print(f"데이터셋을 찾을 수 없습니다: {dataset_path}")
        print("먼저 scripts/generate_dataset.py 를 실행하세요.")
        return

    dataset = load_dataset("json", data_files={"train": dataset_path}, split="train")

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

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # 4. 학습 파라미터 (Trainer) 설정
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=3,  
            learning_rate=2e-4,
            fp16=False,
            bf16=True,  
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="logs/outputs",
            report_to="none",  # 불필요한 wandb 로그 에러 방지
        ),
    )

    # 5. 파인튜닝 실행
    print("학습을 시작합니다...")
    trainer_stats = trainer.train()
    print(trainer_stats)

    # 6. 모델 저장
    save_path = "logs/lora_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"LoRA 모델이 성공적으로 저장되었습니다: {save_path}")

if __name__ == "__main__":
    main()