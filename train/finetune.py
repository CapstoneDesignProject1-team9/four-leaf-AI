import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# scripts/generate_qa.py, scripts/generate_dataset.py 가 만드는 파일 (우선순위 순)
DATA_CANDIDATES = [
    "data/validated_qa_dataset.jsonl",
    "data/synthetic_dataset.jsonl",
]


def find_data_path():
    for path in DATA_CANDIDATES:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    raise FileNotFoundError(
        "학습 데이터가 없습니다. 아래 파일 중 하나가 필요합니다: "
        + ", ".join(DATA_CANDIDATES)
        + "\n(scripts/generate_qa.py 실행 또는 팀원에게 데이터 파일 요청)"
    )


def to_messages(example):
    # {"instruction": ..., "output": ...} -> 채팅 형식
    # SFTTrainer가 모델의 chat template(Phi-3)을 자동 적용합니다.
    return {
        "messages": [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
    }


def main():
    print("=== 1. 모델 설정 및 토크나이저 로드 중 ===")
    # 에어갭 환경이면 로컬 경로로 바꾸세요
    model_id = "microsoft/Phi-3-mini-4k-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # bf16 지원 GPU(A6000 등)면 bf16, 아니면 fp32 로드 + fp16 학습
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"bf16 사용: {use_bf16}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map="auto",
    )

    print("=== 2. LoRA 어댑터 설정 중 ===")
    # Phi-3는 q/k/v가 qkv_proj로, gate/up이 gate_up_proj로 합쳐져 있음
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("=== 3. 데이터셋 준비 중 ===")
    data_path = find_data_path()
    print(f"학습 데이터: {data_path}")
    raw = load_dataset("json", data_files=data_path)["train"]
    train_dataset = raw.map(to_messages, remove_columns=raw.column_names)
    print(f"데이터 개수: {len(train_dataset)}")
    print("샘플:", train_dataset[0]["messages"])

    print("=== 4. SFTConfig 및 SFTTrainer 설정 중 ===")
    training_args = SFTConfig(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        max_length=1024,  # 구버전 trl이면 max_seq_length
        bf16=use_bf16,
        fp16=not use_bf16,
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

    trainer.model.save_pretrained("./final_adapter")
    tokenizer.save_pretrained("./final_adapter")
    print("=== 학습 완료 및 어댑터 저장 성공! ===")


if __name__ == "__main__":
    main()
