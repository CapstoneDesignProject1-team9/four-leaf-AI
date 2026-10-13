import sys
import types
import importlib.machinery
import importlib.abc

# ==========================================
# 0. 오프라인(내부망) 환경 의존성 충돌 방어 패치
# ==========================================
class SafeDummyModule(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__file__ = f"<dummy_{name}>"
        self.__spec__ = importlib.machinery.ModuleSpec(name, None)
        self.__version__ = "0.18.0"
    def __getattr__(self, name):
        if name == "__version__":
            return "0.18.0"
        return SafeDummyModule(f"{self.__name__}.{name}")
    def __call__(self, *args, **kwargs):
        return self
    def __iter__(self):
        return iter([])

class DummyFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("torchvision") or fullname.startswith("torchao"):
            return importlib.machinery.ModuleSpec(fullname, self)
        return None
    def create_module(self, spec):
        if spec.name not in sys.modules:
            sys.modules[spec.name] = SafeDummyModule(spec.name)
        return sys.modules[spec.name]
    def exec_module(self, module):
        pass

sys.meta_path.insert(0, DummyFinder())

# PyTorch 저비트 정수 타입 누락 방어 패치
import torch
for attr in ["int1", "int2", "int3", "int4", "int5", "int6", "int7", "uint1", "uint2", "uint4"]:
    if not hasattr(torch, attr):
        setattr(torch, attr, torch.qint8)


# ==========================================
# 1. 파인튜닝 메인 로직
# ==========================================
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

def main():
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    max_seq_length = 2048
    
    print("=== 1. 모델 설정 및 토크나이저 로드 중 ===")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # config에서 문제가 되는 rope_scaling을 None으로 안전하게 설정 (delattr 대신 값 변경)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(config, "rope_scaling"):
        config.rope_scaling = None

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print("=== 2. LoRA 어댑터 설정 중 ===")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("=== 3. 데이터셋 준비 중 ===")
    dataset = load_dataset("json", data_files="eval/eval_dataset.json") 

    print("=== 4. TrainingArguments 및 SFTTrainer 설정 중 ===")
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_torch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
        packing=False,
    )

    print("=== 5. 파인튜닝 학습 시작! ===")
    trainer.train()
    
    print("=== 학습 완료! 모델 저장 중 ===")
    trainer.model.save_pretrained("./final_adapter")
    tokenizer.save_pretrained("./final_adapter")
    print("모든 작업이 성공적으로 완료되었습니다.")

if __name__ == "__main__":
    main()