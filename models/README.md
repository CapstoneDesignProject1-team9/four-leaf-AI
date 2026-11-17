# 평가용 LoRA 어댑터

두 PEFT 어댑터의 실제 파일 위치입니다. vLLM Compose 설정은 이 디렉터리들을 각각 컨테이너의 `/models/student-lora`와 `/models/professor-lora`에 읽기 전용으로 마운트합니다.

```text
models/student-lora/adapter_student_llama/
models/professor-lora/adapter_professor_llama/
```

각 adapter 폴더에 `adapter_config.json`과 `adapter_model.safetensors`가 있습니다. 현재 두 설정은 모두 `meta-llama/Llama-3.1-8B-Instruct`, rank 16, PEFT LoRA입니다. Compose의 base model은 이 ID와 일치해야 하며 `VLLM_MAX_LORA_RANK`는 16 이상이어야 합니다.

vLLM은 경로를 각각 API 모델명 `student-lora`, `professor-lora`로 등록합니다. 생성기는 이 이름을 사용해 다섯 학생용 평가 조건을 요청합니다. 가중치 파일은 크기가 커서 Git 추적 대상에서 제외됩니다.
