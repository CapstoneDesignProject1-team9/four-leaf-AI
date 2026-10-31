# Compare base Llama 3.1 8B vs student/professor adapters (output: test_results.md)
# NOTE: this file is ASCII only on purpose (Korean text is written as unicode escapes).
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "meta-llama/Llama-3.1-8B-Instruct"
STUDENT = "./adapter_student_llama"
PROFESSOR = "./adapter_professor_llama"

# 1st student question and 1st professor question were in the training data; the rest were not.
STUDENT_QS = [
    "\ub370\uc774\ud130\uacfc\ud559\uae30\ucd08 \uc774\uc131\ud604 \uad50\uc218\ub2d8 \uc218\uc5c5 \ub4e4\uc744\uae4c \uace0\ubbfc \uc911\uc778\ub370, \uc218\uc5c5\uc774\ub791 \uc911\uac04\uace0\uc0ac\ub294 \uc5b4\ub5a4\uac00\uc694?",
    "\ubcf5\uc218\uc804\uacf5 \uc2e0\uccad\ud558\ub824\uba74 \uc5b4\ub5a4 \uc808\ucc28\ub97c \uac70\uccd0\uc57c \ud558\ub098\uc694?",
    "\ud559\uad50 \uacf5\uc9c0\uc5d0\uc11c \uc7a5\ud559\uae08 \uc2e0\uccad \uc548\ub0b4\uac00 \uc62c\ub77c\uc624\uba74 \uc5b4\ub5bb\uac8c \ud655\uc778\ud558\uba74 \ub418\ub098\uc694?",
]
PROFESSOR_QS = [
    "2026\ud559\ub144\ub3c4 2\ud559\uae30 \ucef4\ud4e8\ud130\ud559\ubd80 '\uc885\ud569\uc124\uacc4\ud504\ub85c\uc81d\ud2b81'(ITEC0401-002) 1~7\uc8fc\ucc28 \uac15\uc758 \uc77c\uc815\uc5d0 \ub9de\ucdb0 \ud034\uc988 4\ubb38\ud56d\uc744 \ub9cc\ub4e4\uc5b4 \uc8fc\uc138\uc694. \uc815\ub2f5\uacfc \ud55c\ub450 \ubb38\uc7a5 \ud574\uc124\ub3c4 \ud568\uaed8 \ubd80\ud0c1\ub4dc\ub9bd\ub2c8\ub2e4.",
    "\ucef4\ud4e8\ud130\ud559\ubd80 '\uc6b4\uc601\uccb4\uc81c' \uc218\uc5c5 1~3\uc8fc\ucc28 \ub0b4\uc6a9\uc73c\ub85c \uc218\uc5c5 \uc900\ube44 \ub9ac\ud3ec\ud2b8\ub97c \ub9cc\ub4e4\uc5b4 \uc8fc\uc138\uc694. \uac15\uc870\ud560 \ud575\uc2ec \ud0a4\uc6cc\ub4dc 3~5\uac1c\uc640 \uac15\uc870 \ud3ec\uc778\ud2b8\ub3c4 \uc815\ub9ac\ud574 \uc8fc\uc138\uc694.",
]

tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, STUDENT, adapter_name="student")
model.load_adapter(PROFESSOR, adapter_name="professor")
model.eval()


def ask(question, max_new_tokens):
    enc = tok.apply_chat_template(
        [{"role": "user", "content": question}], add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


lines = []
for group, questions, adapter, max_new in [
    ("STUDENT", STUDENT_QS, "student", 350),
    ("PROFESSOR", PROFESSOR_QS, "professor", 700),
]:
    for q in questions:
        lines.append("\n\n## [" + group + "] " + q + "\n")
        with model.disable_adapter():
            lines.append("\n### BASE (no adapter)\n\n" + ask(q, max_new) + "\n")
        model.set_adapter(adapter)
        lines.append("\n### " + group + " ADAPTER\n\n" + ask(q, max_new) + "\n")
        print("done:", q[:20].encode("unicode_escape").decode(), flush=True)

with open("test_results.md", "w", encoding="utf-8") as f:
    f.write("# Adapter comparison\n" + "".join(lines))
print("saved test_results.md")
