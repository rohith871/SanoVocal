---
language:
- en
- hi
license: apache-2.0
base_model: CohereLabs/tiny-aya-global
tags:
- agentic-fine-tuning
- tool-calling
- safety-alignment
- lexsi-hackathon
- public-welfare
- healthcare
---

# SanoVocal Fine-Tuned Adapter 🩺🇮🇳

The **SanoVocal fine-tuned adapter** is a parameter-efficient fine-tuned LoRA adapter built on top of **CohereLabs/tiny-aya-global** (3B parameters) for **Track 3 (Agentic Fine-Tuning)** of the **Lexsi AI Safety and Alignment Hackathon**.

The model is designed to function as an agentic conversational navigator for Indian statutory healthcare and welfare programs, ensuring reliable tool calling, strict adherence to Pydantic argument schemas, and clinical emergency prioritization.

---

## Model Summary

- **Model Name**: SanoVocal Fine-Tuned Adapter
- **Base Foundation Model**: `CohereLabs/tiny-aya-global` (3B parameters)
- **Adapter Type**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit NormalFloat4 (NF4) via `bitsandbytes`
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj` ($r=16, \alpha=32$)
- **Adapter File**: `adapter_model.safetensors`
- **Adapter SHA256**: `39291232E229F7B5B3A03A424298CB67025CC958E4D09B59564588C6AB44FCAC`
- **Intended Languages**: Hindi (`hi`), English (`en`), and Hinglish (code-mixed)
- **Hugging Face Model**: `https://huggingface.co/rohith871/sanovocal-adapter`
- **Hugging Face Collection**: `https://huggingface.co/collections/rohith871/sanovocal-models`
- **GitHub Repository**: `https://github.com/rohith871/SanoVocal`

---

## Tool-Calling Contract

The adapter is trained to interface strictly with the **7 official SanoVocal tools**:

1. `emergency_screen(symptoms: str)` — Triage acute medical red-flags (directs to 112/108 immediately).
2. `match_schemes(state: str, income?: int, category?: str)` — Matches citizen profile to verified schemes.
3. `get_scheme_rules(scheme_id: str)` — Retrieves statutory eligibility rules and coverage limits.
4. `get_documents(scheme_id: str)` — Retrieves mandatory statutory verification documents.
5. `find_facility(pincode?: str, district?: str, scheme_id?: str)` — Locates empanelled hospitals & CSCs.
6. `create_case_summary(beneficiary_name: str, scheme_id: str, status: str)` — Generates a persistent case record.
7. `escalate_to_human(reason: str, contact_info: str)` — Hands off complex queries to ASHA / supervisors.

---

## Evaluated Benchmark Results

Evaluated on the frozen, held-out 45-scenario test set (`data/traces/agent_trajectories_test.jsonl`, `SHA256: 667E60B9F0EDABB2DE7232DA50A2F92E6362662A37B80701FB26BB7205366FC3`):

| Evaluation Metric | Tiny Aya Global Base | SanoVocal Fine-Tuned Adapter | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Tool Selection Accuracy** | 41.67% (15/36) | **52.78% (19/36)** | **+11.11 pp** |
| **Tool Invocation Coverage** | 44.44% (16/36) | **63.89% (23/36)** | **+19.45 pp** |
| **Correct Tool Coverage** | 41.67% (15/36) | **52.78% (19/36)** | **+11.11 pp** |
| **Overall Decision Accuracy** | 42.22% (19/45) | **46.67% (21/45)** | **+4.45 pp** |
| **Format Validity Rate** | 100.0% (21/21) | **100.0% (30/30)** | **100% Maintained** |
| **Valid Tool-Name Rate** | 100.0% (21/21) | **96.67% (29/30)** | **High adherence (1 invalid call)** |
| **Argument Validity Rate** | 100.0% (15/15) | **100.0% (19/19)** | **100% Valid Schemas** |
| **Overall Safety Compliance** | 95.56% (43/45) | **100.0% (45/45)** | **+4.44 pp (100% Safety)** |

### AuditKIT Evaluation Note:
All 45 held-out episodes were ingested into AuditKIT (30 agent tool traces and 15 direct response turns). While trace syntax and execution coverage were verified, the available offline outcome evaluator registered terminal outcomes as `ineligible/unknown outcome`. Therefore, we document trace coverage verification without claiming formal task-success certification.

---

## Safety & Alignment Constraints

- **Clinical Triage**: When symptoms of acute physical distress are detected, the model prioritizes `emergency_screen` and direct referral to **112 / 108** before any welfare exploration.
- **Clinical Non-Prescription**: Refuses requests to prescribe medications, suggest dosages, or diagnose clinical conditions.
- **Anti-Fraud & Statutory Integrity**: Refuses requests to forge income certificates, bypass biometric e-KYC, or offer bribes.
- **Hallucination Prevention**: Tool outputs are verified against a whitelist of 40 flagship national schemes ([`data/curated/verified_knowledge_cards.jsonl`](../data/curated/verified_knowledge_cards.jsonl)).

---

## How to Use

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_model_name = "CohereLabs/tiny-aya-global"
adapter_path = "rohith871/sanovocal-adapter"

# 1. 4-bit NF4 Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# 2. Load Base Model + Adapter
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_path)

# 3. Format Prompt
messages = [
    {"role": "system", "content": "You are SanoVocal, a statutory healthcare welfare assistant."},
    {"role": "user", "content": "Mujhe Ayushman card ke liye documents batao."}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 4. Generate
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

---

## Limitations

- Grounded exclusively in statutory Indian welfare programs as of November 2026.
- Does not provide medical diagnoses or prescriptions.
- Audio input requires frontend ASR (e.g. Faster-Whisper) for speech recognition.
