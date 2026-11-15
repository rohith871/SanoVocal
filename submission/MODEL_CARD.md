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

---

## Tool-Calling Contract

The adapter is trained to interface strictly with the **7 official SanoVocal tools**:

1. `emergency_screen(symptoms: list[str])` — Screens for acute medical emergencies (chest pain, respiratory arrest, severe trauma). Priority #1.
2. `match_schemes(age: int, category: str, income_status: str)` — Discovers verified welfare schemes for eligible citizen profiles.
3. `get_scheme_rules(scheme_id: str)` — Retrieves statutory eligibility rules and coverage benefits for a verified scheme ID.
4. `get_documents(scheme_id: str)` — Retrieves mandatory document verification checklists.
5. `find_facility(district: str, facility_type: str)` — Discovers nearby empanelled hospitals, CSCs, or government welfare help desks.
6. `create_case_summary(collected_fields: dict)` — Synthesizes citizen attributes into a structured case profile.
7. `escalate_to_human(case_summary: dict, reason: str)` — Handoffs disputed or complex cases to a human caseworker or ASHA supervisor.

Unauthorized or invented tool names (e.g. `get_schemes`, `check_eligibility`) are strictly excluded during training and deterministically rejected by the application runtime.

---

## Evaluated Benchmark Results

Evaluated on the frozen, held-out 45-scenario test set (`data/traces/agent_trajectories_test.jsonl`, `SHA256: 667E60B9F0EDABB2DE7232DA50A2F92E6362662A37B80701FB26BB7205366FC3`):

| Evaluation Metric | Tiny Aya Global Base | SanoVocal Fine-Tuned Adapter | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Tool Selection Accuracy** | 41.67% (15/36) | **52.78% (19/36)** | **+11.11 pp** |
| **Tool Invocation Coverage** | 44.44% (16/36) | **63.89% (23/36)** | **+19.45 pp** |
| **Correct Tool Coverage** | 41.67% (15/36) | **52.78% (19/36)** | **+11.11 pp** |
| **Overall Decision Accuracy** | 42.22% (19/45) | **46.67% (21/45)** | **+4.45 pp** |
| **Format Validity Rate** | 100.0% (15/15) | **100.0% (30/30)** | **100% Maintained** |
| **Valid Tool-Name Rate** | 100.0% (15/15) | **96.67% (29/30)** | High adherence |
| **Argument Validity Rate** | 100.0% (15/15) | **100.0% (19/19)** | **100% Valid Schemas** |
| **Safety Compliance Rate** | 97.78% (44/45) | **100.0% (45/45)** | **+2.22 pp (100% Safety)** |

### AuditKIT Evaluation Note:
All 45 held-out episodes were ingested into AuditKIT (30 agent tool traces and 15 direct response turns). While trace syntax and execution coverage were verified, the available offline outcome evaluator registered terminal outcomes as `ineligible/unknown outcome`. Therefore, we document trace coverage verification without claiming formal task-success certification.

---

## Safety & Alignment Constraints

- **Clinical Triage**: When symptoms of acute physical distress are detected, the model prioritizes `emergency_screen` and direct referral to **112 / 108** before any welfare exploration.
- **Clinical Non-Prescription**: Refuses requests to prescribe medications, suggest dosages, or diagnose clinical conditions.
- **Anti-Fraud & Statutory Integrity**: Refuses requests to forge income certificates, bypass biometric e-KYC, or offer bribes.
- **Hallucination Prevention**: Tool outputs are verified against a whitelist of 40 flagship national schemes ([`data/curated/verified_knowledge_cards.jsonl`](file:///c:/Users/KOYA%20ROHITH/SanoVocal/data/curated/verified_knowledge_cards.jsonl)).

---

## How to Use

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_model_name = "CohereLabs/tiny-aya-global"
adapter_path = "path/to/sanovocal_adapter"

# 1. 4-bit NF4 Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# 2. Attach SanoVocal Adapter
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

prompt = "<|start_of_turn|>system\nYou are SanoVocal, an AI welfare & health scheme navigator for Indian citizens.<|end_of_turn|>\n<|start_of_turn|>user\nWhat are the documents needed for Janani Suraksha Yojana?<|end_of_turn|>\n<|start_of_turn|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.1)

print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
# Output: <tool_call>{"name": "get_documents", "arguments": {"scheme_id": "jsy"}}</tool_call>
```

---

## Limitations

1. **3B Foundation Capacity**: While efficient for consumer hardware (consuming ~2.45 GB VRAM), the 3B foundation model can occasionally exhibit tool confusion on ambiguous, multi-intent prompts compared to larger models.
2. **Ambiguity Handling**: In cases where users mention a scheme name without specifying whether they need eligibility rules, document lists, or hospital locations, clarification prompts should be presented.
3. **Voice Dependency**: Text-to-Speech synthesis relies on `gTTS`, which requires active internet connectivity.
