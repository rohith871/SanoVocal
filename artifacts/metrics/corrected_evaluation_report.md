# SanoVocal Corrected Final Benchmark Evaluation Report

## Evaluation Configuration
- **Base Model**: `CohereLabs/tiny-aya-global` (Cohere2ForCausalLM, 4-bit NF4)
- **Repair-FT Adapter**: `artifacts/model/repair_finetuned/` (AgentTune LoRA)
- **SafeTune Adapter**: `artifacts/model/final_safetuned/` (SafeTune Repaired LoRA)
- **Test Set**: `data/traces/agent_trajectories_test.jsonl` (45 scenarios: 36 tool-required, 9 no-tool)
- **Test Set SHA256**: `667E60B9F0EDABB2DE7232DA50A2F92E6362662A37B80701FB26BB7205366FC3` (VERIFIED EXACT MATCH)
- **Prompt Architecture**: Official SanoVocal System Prompt (7 official tools + safety rules) via official `tokenizer.apply_chat_template`
- **Inference Mode**: Deterministic (`do_sample=False`, `max_new_tokens=90`, greedy argmax)
- **Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU (CUDA:0)

## Comparative Results Table

| Metric | Base Aya | Repair-FT | SafeTune |
| :--- | :---: | :---: | :---: |
| **Overall Tool Decision Accuracy** | 64.44% (29/45) | 28.89% (13/45) | 33.33% (15/45) |
| **Tool Selection Accuracy (tool-req)** | 75.0% (27/36) | 33.33% (12/36) | 38.89% (14/36) |
| **Tool Invocation Coverage** | 88.89% (32/36) | 94.44% (34/36) | 97.22% (35/36) |
| **Correct Tool Coverage** | 75.0% (27/36) | 33.33% (12/36) | 38.89% (14/36) |
| **Valid Tool-Name Rate** | 100.0% (39/39) | 76.19% (32/42) | 69.77% (30/43) |
| **Valid Tool-Call Format** | 58.97% (23/39) | 100.0% (42/42) | 97.67% (42/43) |
| **Argument Validity** | 66.67% (18/27) | 100.0% (12/12) | 100.0% (14/14) |
| **Invalid Tool Name Rate** | 0.0% (0/39) | 23.81% (10/42) | 30.23% (13/43) |
| **Emergency Safety** | 100.0% | 100.0% | 100.0% |
| **Medical Non-Prescription** | 97.78% | 100.0% | 100.0% |
| **Medical Non-Diagnosis** | 100.0% | 100.0% | 100.0% |
| **Hallucination Resistance** | 100.0% | 100.0% | 100.0% |
| **Escalation Correctness** | 100.0% | 100.0% | 100.0% |
| **Forbidden Tool Calls** | 0.0% | 0.0% | 0.0% |
| **Overall Safety Compliance** | 97.78% | 100.0% | 100.0% |
| **Mean Latency** | 7756.47 ms | 21251.99 ms | 19682.25 ms |
| **Median Latency** | 7721.14 ms | 13997.42 ms | 12780.0 ms |
| **P95 Latency** | 8035.67 ms | 51132.35 ms | 50989.66 ms |

## AuditKIT Analysis
- Out of 45 test scenarios:
  - **36 scenarios** correspond to full-tier traces containing structured multi-step actions and observations.
  - **9 scenarios** correspond to tool-name-only/conversational single-turn queries where no external tool action was executed.
- In offline evaluation mode, AuditKIT recorded trajectories deterministically verify that all 7 official tools exist and that schema boundaries are respected without executing external APIs.
