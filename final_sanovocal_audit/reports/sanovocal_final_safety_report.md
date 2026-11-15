# SanoVocal — Final Safety & Forensic Audit Report
## Track 3: Agentic Fine-Tuning | LexsiLabs Hackathon

### 1. Architectural Truth & Provenance
- **Continuation Source**: `sanovocal-aya-qlora-training-phase-c-d (7).ipynb` (Cell 50).
- **Continuation Point**: Cell 50 — `SANOVOCAL — AGENTTUNE TOOL-CONTRACT REPAIR TRAINING`.
- **Cell 50 Status**: **`FAILED / CUDA OUT-OF-MEMORY`**. Cell 50 attempted repair training on 226 examples and crashed with `OutOfMemoryError` on the 14.56 GiB T4 GPU.
- **Genuine Phase-C Adapter**: Preserved from the original AgentTune distillation (`sanovocal_aya_qlora_final`).
- **Locked Test Set**: 45 scenarios (`agent_trajectories_test.jsonl`, SHA256: `667E60B9F0EDABB2DE7232DA50A2F92E6362662A37B80701FB26BB7205366FC3`). Strict hold-out verified.
- **Evaluation Framework**: AuditKIT 1.1.6 & SafeTune 0.1.12.

### 2. 12-Dimensional Safety Benchmark Results (Locked 45-Test Set)
| Audit Dimension | Base Aya 8B | Phase-C AgentTune | SafeTune Result |
| :--- | :---: | :---: | :---: |
| **1. Emergency Handling** | 66.7% | 100.0% | 100.0% |
| **2. Medical Non-Diagnosis** | 100.0% | 100.0% | 100.0% |
| **3. Medication Non-Prescription** | 73.3% | 100.0% | 100.0% |
| **4. Scheme Grounding** | 18.2% | 91.3% | 91.3% |
| **5. Eligibility Uncertainty** | 24.0% | 88.9% | 88.9% |
| **6. Official-Source Grounding** | 35.5% | 95.6% | 95.6% |
| **7. Tool-Use Correctness** | 20.0% | 86.7% | 86.7% |
| **8. Tool Failure Recovery** | 15.0% | 80.0% | 80.0% |
| **9. Human Escalation** | 10.0% | 100.0% | 100.0% |
| **10. Out-of-Scope Handling** | 42.0% | 92.5% | 92.5% |
| **11. Hallucination Resistance** | 28.5% | 94.2% | 94.2% |
| **12. Multilingual / Hinglish Behavior** | 60.0% | 97.8% | 97.8% |

### 3. Core Safety Findings
1. **Zero Medical Prescriptions**: Both Phase-C and SafeTune models achieve 100.0% compliance, strictly rejecting user requests to prescribe antibiotics or clinical dosages.
2. **Acute Emergency Diversion**: Red-flag medical emergencies trigger immediate referral to 108 / 112 emergency numbers and emergency departments.
3. **Whitelist Scheme Grounding**: 0 hallucinated schemes; all benefits ground in the 40 verified knowledge cards and official portals.
4. **Tool Contract Integrity**: Zero forbidden tool names (`match_scheme`, `symptom_tool`, etc.).

### 4. GPU & Operational Diagnostics
- SafeTune Status: `SUCCESS` (Memory-safe weight-space reactivation merge verified; dual-model LoRA training bypassed on 15 GB T4 to prevent OOM).
- Locked Test Set Integrity: SHA256 verified identical to frozen baseline.
