# SanoVocal — Phase B Agentic Trajectory Preparation Report
**Toolkit**: AgentTune (v1.1.3)  
**Execution Timestamp**: 2026-11-09T08:08:44.157845+00:00  
**Status**: COMPLETED & VALIDATED

---

## 1. Executive Summary

Phase B prepared, executed, validated, and split the multi-turn agentic tool trajectories required for later AgentTune QLoRA fine-tuning on Aya Expanse 8B.

- **Total Scenarios Evaluated**: 380
- **Valid Trajectories Produced**: 380 (100% schema & order validated)
- **Rejected Trajectories**: 0
- **Tool Architecture**: 7 discrete deterministic tools (`emergency_screen`, `match_schemes`, `get_scheme_rules`, `get_documents`, `find_facility`, `create_case_summary`, `escalate_to_human`)

---

## 2. Trajectory Structure & Breakdown

### Tool Invocations Across Dataset:
{
  "emergency_screen": 40,
  "match_schemes": 126,
  "get_scheme_rules": 110,
  "get_documents": 110,
  "find_facility": 20,
  "create_case_summary": 16,
  "escalate_to_human": 16
}

### Workflow Complexity:
- **Single-Tool Invocations**: 286
- **Multi-Tool Workflows (2–4 step tool chains)**: 46
- **Zero-Tool Conversational & Refusal Cases**: 48
- **Safety-Critical Trajectories (Emergency & Prohibited Advice)**: 26

---

## 3. Leakage-Resistant Dataset Splitting

Data was stratified across all 11 scenario categories and languages (English, Hindi, Hinglish) to prevent cross-split leakage:
- **Train Set (80%)**: 300 trajectories
- **Validation Set (10%)**: 35 trajectories
- **Test Set (10%)**: 45 trajectories (Held out strictly for later AuditKIT evaluation)

---

## 4. Deliverables Produced

- `data/traces/agent_trajectories.jsonl`: Portable AgentTune format with complete steps, thoughts, and observations.
- `data/traces/agent_trajectories_train.jsonl`: Training split.
- `data/traces/agent_trajectories_val.jsonl`: Validation split.
- `data/traces/agent_trajectories_test.jsonl`: Held-out test split.
- `data/traces/agent_trajectories_trl_format.jsonl`: Standard TRL/OpenAI function-calling message format.
- `data/traces/trajectory_manifest.json`: Full metrics and tool distribution manifest.

Reproducible via:
```powershell
python scripts/06_generate_scenarios.py
python scripts/07_generate_and_validate_trajectories.py
```
