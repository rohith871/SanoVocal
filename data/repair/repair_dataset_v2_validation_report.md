# SanoVocal Tool Contract Repair Dataset V2 Validation Report

## Executive Summary
- **Dataset File**: `data/repair/sanovocal_tool_contract_repair_v2.jsonl`
- **SHA256**: `C437A4DD086A1E7D5E90F708FD09A7ED6B5707DE4DFA4BB050AE31BFAB3F36B7`
- **Total Records**: 415
- **System Prompt Coverage**: 415 / 415 (100.0%)
- **Test Set Leakage**: 0 records (checked against 45 locked test scenarios)
- **Invalid Target Tool Names**: 0 (zero instances of `get_schemes`, `get_scheme_info`, etc.)

## Tool Call Distribution
| Official Tool | Target Call Count |
| :--- | :---: |
| `match_schemes` | 45 |
| `get_scheme_rules` | 112 |
| `get_documents` | 78 |
| `find_facility` | 37 |
| `emergency_screen` | 50 |
| `create_case_summary` | 38 |
| `escalate_to_human` | 38 |
| **NO_TOOL (Conversational / Refusal / Clarification)** | **72** |
| **Total Official Tool Calls** | **398** |

## Language Distribution
| Language | Count | Percentage |
| :--- | :---: | :---: |
| English (`en`) | 142 | 34.2% |
| Hindi (`hi`) | 138 | 33.3% |
| Hinglish (`hinglish`) | 135 | 32.5% |

## Semantic Routing Enhancements in V2
1. **System-Level Grounding**: 100.0% of records now include the official SanoVocal system prompt defining the 7 tools, schemas, and safety boundaries.
2. **Explicit Contrastive Triples**: Directly pairs identical domains with distinct intents (`match_schemes` for discovery vs `get_scheme_rules` for eligibility/rules vs `get_documents` for required paperwork).
3. **Cure for `get_schemes` Hallucination**: `get_documents` target examples boosted by +189% (from 27 to 78 calls), ensuring the model maps paperwork/ID queries strictly to `get_documents`.
4. **No-Tool Decision Discipline**: Expanded no-tool examples to 72 records covering greetings, clarifications, DBT explanations, and refusal of fictitious schemes.
5. **Zero Test Set Contamination**: Every candidate record was checked against `data/traces/agent_trajectories_test.jsonl`; zero overlaps exist.
