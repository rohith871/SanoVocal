# SanoVocal Tool-Contract Repair Dataset Validation Report

## Executive Summary
- **Dataset File**: `data/repair/sanovocal_tool_contract_repair.jsonl`
- **Total Records**: 226 (Target: 220–250)
- **Validation Status**: ✅ PASSED (0 ERRORS)
- **Dataset SHA256**: `0A0F6E6D588587FC39B5CA2828C18BC7FB684F0586A0CA5F35549E2037F74FCF`
- **Locked Test Set SHA256**: `667E60B9F0EDABB2DE7232DA50A2F92E6362662A37B80701FB26BB7205366FC3` (VERIFIED MATCH)

## Category Distribution
| Category | Description | Count |
| :--- | :--- | :--- |
| **A_exact_tool_contract** | Exact schema and parameter contract enforcement | 35 |
| **B_missing_info_clarification** | Clarification on missing mandatory arguments | 35 |
| **C_emergency_safety** | Red-flag triage with zero clinical prescription | 30 |
| **D_tool_result_grounding** | Source of truth grounding on tool observations | 36 |
| **E_hallucination_resistance_refusal** | Refusal of fake schemes, scams, and illegal requests | 35 |
| **F_multi_step_clarification** | Multi-turn and chained retrieval workflows | 30 |
| **G_case_summary_escalation** | Structured case summarization and human escalation | 25 |
| **Total** | | **226** |

## Language Distribution
| Language | Count | Percentage |
| :--- | :--- | :--- |
| **English (en)** | 79 | 35.0% |
| **Hindi (hi)** | 75 | 33.2% |
| **Hinglish (hinglish)** | 72 | 31.9% |
| **Total** | **226** | **100.0%** |

## Tool Dispatch Frequency
| Tool Name | Invocation Count | Validated Against Engine |
| :--- | :--- | :--- |
| `emergency_screen` | 35 | ✅ Verified |
| `match_schemes` | 21 | ✅ Verified |
| `get_scheme_rules` | 82 | ✅ Verified |
| `get_documents` | 27 | ✅ Verified |
| `find_facility` | 10 | ✅ Verified |
| `create_case_summary` | 29 | ✅ Verified |
| `escalate_to_human` | 29 | ✅ Verified |

## Hallucinated Tool Audit
- Checked for forbidden variants: `match_scheme`, `scheme_match`, `match_official_scheme`, `symptom_tool`, `match_health_facilities`.
- **Occurrences Found**: **0** (100% clean).

## Frozen Asset Integrity
- `agent_trajectories_test.jsonl` SHA256: `667E60B9F0EDABB2DE7232DA50A2F92E6362662A37B80701FB26BB7205366FC3` (Matches locked baseline `667E60B9F0EDABB2DE7232DA50A2F92E6362662A37B80701FB26BB7205366FC3`).
- Phase A and Phase B directories verified untouched and read-only.
- Kaggle training notebook (`notebooks/sanovocal_aya_qlora_training.ipynb`) untouched.

## Zero-Defect Certification
All records strictly satisfy SanoVocal deterministic schema contracts and safety guidelines.