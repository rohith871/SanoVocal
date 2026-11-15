# SanoVocal Tool Contract Repair Dataset V2 Final Validation Report

## Executive Summary
- **Dataset File**: `data/repair/sanovocal_tool_contract_repair_v2.jsonl`
- **SHA256**: `D52B266253154ADBBA0B9B5B87E36B2188AC0D3F9415723FF7A2A5357445E60C`
- **Total Records**: 415
- **System Prompt Coverage**: 415 / 415 (100.0%)
- **Test Set Leakage**: 0 records (checked against 45 locked test scenarios)
- **Locked Test Set SHA256**: `667E60B9F0EDABB2DE7232DA50A2F92E6362662A37B80701FB26BB7205366FC3` (VERIFIED UNCHANGED)
- **Invalid Target Tool Names**: 0 (ZERO instances of `get_schemes`, `get_scheme_info`, etc.)
- **Schema Violations**: 0
- **Semantic Argument Violations**: 0
- **Unknown Scheme IDs**: 0 (100% of referenced scheme IDs belong to the 40 official verified schemes)
- **Duplicate Records**: 0

## Tool Target Call Distribution
| Official Tool | Target Call Count | Percentage |
| :--- | :---: | :---: |
| `get_scheme_rules` | 97 | 25.3% |
| `get_documents` | 78 | 20.4% |
| `emergency_screen` | 50 | 13.1% |
| `match_schemes` | 45 | 11.7% |
| `create_case_summary` | 38 | 9.9% |
| `escalate_to_human` | 38 | 9.9% |
| `find_facility` | 37 | 9.7% |
| **Total Official Tool Calls** | **383** | **100.0%** |
| **NO_TOOL (Conversational/Refusal/Clarification)** | **87** | -- |

## Dedicated `get_documents` Audit
- **Total `get_documents` Target Calls**: 78
- **English (`en`) Calls**: 25
- **Hindi (`hi`) Calls**: 26
- **Hinglish (`hinglish`) Calls**: 27
- **Known Scheme Coverage**: 78 / 78 (100.0%)
- **Target Calls to `get_schemes`**: **0** (ZERO)

## NO_TOOL Breakdown (87 records)
| Category | Count | Purpose |
| :--- | :---: | :--- |
| **Fraud & Scam Refusal** | 15 | Explicitly rejects fake scheme claims (iPhone, lottery, gold cash) without calling tools. |
| **Clarification Intake** | 3 | Requests user age/income/need when details are too vague to call a tool. |
| **Medical Prescription Refusal** | 4 | Refuses medication/dosage requests and redirects to qualified doctors. |
| **Concept & Procedure Explanation** | 5 | Explains DBT, Ayushman Mitra roles, etc. without tool dispatch. |
| **Greeting & Persona Introduction** | 2 | Welcomes user and outlines SanoVocal scope. |
| **Conversational & Courtesy** | 58 | Natural dialog closures and non-tool inquiries. |

## Language Distribution
| Language | Count | Percentage |
| :--- | :---: | :---: |
| English (`en`) | 142 | 34.2% |
| Hindi (`hi`) | 138 | 33.3% |
| Hinglish (`hinglish`) | 135 | 32.5% |
