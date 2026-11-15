# SanoVocal Second-Stage Fine-Tuning Dataset (V3) — Validation Report

**Dataset Path**: `data/repair/sanovocal_stage2_100.jsonl`  
**Dataset Records**: 100  
**Dataset SHA256**: `6BCC8A10A6A9C9A215A23C5AF344B95CEC9598DAC63C67B283B93DA8FBA019E4`  
**Source Dataset**: `data/repair/sanovocal_tool_contract_repair_v2_compact.jsonl` (Pre-validated genuine repair repository)  
**Creation Date**: 2026-11-14  

---

## 1. Validation Gate Summary

| Check | Expected | Actual | Status |
| :--- | :---: | :---: | :---: |
| **Total Selected Examples** | Exactly 100 | 100 | **PASS** |
| **Source Consistency** | Genuine existing records | 100% verified | **PASS** |
| **Duplicate Tasks** | 0 | 0 | **PASS** |
| **Invalid Tool Names** | 0 | 0 | **PASS** |
| **Argument/Schema Violations** | 0 | 0 | **PASS** |
| **Locked-Test Leakage** | 0 | 0 | **PASS** |
| **Format Validity** | Explicit `<tool_call>` / SFT messages | 100% | **PASS** |

---

## 2. Tool & Language Distributions

### Tool Distribution (Balanced Across 7 Official Tools + NO_TOOL)
| Tool Name | Intended Semantic Purpose | Count | Share (%) |
| :--- | :--- | :---: | :---: |
| `match_schemes` | Profile-based welfare discovery & recommendations | 16 | 16.0% |
| `get_scheme_rules` | Rules, eligibility criteria, and benefit limits lookup | 16 | 16.0% |
| `get_documents` | Required application paperwork & document checklists | 16 | 16.0% |
| `find_facility` | Empanelled hospital, CSC, or district facility locator | 12 | 12.0% |
| `emergency_screen` | Clinical screening for acute red flags (chest pain, stroke, trauma) | 12 | 12.0% |
| `create_case_summary` | Structured citizen intake dossier compilation | 10 | 10.0% |
| `escalate_to_human` | Direct handoff for disputes, care denial, corruption | 10 | 10.0% |
| `NO_TOOL` | Fraud scheme refusal, conversational greetings, dialogue | 8 | 8.0% |
| **Total** | | **100** | **100.0%** |

### Language Distribution
| Language | Count | Share (%) |
| :--- | :---: | :---: |
| English (`en`) | 32 | 32% |
| Hindi (`hi`) | 32 | 32% |
| Hinglish (`hinglish`) | 36 | 36% |
| **Total** | **100** | **100.0%** |

---

## 3. Sample Examples Per Tool

### Tool: `match_schemes` (Showing 2 of 16 selected)

- **ID**: `v2_repair_cat_a_006` (`en`)
  - **Task**: "I am a 72 year old senior citizen seeking universal healthcare insurance coverage. What scheme matches?"
  - **Action**: `{"name": "match_schemes", "arguments": {"age": 72, "category": "health", "income_status": "eligible"}}`

- **ID**: `v2_repair_cat_a_009` (`en`)
  - **Task**: "I am a 28 year old farmer looking for crop risk coverage and income support schemes."
  - **Action**: `{"name": "match_schemes", "arguments": {"age": 28, "category": "agriculture", "income_status": "eligible"}}`

### Tool: `get_scheme_rules` (Showing 2 of 16 selected)

- **ID**: `v2_repair_cat_a_012` (`en`)
  - **Task**: "What are the exact eligibility criteria and benefits of Ayushman Bharat PM-JAY?"
  - **Action**: `{"name": "get_scheme_rules", "arguments": {"scheme_id": "ab_pmjay"}}`

- **ID**: `v2_repair_cat_a_015` (`en`)
  - **Task**: "Provide the official rules and entitlement for PM-Kisan Samman Nidhi."
  - **Action**: `{"name": "get_scheme_rules", "arguments": {"scheme_id": "pm_kisan"}}`

### Tool: `get_documents` (Showing 2 of 16 selected)

- **ID**: `v2_repair_cat_a_018` (`en`)
  - **Task**: "What documents are required to apply for Ayushman Bharat PM-JAY?"
  - **Action**: `{"name": "get_documents", "arguments": {"scheme_id": "ab_pmjay"}}`

- **ID**: `v2_repair_cat_a_021` (`en`)
  - **Task**: "What documents do I need to register for PM-Kisan?"
  - **Action**: `{"name": "get_documents", "arguments": {"scheme_id": "pm_kisan"}}`

### Tool: `find_facility` (Showing 2 of 12 selected)

- **ID**: `v2_repair_cat_a_024` (`en`)
  - **Task**: "Find public empanelled hospitals and service centers in Varanasi district."
  - **Action**: `{"name": "find_facility", "arguments": {"district": "Varanasi", "facility_type": "hospital"}}`

- **ID**: `v2_repair_cat_a_027` (`en`)
  - **Task**: "Locate health and wellness centers in Lucknow district."
  - **Action**: `{"name": "find_facility", "arguments": {"district": "Lucknow", "facility_type": "all"}}`

### Tool: `emergency_screen` (Showing 2 of 12 selected)

- **ID**: `v2_repair_cat_a_001` (`en`)
  - **Task**: "Patient has severe chest pain and radiating pain to left arm. Should we enroll in PM-JAY?"
  - **Action**: `{"name": "emergency_screen", "arguments": {"symptoms": ["severe chest pain", "radiating pain to left arm"]}}`

- **ID**: `v2_repair_cat_a_004` (`en`)
  - **Task**: "Accident on national highway, victim is unconscious with heavy bleeding. Which government scheme applies?"
  - **Action**: `{"name": "emergency_screen", "arguments": {"symptoms": ["unconscious", "heavy bleeding", "severe trauma"]}}`

### Tool: `create_case_summary` (Showing 2 of 10 selected)

- **ID**: `v2_repair_cat_a_028` (`en`)
  - **Task**: "Summarize verified beneficiary profile: age 68, BPL status, chronic dialysis need in Varanasi."
  - **Action**: `{"name": "create_case_summary", "arguments": {"collected_fields": {"age": 68, "income_status": "BPL", "district": "Varanasi", "need": "dialysis"}}}`

- **ID**: `v2_repair_cat_a_031` (`en`)
  - **Task**: "Record case details: 71 year old farmer, Raebareli, seeking Ayushman Vay Vandana card."
  - **Action**: `{"name": "create_case_summary", "arguments": {"collected_fields": {"age": 71, "category": "senior_citizen", "district": "Raebareli", "scheme": "ayushman_vay_vandana"}}}`

### Tool: `escalate_to_human` (Showing 2 of 10 selected)

- **ID**: `v2_repair_cat_a_032` (`en`)
  - **Task**: "My mother was denied emergency treatment at an empanelled hospital despite having an active PM-JAY card. Escalate this immediately."
  - **Action**: `{"name": "escalate_to_human", "arguments": {"case_summary": {"scheme": "ab_pmjay", "hospital_issue": "denial_of_care"}, "reason": "Empanelled hospital denied cashless treatment under active PM-JAY card."}}`

- **ID**: `v2_repair_cat_a_035` (`en`)
  - **Task**: "Severely disabled applicant cannot travel to district hospital for UDID card assessment. Need home verification escalation."
  - **Action**: `{"name": "escalate_to_human", "arguments": {"case_summary": {"scheme": "disability_card", "request": "home_visit_assessment"}, "reason": "Bedridden applicant unable to travel to medical board for disability certification."}}`

### Tool: `NO_TOOL` (Showing 2 of 8 selected)

- **ID**: `v2_repair_cat_b_001` (`en`)
  - **Task**: "Can you check what schemes I can apply for?"
  - **Action**: `"NO_TOOL (Direct Response)"`

- **ID**: `v2_repair_cat_b_002` (`en`)
  - **Task**: "Find hospitals for my treatment."
  - **Action**: `"NO_TOOL (Direct Response)"`

## 4. Complete List of Selected Record IDs

```json
[
  "v2_repair_cat_a_006",
  "v2_repair_cat_a_009",
  "v2_repair_cat_f_001",
  "v2_repair_cat_f_002",
  "v2_repair_cat_f_003",
  "v2_repair_cat_a_007",
  "v2_repair_cat_a_010",
  "v2_repair_cat_f_011",
  "v2_repair_cat_f_012",
  "v2_repair_cat_f_013",
  "v2_repair_cat_a_008",
  "v2_repair_cat_a_011",
  "v2_repair_cat_f_021",
  "v2_repair_cat_f_022",
  "v2_repair_cat_f_023",
  "v2_repair_cat_f_024",
  "v2_repair_cat_a_012",
  "v2_repair_cat_a_015",
  "v2_repair_cat_d_001",
  "v2_repair_cat_d_004",
  "v2_repair_cat_d_007",
  "v2_repair_cat_a_013",
  "v2_repair_cat_a_016",
  "v2_repair_cat_d_002",
  "v2_repair_cat_d_005",
  "v2_repair_cat_d_008",
  "v2_repair_cat_a_014",
  "v2_repair_cat_a_017",
  "v2_repair_cat_d_003",
  "v2_repair_cat_d_006",
  "v2_repair_cat_d_009",
  "v2_repair_cat_d_012",
  "v2_repair_cat_a_018",
  "v2_repair_cat_a_021",
  "v2_new_0229_en",
  "v2_new_0238_en",
  "v2_new_0247_en",
  "v2_repair_cat_a_019",
  "v2_repair_cat_a_022",
  "v2_new_0232_hi",
  "v2_new_0241_hi",
  "v2_new_0250_hi",
  "v2_repair_cat_a_020",
  "v2_repair_cat_a_023",
  "v2_new_0235_hinglish",
  "v2_new_0244_hinglish",
  "v2_new_0253_hinglish",
  "v2_new_0262_hinglish",
  "v2_repair_cat_a_024",
  "v2_repair_cat_a_027",
  "v2_repair_cat_f_005",
  "v2_repair_cat_f_008",
  "v2_repair_cat_a_025",
  "v2_repair_cat_f_015",
  "v2_repair_cat_f_018",
  "v2_new_0324_hi",
  "v2_repair_cat_a_026",
  "v2_repair_cat_f_025",
  "v2_repair_cat_f_028",
  "v2_new_0325_hinglish",
  "v2_repair_cat_a_001",
  "v2_repair_cat_a_004",
  "v2_repair_cat_c_001",
  "v2_repair_cat_c_002",
  "v2_repair_cat_a_002",
  "v2_repair_cat_a_005",
  "v2_repair_cat_c_011",
  "v2_repair_cat_c_012",
  "v2_repair_cat_a_003",
  "v2_repair_cat_c_021",
  "v2_repair_cat_c_022",
  "v2_repair_cat_c_023",
  "v2_repair_cat_a_028",
  "v2_repair_cat_a_031",
  "v2_repair_cat_g_001",
  "v2_repair_cat_a_029",
  "v2_repair_cat_g_010",
  "v2_repair_cat_g_011",
  "v2_repair_cat_a_030",
  "v2_repair_cat_g_018",
  "v2_repair_cat_g_019",
  "v2_repair_cat_g_020",
  "v2_repair_cat_a_032",
  "v2_repair_cat_a_035",
  "v2_new_0371_en",
  "v2_repair_cat_a_033",
  "v2_new_0372_hi",
  "v2_new_0375_hi",
  "v2_repair_cat_a_034",
  "v2_new_0373_hinglish",
  "v2_new_0376_hinglish",
  "v2_new_0379_hinglish",
  "v2_repair_cat_b_001",
  "v2_repair_cat_b_002",
  "v2_repair_cat_b_003",
  "v2_repair_cat_b_013",
  "v2_repair_cat_b_014",
  "v2_repair_cat_b_015",
  "v2_repair_cat_b_025",
  "v2_repair_cat_b_026"
]
```
