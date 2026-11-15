---
annotations_creators:
- expert-generated
- machine-generated-curated
language:
- en
- hi
- hne
language_creators:
- crowdsourced
- machine-generated
license: cc-by-4.0
multilinguality:
- multilingual
pretty_name: SanoVocal Dataset Collection
size_categories:
- 1K<n<10K
source_datasets:
- myScheme India
- PM-JAY Official Portal
- BharatSchemes
- Synthetic Agentic Post-Training Trajectories
tags:
- agentic-post-training
- ai-safety
- healthcare-welfare
- civic-tech
- india
- tool-calling
task_categories:
- text-generation
- conversational
- question-answering
---

# Dataset Card: SanoVocal Dataset Collection

## Dataset Summary

The **SanoVocal Dataset Collection** is a curated, high-integrity multilingual dataset collection purpose-built for agentic post-training, tool calling, clinical emergency triage, and safety alignment in civic healthcare welfare navigation. Developed for the **Lexsi AI Safety & Alignment Hackathon (Track 3: Agentic Fine-Tuning)**, the collection encompasses statutory knowledge grounding, multi-turn conversational instruction tuning, agentic trajectory rollouts with verified tool call contracts, and locked evaluation benchmarks.

All data is structured to eliminate hallucinated entitlement claims, enforce strict clinical emergency triage overrides (directing acute patients to 112/108), and constrain agentic model actions strictly within the official 7-tool SanoVocal contract across **English, Hindi, and Hinglish**.

---

## Dataset Hierarchy & Artifacts

| Dataset Artifact | File Path | Records | Modality / Format | Description |
| :--- | :--- | :---: | :--- | :--- |
| **Curated SFT Corpus** | `data/curated/curated_sft_train.jsonl` | 1,888 | Conversational JSONL (`messages`) | Multi-turn instruction tuning corpus filtered by CuratorKIT for conversational welfare counseling. |
| **Verified Knowledge Cards** | `data/curated/verified_knowledge_cards.jsonl` | 40 | Structured JSONL | Authoritative ground-truth knowledge cards for 40 statutory schemes vetted against official portals. |
| **User Scenario Benchmark Pool** | `data/evaluation/user_scenarios.jsonl` | 380 | Evaluation Scenarios JSONL | Complete pool of evaluated user queries across welfare matching, document check, facility search, and emergency triage. |
| **Agent Trajectories (Train)** | `data/traces/agent_trajectories_train.jsonl` | 300 | Multi-step Agent Trajectories | Rollout trajectories capturing user intent, multi-step thought, tool invocation, observation, and grounded responses. |
| **Agent Trajectories (Val)** | `data/traces/agent_trajectories_val.jsonl` | 35 | Multi-step Agent Trajectories | Validation split for trajectory tracking and loss monitoring. |
| **Locked Test Trajectories** | `data/traces/agent_trajectories_test.jsonl` | 45 | Held-out Agent Trajectories | Absolute locked test set (SHA256: `667E60B9...`) used for scientific before/after benchmarking and AuditKIT trace ingestion. |
| **Targeted Tool Routing Corpus** | `data/repair/sanovocal_stage2_100.jsonl` | 100 | Dual-format Messages JSONL | 100 high-precision genuine tool routing examples balancing all 7 official tools and English/Hindi/Hinglish requests. |
| **Statutory Schemes Master** | `data/selected_schemes.csv` | 40 | CSV Metadata Index | Master list of 40 flagship Indian central and state healthcare and welfare assistance programs. |

---

## Data Schemas

### 1. Curated SFT Corpus (`curated_sft_train.jsonl`)
- `record_id` (*string*): Unique record identifier (e.g., `sft_pmjay_0012`).
- `messages` (*list[dict]*): Standard chat completion format:
  - `role`: `"system" | "user" | "assistant"`
  - `content`: Text utterance. Assistant responses incorporate tool call schemas where applicable.
- `source` (*string*): Source identifier (`curatorkit_verified`).
- `timestamp` (*string*): ISO 8601 creation timestamp.

### 2. Verified Knowledge Cards (`verified_knowledge_cards.jsonl`)
- `scheme_id` (*string*): Canonical scheme identifier (e.g., `pm_jay`, `pmmvy`, `jsy`).
- `scheme_name` (*string*): Official public name in English and Hindi.
- `category` (*string*): Category tag (e.g., `maternal_health`, `tertiary_care`, `disability_support`).
- `priority` (*string*): Tier priority (`flagship`, `specialized`).
- `official_url` (*string*): Canonical government portal URL (`https://pmjay.gov.in`, `https://myscheme.gov.in`).
- `eligibility` (*dict*): Deterministic criteria including income threshold, SECC criteria, caste, age, state.
- `benefits` (*dict*): Coverage limits (e.g., INR 5,00,000 per family per year), cashless hospital empanelment.
- `documents` (*list[string]*): Mandatory statutory verification checklist (Aadhaar, Ration Card, etc.).
- `verification_status` (*string*): `"statutory_verified"`.
- `requires_human_verification` (*bool*): Flag indicating whether human ASHA / CSC assistance is mandated.
- `provenance` (*dict*): Government gazette / source portal citation.

### 3. Locked Test Set & Agent Trajectories (`agent_trajectories_test.jsonl`)
- `trajectory_id` (*string*): Unique scenario ID (e.g., `test_scenario_019`).
- `task` (*string*): Multilingual citizen utterance (English, Hindi, or Hinglish).
- `steps` (*list[dict]*):
  - `step_index` (*int*): Step order.
  - `thought` (*string*): Agent reasoning.
  - `action` (*dict*): Invoked tool and arguments (`{"tool": "emergency_screen", "arguments": {"symptoms": "chest pain and breathlessness"}}`).
  - `observation` (*dict*): Execution output from the deterministic environment.
- `final_response` (*string*): Grounded citizen-facing explanation.
- `reward` (*float*): Trajectory outcome score.
- `metadata` (*dict*): Ground truth target tool, language tag, safety critical flag.

---

## Languages & Multilinguality

The dataset targets the genuine linguistic reality of Indian civic interaction:
- **English (~40%)**: Formal administrative and conversational queries.
- **Hindi (~35%)**: High-prevalence queries in Devanagari script covering rural and suburban healthcare requests.
- **Hinglish (~25%)**: Latin-script transliterated Hindi-English code-mixed queries (e.g., *"Meri mommi ko saas lene me dikkat ho rahi hai aur chaati me dard hai"*).

---

## Data Curation & Quality Pipeline

1. **CuratorKIT Filtration**:
   - Initial pool of ~7,000 candidate records underwent CuratorKIT processing.
   - Exact deduplication, near-deduplication (MinHash LSH), toxicity/PII scrubbing, and domain relevance scoring against official statutory guidelines.
   - Retained 1,888 high-quality conversational SFT records.
2. **Statutory Knowledge Grounding**:
   - Ground truth extracted exclusively from statutory portals (`myscheme.gov.in` and `pmjay.gov.in`).
   - Every benefit claim, coverage cap, and document requirement was verified against official documentation.
3. **Agentic Trajectory Rollouts**:
   - Agent trajectories were generated using structured multi-turn interaction traces capturing both reasoning steps and explicit tool call payloads.
   - Strict tool contracts enforced: only the 7 official SanoVocal tools are permitted.
4. **Contamination & Leakage Prevention**:
   - The 45-scenario test set was strictly partitioned and cryptographically frozen at SHA256:
     `667E60B9F0EDABB2DE7232DA50A2F92E6362662A37B80701FB26BB7205366FC3`
   - Automated hash and token n-gram overlap checks confirmed **zero data leakage** (0 / 45 overlap) between training/repair corpora and the locked evaluation benchmark.

---

## Safety & Alignment Annotations

The collection contains dedicated safety-critical annotations:
- **Emergency Triage**: Queries exhibiting acute red-flag symptoms (myocardial infarction, acute dyspnea, heavy hemorrhage, severe trauma) map strictly to `emergency_screen` or deterministic emergency redirection.
- **Non-Prescription Boundary**: Prompts requesting medical diagnosis or medication prescription are annotated to trigger clinical disclaimers redirecting to licensed medical professionals.
- **Anti-Fraud Enforcement**: Prompts seeking eligibility forgery or bypass are annotated to trigger administrative refusal.

---

## Intended Use

- Post-training multilingual language models for agentic tool use in public administration.
- Benchmarking tool calling accuracy, schema validity, and safety adherence in low-resource and code-mixed languages.
- Safe AI deployment for civic technology and public healthcare access.

## Limitations

- The statutory knowledge cards reflect schemes active as of November 2026. Subsidies, annual income thresholds, and empanelled hospital lists change per state government notifications.
- The dialectal coverage is focused on Hindi and Hinglish; other Indian languages (Tamil, Telugu, Bengali, Marathi) require additional dialectal expansions.

## Licensing

- Datasets are released under **Creative Commons Attribution 4.0 International (CC-BY 4.0)**.
- Underlying statutory scheme parameters originate from the Open Government Data (OGD) Platform India.
