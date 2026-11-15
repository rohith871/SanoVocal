# SanoVocal — Phase A Data Curation Report
**Toolkit**: CuratorKIT (v0.1.21)  
**Execution Timestamp**: 2026-11-09T08:01:23.497682+00:00  
**Status**: COMPLETED & VALIDATED

---

## 1. Summary of Execution

- **Candidate Training Pool Input**: 1,900 rows
  - Source `bharatschemes`: 1,828
  - Source `multilingual_qa`: 72
- **CuratorKIT Ingestion & Gating**:
  - `JSONLReader`: 1,900 valid JSON objects read (0 parse errors)
  - `SchemaGate` (token bounds 5–2,048, instruction/output presence): 1888 passed, 12 rejected (excessive scraped context above 2,048 tokens quarantined)
  - `ExactDeduplicator`: 1888 unique records preserved
  - `TextCleaner`: Unicode normalization, HTML stripping, and whitespace collapsing applied to all records
- **Accepted Training Records**: 1888 (99.37%)
- **Quarantined Records**: 12 (0.63%)
  - Rejection Reason: `above_max_tokens` (scraped web pages exceeding 2,048 token boundary)

---

## 2. Deliverables Generated

1. `data/curated/curated_sft_train.jsonl`: Cleaned, validated conversational messages format with stable IDs (`sv_curated_XXXXX`).
2. `data/curated/verified_knowledge_cards.jsonl`: 40 official knowledge cards covering 7 key domains (Health, Maternal Care, Financial Inclusion, Housing, Utilities, Agriculture, Vulnerable Groups).
3. `data/curated/rejected_records.jsonl`: Fully auditable rejection log with per-sample reasons.
4. `data/curated/curation_manifest.json`: End-to-end metadata, stage counts, and provenance trail.

---

## 3. Provenance & Reproducibility

Every accepted sample preserves provenance traceability back to either BharatSchemes or the Multilingual QA dataset.
Reproducible locally via:
```powershell
python scripts/01_extract_candidate_pool.py
python scripts/02_create_selected_schemes.py
python scripts/04_curatorkit_pipeline.py
```
