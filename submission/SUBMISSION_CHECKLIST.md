# Lexsi AI Safety & Alignment Hackathon — Submission Checklist

**Project Name**: SanoVocal  
**Track**: Track 3 — Agentic Fine-tuning  
**Primary Problem Statement**: Reliable Tool Calling  
**Date**: November 15, 2026  

---

### Mandatory Compliance & Deliverables Checklist

- [x] **Abstract < 500 words** — Verified: 423 words located in [`submission/ABSTRACT.md`](file:///c:/Users/KOYA%20ROHITH/SanoVocal/submission/ABSTRACT.md).
- [ ] **Public GitHub repo** — Codebase prepared and sanitized; pending user push to `https://github.com/[YOUR-USERNAME]/sanovocal`.
- [ ] **Public Hugging Face model repo** — Model card prepared in [`submission/MODEL_CARD.md`](file:///c:/Users/KOYA%20ROHITH/SanoVocal/submission/MODEL_CARD.md); pending user upload to `https://huggingface.co/[YOUR-USERNAME]/sanovocal-adapter`.
- [ ] **Public Hugging Face model collection** — Collection placeholder: `https://huggingface.co/collections/[YOUR-USERNAME]/sanovocal-models`.
- [ ] **Public Hugging Face dataset repositories** — Dataset card prepared in [`submission/DATASET_CARD.md`](file:///c:/Users/KOYA%20ROHITH/SanoVocal/submission/DATASET_CARD.md); pending user upload to `https://huggingface.co/datasets/[YOUR-USERNAME]/sanovocal-datasets`.
- [ ] **Public Hugging Face dataset collection** — Collection placeholder: `https://huggingface.co/collections/[YOUR-USERNAME]/sanovocal-datasets`.
- [ ] **Public demo / deployment** — Streamlit app functional locally at `http://localhost:8501`; public deployment link placeholder: `https://sanovocal.streamlit.app/`.
- [x] **README / write-up** — Comprehensive public README available in [`README.md`](file:///c:/Users/KOYA%20ROHITH/SanoVocal/README.md) and [`submission/README.md`](file:///c:/Users/KOYA%20ROHITH/SanoVocal/submission/README.md).
- [x] **Reproducibility instructions** — Exact instructions, seed, hardware, and environment commands documented in README and evaluation summary.
- [x] **Before / after metrics** — Multiple measured metrics reported (Selection Accuracy, Invocation Coverage, Overall Decision Accuracy, Validity, Safety Compliance).
- [x] **Track specified** — Track 3: Agentic Fine-tuning.
- [x] **Problem statement specified** — Primary: Reliable Tool Calling; Secondary: Agentic Distillation & Decision Workflows.
- [x] **Base model specified** — `CohereLabs/tiny-aya-global` (3B parameters).
- [x] **Training data documented** — Verified records (1,888 SFT records, 40 knowledge cards, 300 training trajectories, 100 final contract records).
- [x] **Evaluation data documented** — Held-out 45-scenario test set documented in [`data/traces/agent_trajectories_test.jsonl`](file:///c:/Users/KOYA%20ROHITH/SanoVocal/data/traces/agent_trajectories_test.jsonl).
- [x] **Locked test hash documented** — SHA256: `667E60B9F0EDABB2DE7232DA50A2F92E6362662A37B80701FB26BB7205366FC3`.
- [x] **Adapter hash documented** — SHA256: `39291232E229F7B5B3A03A424298CB67025CC958E4D09B59564588C6AB44FCAC`.
- [x] **No secrets committed** — Repository-wide automated scan completed (0 active tokens/keys in tracked files).
- [x] **No fake metrics** — Exact, non-fabricated metrics benchmarked directly on the frozen evaluation set.
- [x] **No unsupported claims** — AuditKIT limitations accurately reported (trace coverage verified; ineligible for formal outcome certification).
