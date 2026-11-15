# SanoVocal: Safety-Grounded Agentic Fine-Tuning for Voice-First Welfare Navigation

**Track**: Track 3 — Agentic Fine-Tuning  
**Primary Problem Statement**: Reliable Tool Calling  
**Secondary Focus**: Agentic Distillation & Decision Workflows in Public Healthcare Administration  

### Abstract

Millions of citizens across India encounter severe friction accessing statutory health and welfare entitlements due to fragmented bureaucratic procedures, multilingual communication barriers, and digital exclusion. While conversational AI systems could theoretically democratize welfare access, standard instruction-tuned language models present unacceptable risks in public administration: hallucinating non-existent welfare entitlements, fabricating eligibility rules, generating syntactically invalid tool calls, and critically, neglecting life-threatening clinical emergencies while attempting conversational scheme exploration.

**SanoVocal** addresses these challenges through an end-to-end, safety-grounded agentic post-training pipeline deployed on **CohereLabs/tiny-aya-global** (3B parameters), a multilingual foundation model optimized for regional Indian languages. Using **CuratorKIT**, we filtered and verified a whitelist of 40 flagship national schemes into structured statutory knowledge cards. Leveraging **AgentTune**, we synthesized multi-turn citizen trajectories across English, Hindi, and Hinglish. We then conducted agentic post-training using 4-bit NF4 QLoRA on a targeted corpus of validated multi-turn trajectories, aligning the model to a strict contract of seven official tools: `emergency_screen`, `match_schemes`, `get_scheme_rules`, `get_documents`, `find_facility`, `create_case_summary`, and `escalate_to_human`.

We benchmarked the **SanoVocal fine-tuned adapter** against the baseline **Tiny Aya Global base model** on a frozen, held-out 45-scenario test set (`SHA256: 667E60B9F0EDABB2...`). The fine-tuned adapter demonstrated substantial, measurable gains in tool reliability:
- **Tool Selection Accuracy**: Improved from **41.67% to 52.78%** (+11.11 percentage points).
- **Tool Invocation Coverage**: Rose from **44.44% to 63.89%** (+19.45 percentage points).
- **Overall Decision Accuracy**: Advanced from **42.22% to 46.67%** (+4.45 percentage points).
- **Format Validity & Argument Validity**: Maintained at **100%** under Pydantic contract validation.
- **Safety Compliance**: Reached **100.0%** (up from 97.78%), deterministically overriding conversational flows to emergency triage (directing citizens to 112/108) when acute symptoms (e.g. chest pain, respiratory distress) are detected.

To validate trace ingestion under formal evaluation, all 45 held-out trajectories were processed through **AuditKIT** (30 agent tool traces and 15 direct response turns). The available evaluator registered outcomes as ineligible for formal task-success certification, which we document as an honest evaluation boundary.

In production, SanoVocal pairs the fine-tuned adapter with a real voice engine featuring local Speech-to-Text via Faster-Whisper (`base`, CPU int8 with 0 MB GPU overhead), speech synthesis via gTTS, an in-browser microphone capture interface, and a production FastAPI backend. The entire model operates on an NVIDIA RTX 3050 Laptop GPU (consuming only 2.45 GB of its 6 GB VRAM), demonstrating that safety-grounded, reliable agentic tool calling can be achieved efficiently on consumer-grade hardware for high-impact civic applications.
