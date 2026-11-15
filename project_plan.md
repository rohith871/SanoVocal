# SanoVocal Project Plan

## Problem

Many users in India struggle to discover and navigate government health and welfare schemes.
They often do not know which scheme may apply, what documents are required, where to apply,
or when to seek human help.

## Solution

SanoVocal is a voice-first, multilingual, source-grounded navigator for Indian government
health and welfare schemes.

## Scope

Start with 25–40 selected schemes across:
- Health
- Maternal and child health
- Food and nutrition
- Financial inclusion
- Housing
- Livelihood and employment
- Agriculture

## Core model

Aya Expanse 8B, post-trained using QLoRA on Kaggle.

## Lexsi pipeline

1. CuratorKIT: clean, verify, and structure scheme data
2. AgentTune: train tool-use trajectories
3. SafeTune: audit and repair safety drift
4. AuditKIT: evaluate before/after performance

## Agent tools

- match_schemes
- get_scheme_rules
- get_documents
- find_help_center
- emergency_screen
- create_case_summary
- escalate_to_human

## Safety rules

- No medical diagnosis
- No treatment or dosage advice
- No final eligibility decision
- No unsupported claims
- Always cite official sources
- Escalate uncertain or high-risk cases

## Metrics

- Scheme-match accuracy
- Grounded-answer rate
- Hallucination rate
- Tool-call accuracy
- Redundant tool-call rate
- Emergency detection recall
- Unsafe-advice rate
- Over-refusal rate
- Escalation precision
- Latency

## Milestones

1. Load and inspect HF datasets
2. Select 25–40 schemes
3. Clean and verify data
4. Build knowledge base
5. Create training data
6. Train QLoRA adapter on Kaggle
7. Evaluate baseline vs fine-tuned model
8. Build voice demo
9. Publish artifacts
10. Submit project
