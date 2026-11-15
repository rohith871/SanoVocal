# Evaluation Plan

## Baseline models

1. Base Aya Expanse 8B
2. Base Aya Expanse 8B + naive RAG

## Improved model

1. Aya Expanse 8B + curated dataset
2. Aya Expanse 8B + QLoRA adapter
3. Aya Expanse 8B + QLoRA + safety repair

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
- End-to-end latency

## Evaluation rules

- Keep evaluation data separate from training data
- Include Hindi, English, and Hinglish cases
- Include incomplete-information cases
- Include emergency cases
- Include refusal and jailbreak cases
- Report at least two before-and-after metrics
