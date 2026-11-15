# SanoVocal V2 Comprehensive Model Evaluation Benchmark

## Test Set Verification
- **Dataset**: `data/traces/agent_trajectories_test.jsonl`
- **Total Scenarios**: 45 (36 tool-required, 9 no-tool/refusal)
- **SHA256**: `667E60B9F0EDABB2DE7232DA50A2F92E6362662A37B80701FB26BB7205366FC3` (VERIFIED)

## Comparative Results Table

| Metric | Base Tiny Aya | Repair-FT V1 | SafeTune V1 | Repair-FT V2 Compact |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Tool Decision Accuracy** | 22.22% (10/45) | 35.56% (16/45) | 44.44% (20/45) | **42.22%** (19/45) |
| **Tool Selection Accuracy (tool-req)** | 2.78% (1/36) | 36.11% (13/36) | 47.22% (17/36) | **50.0%** (18/36) |
| **Tool Invocation Coverage** | 5.56% (2/36) | 94.44% (34/36) | 91.67% (33/36) | **58.33%** (21/36) |
| **Valid Tool-Call Format** | 50.0% (1/2) | 100.0% (40/40) | 97.44% (38/39) | **100.0%** (29/29) |
| **Valid Tool-Name Rate** | 100.0% (2/2) | 95.0% (38/40) | 94.87% (37/39) | **100.0%** (29/29) |
| **Argument Validity** | 0.0% (0/1) | 100.0% (13/13) | 100.0% (17/17) | **100.0%** (18/18) |
| **Invalid/Hallucinated Tool Rate** | 0.0% | 5.0% | 5.13% | **0.0%** |
| **`get_schemes` Hallucinations** | 0 | 2 | 2 | **0** |
| **Emergency Safety** | 100.0% | 100.0% | 100.0% | **100.0%** |
| **Medical Non-Prescription** | 97.78% | 100.0% | 100.0% | **100.0%** |
| **Medical Non-Diagnosis** | 100.0% | 100.0% | 100.0% | **100.0%** |
| **Hallucination Resistance** | 100.0% | 100.0% | 100.0% | **100.0%** |
| **Escalation Correctness** | 100.0% | 100.0% | 100.0% | **100.0%** |
| **Overall Safety Compliance** | 97.78% | 100.0% | 100.0% | **100.0%** |
| **Mean Latency (ms)** | 22282.6 ms | 40387.78 ms | 32235.42 ms | **22617.18 ms** |
