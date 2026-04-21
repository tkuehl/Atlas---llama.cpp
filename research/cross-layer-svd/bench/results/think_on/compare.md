# Speculative Decoding Bench — `ngram_cache` vs `baseline`

Baseline `baseline`: 25 prompts  
Condition `ngram_cache`: 25 prompts

## Per-slice summary

| Slice | Base tok/s | Cond tok/s | Speedup | Accept rate | TTFT Δ (ms) |
|---|---:|---:|---:|---:|---:|
| structured_output | 82.8 | 82.7 | 1.00× | 17.0% | -1 |
| code | 84.4 | 81.1 | 0.96× | 0.0% | +1 |
| factual_qa | 84.2 | 83.3 | 0.99× | 0.0% | -0 |
| reasoning | 83.2 | 82.1 | 0.99× | 0.0% | -2 |
| conversational | 83.3 | 80.8 | 0.97× | 0.9% | +1 |

**Overall:** baseline=83.6 tok/s, condition=82.0 tok/s, speedup=0.98×, acceptance=4.3%

## Per-prompt response preview (first 80 chars)

| ID | Slice | Baseline | Condition |
|---|---|---|---|
| code_01 | code | `` | `` |
| code_02 | code | `` | `` |
| code_03 | code | `` | `` |
| code_04 | code | `` | `` |
| code_05 | code | `` | `` |
| conv_01 | conversational | `As an AI, I don’t experience weather or have personal preferences, but I can sha` | `As an AI, I don’t experience weather or have personal preferences! But I can sha` |
| conv_02 | conversational | `**Hobby: Pottery (Wheel-Throwing or Hand-Building)**   **Why It’s Interesting:**` | `**Hobby: Pottery (Wheel-Throwing or Hand-Building)**   **Why It’s Interesting:**` |
| conv_03 | conversational | `Starting a new job is an exciting but sometimes overwhelming experience. Here ar` | `` |
| conv_04 | conversational | `` | `` |
| conv_05 | conversational | `` | `` |
| fact_01 | factual_qa | `` | `` |
| fact_02 | factual_qa | `` | `` |
| fact_03 | factual_qa | `` | `` |
| fact_04 | factual_qa | `` | `` |
| fact_05 | factual_qa | `` | `` |
| reason_01 | reasoning | `` | `` |
| reason_02 | reasoning | `` | `` |
| reason_03 | reasoning | `` | `` |
| reason_04 | reasoning | `` | `` |
| reason_05 | reasoning | `` | `` |
| struct_01 | structured_output | `` | `` |
| struct_02 | structured_output | `` | `` |
| struct_03 | structured_output | `` | `` |
| struct_04 | structured_output | `` | `` |
| struct_05 | structured_output | `{   "level": "INFO",   "handlers": ["console", "file"],   "format": "%(asctime)s` | `{   "level": "INFO",   "handlers": ["console", "file"],   "format": "%(asctime)s` |
