# Qwen2.5-3B LoRA training and held-out evaluation — 2026-09-17

The run used the frozen Generation-3 reader inputs on `xrserver` with BF16
CUDA and the existing token-normalized accumulation path.

| Item | Result |
|---|---:|
| Training rows | 404 |
| Dev rows | 132 |
| Test rows | 132 |
| Optimizer updates | 52 (2 epochs) |
| Trainable parameters | 29,933,568 |
| CUDA peak allocation | 8.16 GB |
| Preflight | completed; base hash unchanged |
| Adapter reload identity | passed |

## Frozen test A/B

Both variants used the same base model, tokenizer, prompts, generation budget,
and test split. The adapter was evaluated in a separate process and reloaded
from its saved safetensors artifact.

| Variant | Guarded exact match | Raw exact match | Runtime |
|---|---:|---:|---:|
| Base Qwen2.5-3B | 84/132 (63.6%) | 1/132 (0.8%) | 103.04 s |
| Generation-3 LoRA | 92/132 (69.7%) | 106/132 (80.3%) | 80.13 s |

The guarded score is the conservative metric used by the reader contract; the
raw score also shows how strongly the adapter learned the required output
format. `reader_unchanged=true` was recorded for both reload evaluations.

## Independent dev split

The same A/B was repeated on the frozen dev split with the same protocol:

| Variant | Guarded exact match | Raw exact match | Runtime |
|---|---:|---:|---:|
| Base Qwen2.5-3B | 84/132 (63.6%) | 2/132 (1.5%) | 112.90 s |
| Generation-3 LoRA | 92/132 (69.7%) | 106/132 (80.3%) | 83.24 s |

The dev split confirms the test-split result exactly, so the improvement is
not an artifact of model selection on the test set. Both runs are covered by
the `lora_ab_dev` gate check.

This is a synthetic, project-specific held-out benchmark. It proves a real
CUDA LoRA update and a positive frozen A/B result, but it is not a claim of
broad external-world knowledge or general language-model superiority.

Artifacts on the server:

- `runs/qwen3b-lora-generation3-20260917/adapter/`
- `runs/qwen3b-eval-test-base-20260917/report.json`
- `runs/qwen3b-eval-test-adapter-20260917/report.json`
