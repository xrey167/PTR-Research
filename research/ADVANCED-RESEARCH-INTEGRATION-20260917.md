# Advanced tracks: Duplex, SID, Multiplex

## vLLM-Omni Realtime Duplex

Duplex is a runtime contract: sessions have server-allocated IDs, ordered events, interruption epochs, playback acknowledgement, reconnect/resume and capability negotiation. We use it for a **runtime Pod**, not as ordinary SFT knowledge. The runtime adapter must preserve `session_id`, `server_event_seq`, `epoch`, `turn_id`, `response_id`, deadlines and ACL context. Evaluation measures first-text/audio latency, audio cadence, resume correctness, barge-in behavior and fallback to turn-based HTTP. Current vLLM-Omni documents MiniCPM-o 4.5 as the deployed duplex plugin, so this is not assumed for Qwen without an adapter/plugin.

Source: [vLLM-Omni Realtime Duplex](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/realtime_duplex_api)

## SID-style retrieval Pod

The research Pod should be trained as a document-ranking subagent, not as a final answer generator. Each rollout can issue BM25, ANN, metadata and graph calls, then return an ordered document list. Reward uses NDCG plus latency, tool efficiency, format and provenance. This avoids the recall-only shortcut of returning every seen document and keeps the Pod composable with the main reader. Start with SFT for tool format, then GRPO with 4–16 attempts and a corpus-specific gold-document set.

Sources: [SID-1](https://www.sid.ai/research/sid-1), [SID-1 technical report](https://www.sid.ai/research/sid-1-technical-report)

## Multiplex Thinking

Multiplex Thinking is an experimental reasoning mechanism that samples multiple candidate tokens at each step, aggregates them into a continuous multiplex token, and optimizes the trajectory with on-policy RL. We keep it isolated to a reasoning/math Pod. The new dataset records branch count, confidence-weighted merge, correctness and trajectory length; it does not pretend that a normal Qwen model can consume multiplex tokens. Integration requires the repository's model/trainer changes first, followed by a toy branch/merge smoke test and a discrete-CoT baseline.

Source: [Multiplex Thinking](https://huggingface.co/papers/2601.08808)

## LoftQ

LoftQ remains a quantization initialization path for LoRA: quantization and low-rank initialization are optimized jointly to reduce the gap to full precision. It is applied after the functional baseline and evaluated on an unseen holdout; it is not a replacement for the retrieval or reasoning objectives.

Source: [LoftQ](https://arxiv.org/abs/2310.08659)

## Generated advanced dataset

`runs/advanced-tracks` contains 150 records per split:

- `sid_retrieval`: 50 document-ranking/RL records;
- `multiplex_reasoning`: 50 experimental reasoning records;
- `duplex_runtime`: 50 runtime protocol records.

The generator is `research/prepare_advanced_tracks.py`. Duplex rows are evaluation/contract data only and must not be sent through the reader SFT trainer.
