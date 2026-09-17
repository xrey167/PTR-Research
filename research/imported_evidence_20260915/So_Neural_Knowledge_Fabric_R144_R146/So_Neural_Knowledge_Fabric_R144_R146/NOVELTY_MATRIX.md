# Novelty matrix — September 12, 2026

| Prior work | Relevant capability | Consequence for So |
|---|---|---|
| Larimar (ICML 2024) | LLM-agnostic episodic memory, one-shot knowledge updates, selective forgetting/leakage controls | Editable external neural memory is not novel. |
| MindBridge (ACL Findings 2025) | Cross-model knowledge editing through a separate memory modality | Cross-model memory decoupling is not novel. |
| NeuralDB (ICLR 2026) | neural KV database, gated retrieval, append/modify/delete, scaling to 100k edits | Neural KV edit store and large edit capacity are not novel. |
| LMLM (ICLR 2026) | pretraining teaches external factual lookup instead of memorizing facts | “weights reason, external DB stores facts” is not novel by itself. |
| Knowledge Externalization (ICLR 2026) | external memory tokens; forget/restore/edit/composition | reversible modular knowledge tokens are not novel. |
| NeuRAG (ACL Findings 2026) | documents encoded as LoRA-like Hyper-Neurons, dynamically fused into model inference | neuralized RAG / deep model integration is not novel. |
| ChronoMem (2026) | memory version control, snapshots and semantic rollback | versioning/rollback of agent memory is not novel. |
| XMemTransfer (Aug 2026) | frozen external memory reused across backbones via target-side reader | cross-model memory portability is not novel. |

## Candidate differentiator to test

A *revision-native neural knowledge execution model* where the canonical truth is a model-neutral immutable knowledge revision, model-specific Neural Images are compiled cache lines, requests execute under snapshot isolation, mutable facts are excluded from backbone memorization by changing-world training, and any persistent derived neural state must carry either a certified lifecycle correction algebra or a generation dependency that invalidates/replays it.

No novelty claim should be made until this exact combination is audited against papers and patents and demonstrated on pretrained LLMs.
