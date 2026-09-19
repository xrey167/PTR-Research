# NeoHorse-1-4B reference integration

NeoHorse is a published 4B Qwen3.5-derived agentic post-training reference. Its routing harness records tool interactions and capability demand, then feeds those signals into later training mixtures. That maps directly onto our typed Model-Pods, recursive replay, and Dragonfly routing layer.

## Comparison boundary

The published ten-benchmark table and sampling protocol are copied verbatim as numeric metadata in `runs/neohorse-reference-manifest-001.json`. They remain external published results. Our local probe loads the real `TokenRhythm/NeoHorse-1-4B` checkpoint on `xrserver-dev` and measures generation throughput only. It is not an OSWorld/ALE score and does not claim to reproduce the paper table.

## Integration decision

NeoHorse is represented as a `model` Pod with variant `base` (or `lora` when a Pod adapter is attached), preserving reader identity, generation binding, and revocation checks. Its routing-harness traces can enter `ExperiencePod` replay only after verifier checks, with memory frozen on held-out evaluation.

The SGLang v0.5.17 path is installed on the server for protocol-faithful serving. The current two RTX 3090 cards are occupied by the existing llama-server, so SGLang could not allocate its hybrid KV/state cache; the Transformers GPU probe is the fallback measurement.

The complete NeoHorse checkpoint is also backed up on the mounted data SSD at
`/srv/ai/extra-ssd/neural-pods-workspace/models/NeoHorse-1-4B`. The two
Safetensors shards, file count and total byte count were compared against the
workspace copy; all hashes match (`runs/xrserver-model-copy-001/report.json`).

Sources: [NeoHorse model](https://huggingface.co/TokenRhythm/NeoHorse-1-4B), [technical report](https://arxiv.org/abs/2609.08183), [official repository](https://github.com/TokenRhythm/NeoHorse).
