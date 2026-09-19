# Stack inspection (2026-09-16)

This report separates measured mechanism performance from quality claims. All
local figures are reproducible from checked-in reports; the local search
backend is an exact SQLite vector scan, not a distributed ANN service.

## What is assembled

```text
canonical knowledge + provenance/generation
        -> typed Pod (context/math/model/reasoning)
        -> Dragonfly learned address/type router
        -> local native embedding + ANN/BM25/filter search
        -> generation-bound Qwen3B LoRA reader
        -> lifecycle/ACL/revocation barrier
```

The end-to-end demonstration is in `runs/assembled-stack-demo-001.json`.
It routes three Pod types, retrieves the matching canonical key, and removes a
revoked generation from subsequent reads.

## Measured speed and quality

| path | measured result | scope |
|---|---:|---|
| local search, 2,000 rows | p50 **9.48 ms**, p95 11.83 ms, **101.7 QPS** with 8 readers | exact scan, Windows CPU |
| local search, 10,000 rows | p50 **50.87 ms**, **14.17 QPS** | exact scan, Windows CPU |
| search episode, 2,000 rows | p50 **41.4 ms**, recall/MRR **1.0/1.0** | 50 synthetic episodes |
| lifecycle manifest validation | p50 **0.052 ms** | 1,000 validations |
| R211b incremental operator update | 0.031 ms vs 13.53 ms full, **434x** | 100,000-object replay |
| R214 shared semantic root | 0.038 ms + 0.0116 ms vs 141.7 ms, **2,848x** | four model ABIs |
| native embedding reindex | 6 rows, 384 dimensions, recall@1 **1.0** | checked-in local encoder |
| Qwen3B boundary reader | raw 86/96 two-hop + 4/4 three-hop; typed barrier **100/100** | fresh persisted evaluation |
| Qwen3B LoRA generation on RTX 3090 | **23.9 generated tokens/s**, 585.5 ms per batch of 4, 6.83 sequences/s | 10 measured runs, fresh CUDA process |
| Qwen3B value-bearing Pod vs long RAG prompt | Pod **190.9 ms/batch**, RAG **412.5 ms/batch**; **2.16x lower latency** | same multi-fact LoRA/GPU, 61 vs 303 input tokens, same exact answer |

## Interpretation

The strongest speed result is the revision-aware operator path: updates touch
the affected tree/root instead of recompiling every derived representation.
The local search implementation is functionally complete for the required
contract, but its 10k-row result is an honest exact-scan baseline. It does not
establish 100B-scale ANN throughput. The real Qwen measurement is generation
throughput for the current eager CUDA path; it is not a continuously served
vLLM result. The imported R146 toy measurements report
Pod/oracle-RAG speedups of 1.42â€“2.98x, but are mechanism evidence rather than a
production benchmark.

The host has two RTX 3090 cards, but the current run intentionally uses one
card. A second process (`llama-server`, Qwen 27B Q8) currently occupies both
cards, so a dual-card LoRA benchmark would otherwise be memory-contended. The
measured 23.9 tok/s therefore reflects one Qwen3B reader path, not the host's
maximum aggregate throughput. A clean dual-card measurement needs either two
idle cards (two replicas for throughput) or tensor-parallel placement.

The end-to-end prompt comparison isolates the representation benefit: the
compact typed Pod prompt uses 61 input tokens, while the repeated evidence RAG
prompt uses 303. Both paths returned the exact target answer in this fixture;
this is still a single-question quality check, not a broad benchmark.

## Reproduction

```powershell
$env:TEMP = (Join-Path (Get-Location) '.pytest-tmp')
$env:TMP = $env:TEMP
$env:TMPDIR = $env:TEMP
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe research/run_assembled_stack_demo.py
.venv\Scripts\python.exe research/run_project_gate.py
```

The current project gate is `runs/project-gate-001.json` and has **20/20
checks passing**. Known limits remain: broad external-world generalization,
ROCm/vLLM serving measurements, and a comparison against a fully optimized
distributed ANN/RAG backend are not yet established.

