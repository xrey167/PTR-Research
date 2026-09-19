# LoRA-Ensemble-Serving über vLLM Multi-LoRA — 2026-09-19

## Aufbau

- **D1 Pod-Publikation:** `research/publish_adapter_pods.py` registriert
  reader-gen3 (Fallback) und reader-gen5 (promotiert) als versionierte
  Model-Pods in der Registry mit vollständiger Provenance-Kette
  (dataset protocol hash → training report hash → adapter file hashes →
  reader identity). Inventar: `runs/adapter-pod-registry-001/inventory.json`.
- **D2 Serving:** `research/run_vllm_ensemble.sh` startet zwei vLLM-Replicas
  (qwen3b-Basis, je eine GPU) mit `--enable-lora` und beiden Adaptern
  (`--lora-modules reader-gen3=… reader-gen5=…`, max-lora-rank 32).
  Notwendig: `VLLM_USE_FLASHINFER_SAMPLER=0` und
  `VLLM_ENABLE_V1_MULTIPROCESSING=0` (bekanntes Projektthema).
- **Router:** `VllmReplicaRouter` hat jetzt zusätzlich `completion()`
  (raw `/v1/completions` mit identischem Failover wie `chat()`) — nötig,
  weil die Reader-Prompts das Prefill-Suffix-Format von evaluate_reader
  nutzen (Prefix+Suffix über den Chat-Template-Punkt hinaus).

## Gemessene Ergebnisse (frozen test split, 132 Cases)

| Metrik | peft-Eval (Gen-5) | **vLLM-LoRA-Serving (Gen-5)** |
|---|---:|---:|
| Raw exact match | 119 | **119** |
| Guarded exact match | 92 | **92** |

Serving-Identität: Der vLLM-LoRA-Pfad reproduziert die Trainings-Eval-
Zahlen exakt — keine Qualitätseinbuße durch das Serving.

## Ensemble-Fallback

13 der 132 Fälle misslingt Gen-5 wörtlich; für sie liefert der Router die
Anfrage an reader-gen3 weiter (`fallback_used=13`). Gen-3 korrigierte 0
dieser Fälle (die Fehlermengen überlappen) — union bleibt 119/92. Der
Fallback-Pfad ist funktional gemessen (Routing, Latenz, Response-Ok), bringt
aber keinen Qualitätszuwachs auf diesem Split; ein heterogenerer zweiter
Pod wäre der Hebel (z. B. größerer Base-Modell-Pod in Gen-6).

## Failover

GPU-0-Replica hart gestoppt → 32/32 Requests erfolgreich über GPU-1,
16 Failovers, 0 Fehler. Gate-Checks:

- `ensemble_routing` (132 ok, 0 errors, Gen-5-Zahlen == peft-Eval) grün
- `ensemble_failover` (32 ok, 16 failovers) grün

Messdatei: `research/runs/ensemble-20260919.json`.

## Betrieb

```bash
research/run_vllm_ensemble.sh start   # 2 Replicas, beide Adapter, Ports 18000/18001
research/run_vllm_ensemble.sh stop   # beide runter, GPUs frei
PYTHONPATH=. python research/benchmark_ensemble_router.py
```
