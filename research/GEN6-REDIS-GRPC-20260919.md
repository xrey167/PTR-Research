# Gen-6 heterogener Pod, Redis-L2-Tier, gRPC-Entscheidung — 2026-09-19

## Gen-6: heterogener Fallback-Pod (NeoHorse-1-4B)

`research/prepare_generation6.py` richtet dieselben frozen Gen-5-Cases auf
dem NeoHorse-1-4B-Basis-Modell ein (frisches verifiziertes Manifest,
modell-path umgebogen). Training: 824 Zeilen, Preflight 9,7 GB CUDA-Peak.

| Pod | Base | Test raw | Test guarded |
|---|---|---:|---:|
| Gen-5 (promotiert) | Qwen2.5-3B | 119 | 92 |
| **Gen-6** | **NeoHorse-1-4B** | **125** | **92** |

`research/run_vllm_hetero.sh` served beide Basen gleichzeitig (GPU0:
qwen3b+gen3+gen5, GPU1: NeoHorse+gen6). `research/benchmark_hetero_ensemble.py`
messen pro Pod eigene Tokenizer-Templates (wichtig: ein gemischter Prompt-
Template-Pfad drückt Gen-5 auf 43 raw — Template-Identität pro Pod ist
Pflicht).

**Heterogene Union (Gen-5 primär → Gen-6-Fallback): raw 126/132** — Gen-6
korrigiert 7 der 13 Gen-5-Fehlfälle. Bemerkenswert ehrlich: Gen-6 allein
(125) ist fast stärker als die Union; NeoHorse-4B ist auf diesem
synthetischen Benchmark der bessere Basis-Modell-Kandidat. Konsequenz für
Gen-7: Prompt-Template-Kontrast vermeiden und den Promoted-Pod auf dem
stärkeren Base evaluieren.

## Redis-L2-Cache-Tier

`PodCache` hat einen optionalen Redis-Tier (cache-aside, ACL-safe: der Key
trägt namespace/branch/principal/generation/revision über den kanonischen
CacheKey-Digest; TTL; best-effort mit Error-Zähler). Redis läuft in
np-node1 (Containernetz). Benchmark (`research/benchmark_redis_cache_tier.py`):
L2-only-Hits nach LRU-Neustart über Netz **p50 0,38 ms / p99 0,76 ms**,
2000/2000 redis_hits, 0 Fehler. Damit ist die 3-Tier-Hierarchie aus der
Design-Doku (Prozess-LRU → Redis → NVMe) implementiert und gemessen.

Fix während der Arbeit: Der erste Patch hatte einen Deadlock (Owner eines
inflight-Events wartete auf sich selbst) — behoben; `socket_connect_timeout`
ist jetzt gesetzt, damit ein nicht erreichbarer Redis nicht blockiert.

## gRPC-Entscheidung (grpc-rs bleibt verworfen)

`research/benchmark_grpc_vs_tcp.py` (256-Byte-Payload, 2000 Round-Trips,
gleicher Host): Length-framed TCP **p50 0,024 ms / 40.704 req/s** vs.
gRPC unary **p50 0,228 ms / 4.184 req/s** — TCP ist 9,7× schneller. Die
in RUST-TRANSPORT-CONCURRENCY dokumentierte Zurückhaltung ist damit
datengestützt; gRPC bleibt nur für Cross-Language-Peer-Interop relevant.

## Gate

Neue Checks (alle grün): `gen6_hetero_pod`, `ensemble_hetero_union`,
`redis_cache_tier`, `grpc_transport_decision` — Gate steht bei **28/28**,
268 Tests bestanden.

Messdateien: `research/runs/{ensemble-hetero,redis-cache,grpc-vs-tcp}-20260919.json`,
`runs/qwen3b-eval-test-gen6-20260919-report.json`.


## Gen-6 Promotion (2026-09-19, final)

Dev-Split-Eval bestätigt: **raw 124, guarded 92, typed 80, concept 44/44
(periode)** — Gen-6 dominiert Gen-5 per Familie auf beiden Splits und ist
damit der promoted Reader-Adapter. Gate-Check `gen6_promoted_dev` ergänzt;
Gate steht bei **30/30 grün** (268 Tests).

| Split | Gen-5 (Qwen3B) | **Gen-6 (NeoHorse, promoted)** |
|---|---:|---:|
| Test raw / guarded | 119 / 92 | **125 / 92** |
| Dev raw / guarded | 124 / 92 | **124 / 92** |
| Dev typed / concept | 76 / 43 | **80 / 44** |
