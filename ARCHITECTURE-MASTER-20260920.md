# Neural-Pods v1.0 — Gesamtarchitektur (Master-Dokument)

Stand: 2026-09-20 · Gate 40/40 grün · 305 Tests · 33 Commits ·
GitHub `xrey167/PTR-Research` · Server-Klon pull-basiert.

Dieses Dokument integriert alle fünf Design-Dokumente zu einer kohärenten
Architektur und legt die Umsetzung der offenen Bausteine fest.

## 1. Schichtenmodell (Abhängigkeitsgraph, von unten nach oben)

```
┌─────────────────────────────────────────────────────────────────┐
│ GATE   verify_architecture_gate.py — fail-closed, 40+ Checks    │
├─────────────────────────────────────────────────────────────────┤
│ POD-ARM / DREAM (Schicht 5)                                     │
│ Reflex-Kanal (reflex.py) · Dream-Pod (dream.py, validiert:      │
│ 125 predicted → 125 real) · Improve-Kreislauf                   │
├─────────────────────────────────────────────────────────────────┤
│ NERVENSYSTEM (Schicht 4)                                        │
│ Mesh (mesh.py, RTT 0,3 ms) · native Protokolle (native_comm.py, │
│ 100 % Frame-Validität) · TaskGraph (taskgraph.py, 2,59×)        │
│ PerceptionStream (perception.py, 13,3k Events/s)                │
├─────────────────────────────────────────────────────────────────┤
│ HAUSHALT (Schicht 3) — Colibri/Lumabri-Verhalten                │
│ Household (household.py): join/offer/approval/BUSY/             │
│ calibration/token-cache · DatasetStore (H7) · Training (H5)     │
├─────────────────────────────────────────────────────────────────┤
│ STORAGE (Schicht 2) — Vier-Tier-Modell                          │
│ L0 Pod-lokal (crossbeam-Muster, jemalloc) · L1 Redis (0,3 ms)   │
│ L2 LanceDB (Vektoren/Dokumente/Traces) · L3 Snapshots (SSD/S3)  │
│ PodStorage-Fassade (storage.py) — die einzige Pod-API           │
├─────────────────────────────────────────────────────────────────┤
│ KERN (Schicht 1)                                                │
│ Registry (SQLite-Provenance) · PodExecutor-Factory ·            │
│ Raft-Binding (Rust/PyO3) · Serving (vLLM Multi-LoRA)            │
└─────────────────────────────────────────────────────────────────┘
```

Abhängigkeitsregel: eine Schicht spricht nur die Schicht darunter an.
Pods sprechen Storage nur über die PodStorage-Fassade, nie Backends direkt.

## 2. Komponentenlandkarte (Stand 2026-09-20)

### Implementiert + validiert (Gate-Checks)

| Komponente | Datei | Gate-Check(s) | Gemessen |
|---|---|---|---|
| Registry/Provenance | registry.py | tests, authenticated_transport | — |
| Reader Gen-3→7 | research/train_reader.py | lora_ab…gen5_dev, gen7_dream_validated | raw 125/124 |
| Dream-Pod | dream.py + run_dream_cycle.py | dream_pipeline | Backtest 0,01 |
| vLLM Multi-LoRA | research/run_vllm_ensemble.sh | ensemble_routing/failover | 20/20 acks |
| Mesh | mesh.py | mesh_presence | RTT 0,30 ms |
| Native Protokolle | native_comm.py + train_native_comm.py | native_protocol | Validität 1.0 |
| Task-Graph | taskgraph.py | taskgraph_parallel | Speedup 2,59× |
| Mesh-Cache | mesh_cache.py | mesh_cache | Cross-Knoten ✓ |
| Executor-Factory | pod_executor.py | (tests) | deterministisch ✓ |
| Reflex | reflex.py | reflex_dispatch | Mechanik ✓ |
| Perception | perception.py | (Benchmark) | 13,3k Events/s |
| Token-Cache | household.py | (tests) | saved_tokens ✓ |

### Implementiert, aber ungeprüft/defekt

| Komponente | Problem | Fix geplant in |
|---|---|---|
| household.py H1–H3/H6 | Tests schlagen fehl: registry.events() fehlt als öffentliche API | F1 |
| perception.py, mesh_cache.py | keine Unit-Tests (nur Benchmarks) | Tests in H4/S2 nachziehen |

### Design-only (Umsetzung in diesem Plan)

| Baustein | Design | Umsetzungsphase |
|---|---|---|
| PodStorage-Fassade | STORAGE-ARCHITECTURE §4 | F3 |
| LanceDB-L2 | STORAGE §2 | F4 |
| DatasetStore | HOUSEHOLD (H7) | H7 |
| Training auf Donoren | HOUSEHOLD (H5) | H5 |
| KVCache-Affinity | STORAGE §1 | S2 |
| jemallocator im Raft-Binding | STORAGE §3 | S3 |
| vLLM-Multistream | STORAGE §3 | S4 |
| Work-Stealing | STORAGE §3 | S5 |
| Household-E2E über Mesh | HOUSEHOLD (H4) | H4 |

## 3. Entscheidungs-Records (ADR)

| ADR | Entscheidung | Begründung |
|---|---|---|
| ADR-1 (turbopuffer) | L3 = Objekt-/SSD-Snapshots, Compute ephemeral | 3 System-Freeze-Hänger haben gezeigt: Dauerzustand im Pod verliert Daten |
| ADR-2 (SurrealDB) | Multi-Model über LanceDB, NICHT zweite Engine | Registry-DAG bleibt SQLite-Kontroll-Ebene; Lance deckt Vektor+Dokument+Versionen ab |
| ADR-3 (Mooncake) | KVCache-Session-Affinity statt KV-Transfer | vLLM exponiert KV-Caches nicht stabil transferierbar; Affinity über Redis jetzt messbar |
| ADR-4 (crossbeam) | Work-Stealing-Muster im TaskGraph/Mesh | bounded queues + deadline drops existieren; Stealing-Semantik kommt mit S5 |
| ADR-5 (jemallocator) | Rust-Binding auf jemallocator, Gate = before/after | Raft-Ready-Frames sind viele kleine Allokationen; Gewinn muss messbar sein |
| ADR-6 (lumabri) | Donor-Approval + BUSY als Haushaltsvertrag | Provenance-Events machen Approvals auditierbar; BUSY ersetzt stillen Override |
| ADR-7 (colibri) | Tier-Preference vram→ram→disk, harte Semantik | Präzision steht im Pod-Manifest; Änderung = neue Generation |
| ADR-8 (Datasets) | Frozen JSON führend, Lance als abfragbarer Spiegel | bewährte train_reader-Pipeline bleibt; Lance liefert SQL-Abfragen |

## 4. Umsetzungsplan (Abhängigkeitsreihenfolge)

```
F1 registry.events() ──┐
F2 Dependencies ───────┼─▶ F3 PodStorage-Fassade ─▶ F4 LanceDB-L2
                       │                              │
H4-fix test_household ◀┘                              ▼
H5 request_training ◀────────────── H7 DatasetStore ◀─┘
H4 Household-E2E ◀── H5/H7
S2 KVCache-Affinity ── S3 jemalloc ── S4 Multistream ── S5 Work-Stealing
Getracer Regressionslauf über alles → README/HANDOVER v1.0 → Gate 48/48
```

| Phase | Gate-Checks (neu) | Endstand |
|---|---|---|
| F1–F4 | storage_facade, storage_l2_lance | 42/42 |
| H4/H5/H7 | household_e2e, household_training, dataset_store | 45/45 |
| S2–S5 | kvcache_affinity, raft_jemalloc, vllm_multistream, work_stealing | 49/49 |
| Abschluss | Regressionslauf über PodStorage | 50/50 |

## 5. Sicherheitsregeln (unverändert, über alle Schichten)

1. Promotion nie ohne Gate · 2. Rollback via Supersedes/Snapshots ·
3. Autonomie-Budget im Event-Log · 4. Delete-Tokens als Endabschaltung ·
5. Frozen Splits tabu · 6. Egress-ACL je Pod · 7. Fail-closed Parser ·
8. Kdump-Crash-Capture aktiv (System-Freeze-Ursache lesbar beim nächsten Mal)

## 6. Betriebs-Fakten (Server)

Root 591 GB (480 frei, LVM +700 GiB Reserve) · /srv/ai 492 GB ·
/mnt/neural-data 932 GB (ntfs3, fstab) · NVMe-Namen wechseln zwischen Boots
(UUIDs in fstab) · Mosquitto in np-node1 · Redis in np-node1 · Postgres 16 in
allen 3 LXD-Knoten · kdump aktiv · GPU-Läufe einzeln via nohup.
