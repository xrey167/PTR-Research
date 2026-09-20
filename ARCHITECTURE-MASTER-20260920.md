# Neural-Pods v1.0 — Gesamtarchitektur (Master-Dokument)

**Dieses Dokument ist die einzige Quelle der Wahrheit für den Projektstand.**
README und HANDOVER verweisen hierher und führen keine eigenen Zählstände mehr.

## Stand (2026-09-20, aus einem frischen Klon nachgemessen)

| Größe | Wert | womit geprüft |
|---|---|---|
| Gate-Checks definiert | **44** | `research/verify_architecture_gate.py` |
| Gate-Checks grün im Klon | **37** | `python research/verify_architecture_gate.py` |
| Gate-Checks rot | **7** — alle mangels Evidenz, keine Regression | Gate nennt die 8 fehlenden Dateien |
| Tests | **381 passed, 0 failed, 8 skipped** | `python research/record_test_run.py` |
| Module `neural_pods/` | **56**, alle einer Schicht zugeordnet | `python neural_pods/architecture.py` |
| Schichtverstöße | **0** | Gate-Check `layering` |

Die sieben roten Checks brauchen Eval-Reports, die nur auf dem Server liegen
(`runs/`, gitignoriert). Die `.gitignore`-Ausnahme `!research/runs/*.json`
existiert; es fehlt ein Commit vom Server, dann sind 43/43 aus einem Klon
prüfbar. Der Befundbericht dazu: `research/STATE-DEEP-RESEARCH-20260920.md`.

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

**Beide Sätze sind seit 2026-09-20 geprüft, nicht behauptet.**
`neural_pods/architecture.py` führt das Schichtenmodell als Daten
(`LAYER_OF`, `BACKEND_OWNERS`) und prüft den Baum statisch dagegen:
Import nach oben, Backend-Zugriff an der Fassade vorbei, Laufzeit-Zyklus
oder ein Modul ohne Schichtzuordnung sind Verstöße. Der Gate-Check
`layering` und `tests/test_architecture_layers.py` (17 Tests, inklusive
Gegenproben an einem synthetischen Paket) führen ihn aus. Importe in
`if TYPE_CHECKING:` zählen nicht als Laufzeitkante — `ranking` und
`local_search` verweisen genau so aufeinander, was sonst als Zyklus
erschiene. Aktueller Stand: 56 Module, 28 Laufzeitkanten, 0 Verstöße.

Ein neu angelegtes Modul ohne Eintrag in `LAYER_OF` lässt den Check
durchfallen: die Einordnung wird einmal bewusst entschieden statt später
entdeckt.

## 2. Komponentenlandkarte (Stand 2026-09-20)

### Implementiert + validiert (Gate-Checks)

| Komponente | Datei | Gate-Check(s) | Gemessen |
|---|---|---|---|
| Registry/Provenance | registry.py | tests, authenticated_transport | — |
| Reader Gen-3→7 | research/train_reader.py | lora_ab…gen5_dev, gen7_dream_validated | raw 125/124 |
| Dream-Pod | dream.py + run_dream_cycle.py | dream_pipeline | out-of-sample: **0 Generationen** |
| vLLM Multi-LoRA | research/run_vllm_ensemble.sh | ensemble_routing/failover | 20/20 acks |
| Mesh | mesh.py | mesh_presence | RTT 0,30 ms (zwei Endpunkte **auf einem Host**) |
| Native Protokolle | native_comm.py + train_native_comm.py | native_protocol | Frames 1.0, **exact 0.55** |
| Task-Graph | taskgraph.py | taskgraph_parallel | mean_concurrency 2,59 (**kein Speedup**) |
| Mesh-Cache | mesh_cache.py | mesh_cache | Cross-Knoten ✓ |
| Executor-Factory | pod_executor.py | (tests) | deterministisch ✓ |
| Reflex | reflex.py | reflex_dispatch | Mechanik ✓ |
| Perception | perception.py | (Benchmark) | 13,3k Events/s |
| Token-Cache | household.py | (tests) | saved_tokens ✓ |

### Implementiert, aber ungeprüft/defekt

| Komponente | Problem | Status |
|---|---|---|
| household.py H1–H3/H6 | registry.events() fehlte als öffentliche API | **erledigt (F1)** |
| storage.py (F3/F4) | sechs Defekte, u. a. doppelt geschriebener Erstbatch | **erledigt**, Gate-Checks `storage_facade`/`storage_l2_lance` |
| household.py H2 | BUSY wurde nie freigegeben, start() ohne Event, VRAM/RAM ein Zähler | **erledigt**: `release()`, `restore_from_events()` |
| taskgraph.py | `speedup` maß Contention | **erledigt**: `mean_concurrency` + `critical_path_ratio` |
| Reader-Eval | frozen Split gesättigt **und aus dem Repo nicht regenerierbar** (132 vs. 96 Zeilen) | Holdout-Split erzeugt, Bewertung offen |
| perception.py, mesh_cache.py | keine Unit-Tests (nur Benchmarks) | offen |

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
| ADR-9 (A2A/MCP) | Eigener Dialekt bleibt, Signaturmodell wird übernommen | A2A v1.0 (Jan 2026, Linux Foundation/AAIF) löst Interoperabilität; unser Ziel ist ein anderes: Wegfall des Tool-Use-Overheads im Reflex-Pfad. Was A2A besser gelöst hat, ist die Authentizität — signierte Agent Cards. Deshalb: Dialekt behalten, aber `mesh.py` signiert Envelopes per HMAC (`secret=`), sonst bliebe `principal` ein Etikett |
| ADR-11 (ausführbare Architektur) | Schichtenmodell und Fassadenregel als Daten + Gate-Check, nicht als Prosa | Genau die Regeln, die nur im Dokument standen, waren die, die unbemerkt driften konnten. `architecture.py` macht sie zur geprüften Eigenschaft; der Preis ist eine Tabelle, die bei jedem neuen Modul eine Entscheidung erzwingt |
| ADR-10 (Adapter-Pool) | Vier-Tier-Modell deckt L1–L3, **nicht** den GPU-Adapterpool | S-LoRA (Unified Paging, 2.000 Adapter, bis 4× Durchsatz) und Punica (SGMV-Kernel) zeigen: der Sprung von 2 auf viele Adapter ist ein Problem des GPU-Speicherpools und des Batching-Kernels, nicht der Registry. Offener Baustein, bewusst noch nicht terminiert |

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

Zwei dieser Regeln beschreiben weniger, als ihr Wortlaut nahelegt — hier steht,
was sie tatsächlich leisten:

- **Regel 6 (Egress-ACL):** Die Topic-ACL wird im Client geprüft, nicht im
  Broker, und `publish_raw()` umgeht sie bewusst. Das ist eine Leitplanke für
  kooperierende Pods, keine Grenze gegen einen, der nicht kooperiert.
- **Regel 7 (Fail-closed Parser):** Ohne `secret=` besteht die
  Envelope-Prüfung aus einem Vergleich der Protokollversion und der Präsenz
  von `manifest_hash`/`principal`. Erst mit gesetztem `secret` signiert
  `mesh.py` jeden Envelope (HMAC-SHA256) und weist unsignierte ab — das ist
  die Stelle, an der `principal` etwas bedeutet (siehe ADR-9).
- **Regel 1 (Promotion nie ohne Gate):** Das Gate führt seit 2026-09-20 die
  Tests wirklich aus (`--run-tests`) bzw. prüft eine aufgezeichnete
  Testausführung gegen einen Quellcode-Hash. Die übrigen 42 Checks lesen
  weiterhin aufgezeichnete Messdateien; das Gate ist dort ein
  Regressions-Journal, kein Verifikationslauf.

## 6. Betriebs-Fakten (Server)

Root 591 GB (480 frei, LVM +700 GiB Reserve) · /srv/ai 492 GB ·
/mnt/neural-data 932 GB (ntfs3, fstab) · NVMe-Namen wechseln zwischen Boots
(UUIDs in fstab) · Mosquitto in np-node1 · Redis in np-node1 · Postgres 16 in
allen 3 LXD-Knoten · kdump aktiv · GPU-Läufe einzeln via nohup.
