# Storage-Architecture-Design: Die einheitliche Speichertechnik der Pods — 2026-09-20

Referenzen: turbopuffer (zero-cost), SurrealDB, Mooncake (kvcache-ai),
tair-kvcache, LanceDB, crossbeam, tikv/jemallocator, redis-tower-protocol.
Ziel: EINE Storage-Architektur für alle Pods — Vektoren, Dokumente,
KV-Caches, Traces, Registry — mit Multithreading und vLLM-Multistream.

## 1. Das Vier-Schichten-Modell (turbopuffer/zero-cost + Mooncake)

Der Kern-Gedanke von turbopuffers „zero-cost"-Modell und Mooncake: **Zustand
lebt in billiger, dauerhafter Schicht; Rechenleistung ist ephemeral.** Über-
tragen auf neural-pods:

| Tier | Technologie | Inhalt | Latenz |
|---|---|---|---|
| **L0 Pod-lokal** | Prozess-Strukturen (crossbeam-Muster), jemalloc | Live-KV-Caches, offene Sessions, Mesh-Queues | µs |
| **L1 Shared-hot** | Redis in np-node1 (vorhanden) | Mesh-Cache, Presence, Session-Affinity | **p50 0,384 ms / p99 0,757 ms** (gemessen, `redis-cache-20260919.json`) |
| **L2 Warm-columnar** | **LanceDB** (embedded, Lance-Format) je Knoten | Vektoren, Pod-Dokumente, Versionen, Traces | ~1 ms |
| **L3 Cold-object** | SnapshotStore → `/mnt/neural-data` (932 GB) + optional S3 | Adapter, Runs, gefrorene Splits, alte Revisionen | 10 ms (NVMe) |

**Zero-cost-Prinzip:** Kein Pod hält Dauerzustand — L0 ist Wegwerf-State
(ephemeral compute), alles Rekonstruierbare liegt in L2/L3. Ein Pod-Neustart
(= Freezing, was wir dreimal erlebt haben) verliert nichts.

**KV-Cache-Ebene (Mooncake/tair-kvcache):** vLLM hat Prefix-Caching bereits
aktiv (`enable_prefix_caching=True` in unseren Logs). Ergänzt wird eine
**Session-Affinity-Schicht**: Redis mappt session_id → vLLM-Replica, damit
KV-Caches auf derselben Instanz wiederverwendet werden; bei Replica-Ausfall
übernimmt die andere Instanz (Prefix muss neu aufgebaut werden — die
Affinity-Metrik zeigt, wie oft das passiert). Mooncakes weitreichenderer
Schritt (KV-Cache-Transfer zwischen Instanzen via RDMA/Object-Store) bleibt
Phase 2 — vLLM-exponiert KV-Caches dafür noch nicht stabil.

## 2. Multi-Model-Storage (SurrealDB-Gedanke → LanceDB-Entscheidung)

SurrealDB vereint Dokument + Graph + Vektor in einer Engine. Für neural-pods
ist der Graph bereits die Registry (SQLite, Provenance-DAG) — den wollen wir
nicht ersetzen. Die Lücke ist **Vektor + Dokument + Versionen in einer
Engine**: LanceDB (embedded, kein Server, columnar, Versionierung eingebaut)
passt exakt:

- `LocalSearchBackend` (SQLite + eigene HNSW + Qdrant im Prototyp) → Vektor-
  und Dokumentsuche wandert in **LanceDB** (ein Format, versionierte
  Tabellen, Zero-Copy-Zugriff, MVCC)
- Traces (500+ Zeilen JSONL heute) → Lance-Tabelle (spaltenweise abfragbar)
- Eval-Reports/Messungen → Lance-Tabellen statt lose JSONs (Abfrage per SQL)
- Die Registry bleibt SQLite-Kontroll Ebene (Provenance, Transaktionen) —
  bewusst NICHT SurrealDB (zweite DB-Engine wäre Redundanz)

## 3. Parallelität (crossbeam + Multistream + Multithread)

- **crossbeam-Muster** sind im AdaptiveBatcher/TaskGraph bereits nachgebaut
  (bounded queues, deadline drops). Nächster Schritt: **Work-Stealing-Deque**
  im TaskGraph (crossbeam `steal()`-Semantik): leere Worker stehlen Jobs von
  vollen Queues anderer Mesh-Knoten.
- **jemallocator (tikv)** in `bindings/raft_binding` und künftigen Rust-
  Executoren: `jemallocator`-Crate statt System-Allocator — messbar bei
  vielen kleinen Allokationen (Raft-Ready-Frames, Mesh-Batching). Gate:
  Raft-Benchmark vor/nach (prop/s + RSS).
- **redis-tower-protocol**: optionaler Zukunftspfad — ein Rust-Redis-Modul,
  das unser Mesh-Envelope-Format NATIV in Redis spricht (statt JSON-over-
  Redis-Python). Erst wenn L1 zum Flaschenhals wird (aktuell p50 0,384 ms — weit
  davon entfernt).
- **vLLM-Multistream/Multithread**: vLLM V1 überlappt Scheduling; unsere
  zwei Replicas arbeiten bereits parallel. Ergänzend: pro Mesh-Knoten ein
  vLLM-Stream-Pool (Knoten-affine Streams), gemessen über den getracerten
  Pipeline-Lauf (Stage „answer" ist heute 0 ms — frozen Evidenz; mit echtem
  vLLM-Call wird das die dominante Stage und Multistream-Planung zählt).

## 4. Einheitliche Zugriffsschicht (`neural_pods/storage.py`)

Eine Fassade, die alle Tiere hinter einem Vertrag bündelt (Pods sprechen
nur diese API, nie Backends direkt):

```python
class PodStorage:
    def put(self, key, value, *, tier="auto") -> None       # L1→L2→L3 wandern
    def get(self, key) -> Any | None                        # liest rückwärts durch die Tiers
    def search_vectors(self, namespace, query, top_k)       # LanceDB
    def kv_session(self, session_id) -> SessionAffinity     # Mooncake-Muster
    def snapshot(self, refs) -> SnapshotRef                 # L3, rollback-fähig
    def traces(self, query) -> LanceTable                   # getracte Läufe
```

Tier-Wanderung automatisch: L1 (TTL) → L2 (Versionen) → L3 (Snapshot), mit
Provenance-Eintrag je Wanderung (Registry-Event).

## 5. Umsetzungsphasen

| Phase | Inhalt | Gate-Check |
|---|---|---|
| S1 | `neural_pods/storage.py` Fassade + LanceDB als L2 (Vektoren+Dokumente+Traces migriert), Turbopuffer-Tiering (L1-TTL → L2) | `storage_l2_lance` |
| S2 | KVCache-Session-Affinity (Redis-Mapping + Replica-Failover-Metrik) | `kvcache_affinity` (geplant) |
| S3 | jemallocator im Raft-Binding + before/after-Benchmark | `raft_jemalloc` (geplant) |
| S4 | vLLM-Multistream-Benchmark (echte Inferenz-Stages im getracerten Lauf, Stream-Pool je Knoten) | `vllm_multistream` (geplant) |
| S5 | Work-Stealing im TaskGraph über Mesh-Knoten | `work_stealing` (geplant) |

## 6. Risiken & Offene Fragen

- LanceDB ist embedded — Multi-Prozess-Zugriff (Host + Container) über
  denselben Pfad braucht Datei-Locking-Disziplin (oder je Knoten eine
  Instanz + Mesh-Sync)
- KVCache-Transfer zwischen Replicas (Mooncake-Kern) ist ohne vLLM-Support
  nicht realistisch — Affinity zuerst, Transfer beobachten
- jemallocator-Gewinn muss gemessen werden (Raft-Profile sind klein — der
  Effekt könnte unter dem Messrauschen liegen)


## Berichtigung (2026-09-20, Pod-Audit)

**Die Zahl „0,3 ms (gemessen)" für L1 stand hier falsch.** Die einzige
Messung für den Redis-Tier ist `research/runs/redis-cache-20260919.json`
mit **p50 0,384 ms / p99 0,757 ms**. 0,30 ms ist die Mesh-Presence-RTT aus
`mesh-presence-20260920.json` — ein anderer Vorgang über eine andere
Schicht. Beide Tabellenstellen sind korrigiert.

**Die Vier-Tier-Fassade implementiert zwei Tiers.** Der Docstring von
`neural_pods/storage.py` nennt L0 („pod-local") und L3 („cold snapshots →
SnapshotStore"). Beide Begriffe kommen in der Datei ausschließlich im
Docstring vor: `storage.py` importiert weder `snapshot_store` noch irgendein
L0-Konstrukt. `PodStorage` deckt L1 (Redis) und L2 (LanceDB) ab.

**Drei unabhängige L1-Implementierungen.** `pod_cache.PodCache` (LRU +
Redis, revisions- und branch-bewusst), `mesh_cache.MeshCache` (Redis,
principal-gescoped, Invalidierung per Pub/Sub) und `storage.PodStorage`
mit eigenem rohem Redis-Client. Keine benutzt eine andere. Das ist die
Folge davon, dass Fassaden gebaut wurden, um einen Gate-Check zu bedienen,
ohne die Bestandsnutzer abzulösen — die Migration war nie Teil der Phase.
Zu entscheiden ist, welche der drei bleibt; dieses Dokument beschreibt
`PodStorage` als „die einzige Pod-API", und in `neural_pods/` gibt es
derzeit **null** Aufrufer davon.
