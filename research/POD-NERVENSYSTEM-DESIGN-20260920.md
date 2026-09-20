# Pod-Nervensystem-Design: Vernetzte Pods mit nativer Protokoll-Sprache — 2026-09-20

## Vision

Pods werden von statischen Dienstleistern zu **vernetzten Entitäten**:

1. Jeder Pod trägt ein **kleines eigenes Modell** (Qwen2.5-0.5B-Instruct,
   lokal vorhanden) als „Reflex-Gehirn".
2. Dieses Modell **spricht Protokolle nativ** — MQTT und TCP zuerst — als
   trainierten Dialekt (LoRA auf Protokoll-Transkripten), **ohne Tool-Use**:
   Die Modellausgabe IST der Protokoll-Frame; kein JSON-Schema, kein
   Funktionsaufruf-Indirection.
3. Die Pods sind untereinander **vernetzt** (Mesh über MQTT-Broker + direkte
   Sockets), entdecken sich dynamisch (Presence statt statischer Peer-Maps)
   und führen **parallele Aufgaben** als Task-Graph aus.
4. **Token-Zusammenspiel:** Die Token-Ausgabe eines Pods ist die Token-
   Eingabe des nächsten — der Kontext fließt durch den Mesh-Graphen.
5. **Cache** wird zur Mesh-Ressource: ACL-safe Ergebnisse werden pod-
   übergreifend geteilt, Invalidierung läuft als Mesh-Event.

## Ankerpunkte im Bestand (Code-Exploration 2026-09-20)

| Vorhanden | Datei | Nutzung im Nervensystem |
|---|---|---|
| PodHeader.capabilities + TRANSPORT/RUNTIME-Verträge (`link_contract`) | `pod_types.py` | Egress-ACL-Slot für native Kommunikation |
| PodTransport: Hop-Budget, HMAC, Deadline, Retries | `pod_protocol.py` | Sicherheitszaun um jeden Mesh-Hop |
| AdaptiveBatcher (bounded queue, Deadline-Drops) | `adaptive_batcher.py` | Backpressure im Task-Graph |
| PodFanout (ThreadPool 1–64) | `pod_protocol.py` | Parallelausführung unabhängiger Knoten |
| merge_branches (Konfidenz/Latenz) | `pod_streams.py` | Hypothesen-Merge im Task-Graph |
| PodCache L1/L2 (ACL-safe Keys, Redis) | `pod_cache.py` | Mesh-Cache-Basis |
| DuplexSession (Events, epochs, resume) | `pod_streams.py` | Sitzungs-Kanal über Mesh |
| ResourceGovernor (vram/ram-Leases) | `resource_runtime.py` | Pod-Aktivierung im Mesh |
| NPR1-Binary-Framing, mTLS | `raft_transport.py` | Vorbild für Mesh-Framing |

**Bekannte Lücken, die N1–N5 schließen:** keine Protokoll-Versionierung,
keine Peer-Discovery (statische Peer-Maps), kein Pub/Sub, kein Socket-Dial,
kein kleines Modell im Pod, keine Backpressure in DuplexSession.

## N1 — Mesh-Infrastruktur

- **Broker:** Mosquitto in np-node1 (LXD, 10.50.0.121:1883); Topic-
  Konvention `np/{pod_id}/{kanal}`; Presence `np/presence/{pod_id}`
  (retained, Heartbeat)
- **`neural_pods/mesh.py`:** `MeshEndpoint(host, pod_id, principal,
  manifest_hash)` — `publish(topic, payload)`, `subscribe(pattern,
  callback)` mit Principal-ACL; Payload-Envelope `{manifest_hash,
  protocol_version, principal, body}`; `MeshPresence` (online/offline/
  heartbeat) → **dynamische Pod-Discovery** ersetzt statische Peer-Maps
- **`protocol_version: int = 1`** in `PodRequest`/`PodResponse` — die
  lange offene Versionierungs-Lücke
- Gate `mesh_presence`: zwei Endpunkte auf verschiedenen LXD-Knoten
  entdecken sich, RTT < 10 ms

## N2 — Nativer Protokoll-Executor

**Prinzip „Protokoll als Dialekt":** Das 0.5B-Modell wird wie der Reader
trainiert (train_reader-Pipeline-Muster): synthetische, mechanisch
verifizierte Protokoll-Transkripte als Trainingszeilen. Die Modellausgabe
ist kein Tool-Call, sondern direkt der Frame:

```
PUB np/reader/answer {"value": 42, "unit": "days"}
DIAL 10.50.0.153:45300
SEND <base64 frame>
```

- **`neural_pods/native_comm.py`:** `FrameParser` (deterministisch,
  fail-closed — invalid Frames werden verworfen und gezählt),
  `EgressACL` (aus `link_contract`: erlaubte Hosts/Ports/Topics pro Pod),
  `NativeCommExecutor` (führt freigegebene Frames nativ aus: paho-mqtt
  bzw. sockets; Hop-Budget/Rate-Limit je Pod)
- **Training:** `research/train_native_comm.py` — Transkript-Synthese
  (MQTT-PUB/SUB, TCP-DIAL/SEND/RECV-Szenarien über die Mesh-Realität),
  Preflight → LoRA-Training (BF16) → held-out Frame-Validitätseval;
  Reload-Identität wie beim Reader
- **Gate `native_protocol`:** Frame-Validität ≥ 98 % held-out; Egress-ACL
  verweigert verbotene Ziele zu 100 %; Validitätsparser selbst
  deterministisch getestet

## N3 — Token-Zusammenspiel & Task-Graph

- **`neural_pods/taskgraph.py`:** `TaskGraph` — Knoten = Pod-Aufruf
  (capability + payload-Transform), Kanten = Token-Fluss (Ausgabe von A
  fließt als Kontext in B); `run()` führt unabhängige Knoten parallel
  (PodFanout/ThreadPool), Abhängigkeiten per Event; Backpressure über
  AdaptiveBatcher-Queues; Merge über `merge_branches`
- **Gate `taskgraph_parallel`**: 6-Knoten-DAG über Mesh-Endpunkte,
  korrekte Topologie-Einhaltung, Speedup > 1.5× gegenüber sequenziell

## N4 — Mesh-Cache

- Die vorhandene Redis-L2 (np-node1) wird zum Mesh-Cache: Pod A (Knoten 1)
  schreibt ACL-safe Einträge, Pod B (Knoten 2) liest sie; Invalidierung
  als Mesh-Event (`np/cache/invalidate`)
- **Gate `mesh_cache`**: Cross-Knoten Hit, Principal-Isolation belegt

## N5 — End-to-End-Demo

- Zwei Pods auf verschiedenen LXD-Knoten, jeder mit 0.5B-Gehirn +
  Protokoll-LoRA: Pod A beantwortet eine Reader-Frage und **spricht das
  Ergebnis nativ per MQTT**; Pod B empfängt, validiert und antwortet
  nativ; das Hauptmodell (Gen-7) konsumiert nur das fertige Ergebnis
- **Gate `mesh_e2e`**

## Sicherheitsregeln (Nervensystem-spezifisch)

1. **Egress-ACL aus `link_contract`** — kein Frame verlässt einen Pod
   ohne ACL-Prüfung; Verstoß = Refuse + Metrik + Event-Log
2. **Fail-closed Parser** — nur vollständig valide Frames werden ausgeführt
3. **Hop-Budget/Deadline** aus PodTransport gelten über Mesh-Hops weiter
4. **Principal-Isolation** im Cache und auf Topics (`np/{pod}/...` ist
   pod-lokal, Cross-Pod-Zugriff nur über verhandelte Kanäle)
5. **Rate-Limits** je Pod (Token-Budget für native Frames)

## Phasen & Gate-Checks

| Phase | Inhalt | Gate-Check |
|---|---|---|
| N1 | Mosquitto im LXD, MeshEndpoint, Presence, protocol_version | `mesh_presence` |
| N2 | Protokoll-Dialekt-LoRA (0.5B) + NativeCommExecutor + Egress-ACL | `native_protocol` |
| N3 | TaskGraph (Token-Fluss, Parallelität, Merge) | `taskgraph_parallel` |
| N4 | Mesh-Cache über Knoten | `mesh_cache` **rot seit Pod-Audit**, s. u. |
| N5 | E2E: zwei 0.5B-Pods sprechen nativ | `mesh_e2e` |

Akzeptanz: keine der bestehenden 33 Checks bricht; jede Phase liefert
Messdatei + Design-Doku-Abschlussvermerk.


## N1-Status (2026-09-20, ABGESCHLOSSEN)

- Mosquitto 2.0.18 in np-node1 (listener 0.0.0.0:1883, lab-only anonymous).
- `neural_pods/mesh.py`: MeshEndpoint (Presence retained + Heartbeat,
  Envelope mit manifest_hash/protocol_version/principal, Topic-ACL,
  Fail-closed Envelope-Validierung), `protocol_version: int = 1` in
  PodRequest/PodResponse.
- Gemessen (`research/runs/mesh-presence-20260920.json`): Presence-Discovery
  Host↔Container-Peer (np-node2) über den Broker; RTT **p50 0,30 ms /
  p99 0,58 ms bei 100/100 Round-Trips** über den remote Broker.
- Drei während der Umsetzung gefundene und behobene Fehler (alle
  dokumentiert, weil sie für jede MQTT-Nutzung relevant sind):
  1. wait_for_publish im paho-Callback-Thread blockiert den eigenen
     Network-Loop (Deadlock-artig: 400 ms RTT + Verluste) → nie im
     Callback blockieren.
  2. Wiederverwendete client_ids flappen gegen Zombie-Sessions → UUID-Suffix.
  3. Ein globaler Manifest-Hash-Vergleich im Envelope verwirft legitime
     Nachrichten fremder Pods → Hash-Prüfung gehört pro Kanal/PodTransport.
- Gate-Check `mesh_presence` grün → **Gate 34/34, 289 Tests**.

## N2-Status (2026-09-20, KERN ABGESCHLOSSEN)

- Basis-Modell: **Qwen2.5-Coder-1.5B-Instruct** (lokal, unsloth-Cache;
  statt 0.5B — instruct-tuned, thematisch passend, 4,1 GB CUDA-Peak).
- `neural_pods/native_comm.py`: FrameParser (fail-closed, max 16 Frames,
  b64/JSON/Topic-Validierung), EgressACL (Topics + Hosts/Ports aus dem
  Link-Contract, Violation-Zähler), NativeCommExecutor (MQTT via
  MeshEndpoint.publish_raw, TCP via echte Sockets; Rate-Limiter drosselt
  statt verwirft — DIAL/SEND/RECV/CLOSE laufen in Mikrosekunden).
- `research/train_native_comm.py`: 500 synthetische Transkripte (mechanisch
  verifiziert), SFT-LoRA r16, 3 Epochs, ~10 min GPU.
- **Gemessen (`runs/native-comm-eval-20260920-report.json`):**
  Frame-Validitätsrate **1.0** auf 80 held-out Transkripten (Schwelle 0.98),
  exact_rate 0.55 (valide Varianten zulässig), **ACL verweigert verbotene
  Ziele** (refused=true, violation gezählt). Gate-Check `native_protocol`
  grün → **Gate 35/35, 295 Tests.**
- Offen in N2: TCP-Frame-Ausführung über echte Mesh-Knoten (LXD), Integration
  in den Task-Graph (N3).

## N3-Status (2026-09-20, ABGESCHLOSSEN)

- `neural_pods/taskgraph.py`: TaskGraph — DAG mit Token-Fluss (Output eines
  Knotens wird als context in abhängige Knoten injiziert; Einzel-Vorgänger
  direkt, mehrere als {node_id: output}-Dict), parallele Ausführung
  unabhängiger Knoten (ThreadPool, max_parallel), Zyklus- und
  Unbekannten-Abhängigkeits-Prüfung, Fehler werden je Knoten aufgezeichnet
  statt den Graphen abzubrechen. 5 Tests.
- **Gemessen (`research/runs/taskgraph-20260920.json`):** 6-Knoten-DAG über
  Mesh-Endpunkte (4 ingests parallel, 2 abhängige plan/verify, je ~100 ms
  Remote-Verarbeitung): **Speedup 2,59×** (Wall 0,64 s vs. sequenziell
  1,67 s), alle Knoten korrekt. Gate-Check `taskgraph_parallel` grün →
  **Gate 36/36, 300 Tests.**
- Benchmark-Lehren: TaskResult-Serialisierung (asdict), seq-Vergabe unter
  Parallelität braucht Lock, Multi-Dep-Context ist ein {node_id: output}-Dict.

## N4-Status (2026-09-20, ABGESCHLOSSEN)

- `neural_pods/mesh_cache.py`: MeshCache — Redis-getragene Ergebnis-Teilung
  über Knoten; Keys tragen namespace+principal (ACL-Isolation per Hash);
  Invalidierung löscht gezielt.
- **Gemessen (`research/runs/mesh-cache-20260920.json`):** Pod A (Host)
  schreibt, Pod B (np-node2) liest über das Netz: cross_node_read ✓,
  same_principal_visible ✓, invalidation_works ✓. Der Check war damit
  grün → **Gate 37/37, 300 Tests**.
- **Berichtigung (2026-09-20, Pod-Audit).** `principal_isolated ✓` war wahr
  per Konstruktion: der Benchmark las unter einem Principal, unter dem nie
  etwas geschrieben wurde, während derselbe Lauf zwei Zeilen vorher den
  fremden Principal über die Knotengrenze auslas. Der Check liest jetzt die
  ehrlichen Felder, und Gate-Check `mesh_cache` ist **rot**, bis der
  Benchmark am Broker neu läuft. Die N4-Mechanik selbst — Cross-Knoten-Lesen,
  gezielte Invalidierung — ist davon unberührt.

## N5-Status (2026-09-20, ABGESCHLOSSEN — Nervensystem komplett)

- **E2E-Demo (`research/runs/mesh-e2e-20260920.json`):** Pod A (Host,
  20 frozen Gen-7-Antworten) sprach jede Antwort **nativ als MQTT-Frame**
  über den NativeCommExecutor (Dialekt-Form identisch zum N2-Training);
  Pod B (np-node2) empfing, validierte nativ und akkreditierte:
  **20/20 Acks, alle validiert, 0 ACL-Verweigerungen, 7,5 s Gesamtdauer.**
  App-Level-Retry deckt QoS-Verluste (1 verlorenes Ack in Runde 1 durch
  Retry geborgen).
- Gate-Check `mesh_e2e` grün → **Gate 38/38, 300 Tests.**
- **Das Nervensystem ist komplett:** Pods haben ein kleines Modell, das
  MQTT/TCP nativ spricht (100 % Frame-Validität), entdecken sich per
  Presence (RTT 0,3 ms), arbeiten parallel (2,59× Speedup), teilen
  Ergebnisse über Knoten (ACL-isoliert) — alles fail-closed und im Gate.

Offene Erweiterungen: TCP-Frames über echte LXD-Knoten in N2-Rest, P2
(XGBoost-Executor), P5 (latentes Adress-Training), N4-Invalidierung als
Mesh-Event anstelle direktem Redis-Delete.

## N2-Rest (2026-09-20, ABGESCHLOSSEN): TCP über echte Mesh-Knoten

- `research/benchmark_native_tcp.py` + `research/echo_server.py`: Echo-Server
  in np-node2 (setsid! — lxc-exec-Sessionende killt sonst die Prozessgruppe,
  kein nohup allein reicht); der Host-Pod führt DIAL/SEND/RECV/CLOSE-Frames
  nativ gegen 10.50.0.153:45779 aus.
- **Gemessen:** 30/30 Byte-Integrität über das Containernetz, ACL blockiert
  verbotene Ziele (evil.example.com) 100 %, RTT p50 60 ms (4-Frame-Sequenz ×
  20 ms Rate-Limit-Drossel + Verbindungssetup — die Drossel ist
  Sicherheitsparameter, nicht Netzlatenz). Gate-Check
  `native_tcp_cross_node` grün → **Gate 39/39, 300 Tests.**
- Lektion: zombie Ports von früheren Läufen (pkill Muster breit fassen oder
  Ports wechseln); lxc-exec-Hintergrundprozesse IMMER mit setsid + </dev/null.

## Breiter getracer Test + Performance-Optimierung (2026-09-20, ABGESCHLOSSEN)

- **Getracer Pipeline-Lauf** (`research/benchmark_traced_pipeline.py`,
  Trace-Log `research/runs/traced-pipeline-20260920.jsonl`, 500+ Hop-Records):
  alle 132 frozen Cases durch cache → mesh-lookup → Antwort (Gen-7-Evidenz)
  → native MQTT-Frame-Publikation; je Hop trace_id/case/latency/flags.
- **P2-XGBoostExecutor** (`neural_pods/pod_executor.py`): ExecutorFactory +
  GovernedExecutorPool (Budget-Verweigerung), deterministische Inferenz
  verifiziert. P5: trainiertes Dialekt-Modell emittiert Pod-Adressen —
  **Reflex-Hit-Rate 1.0 (31/31)** vs. 0.0 untrainiert
  (`research/runs/reflex-trained-20260920.json`). P3: PerceptionStream,
  2000/2000 Events lossless (~13.3k Events/s), Backpressure droppt sauber —
  **Evidenz zurückgezogen.** Die Messung stammt von einem Drain-Loop, der
  Events auch ohne Consumer als `consumed` zählte, und von einem `drain()`,
  das eine leere Queue für Zustellung hielt. `research/runs/perception-
  20260920.json` wurde deshalb entfernt statt weitergereicht; der Benchmark
  (`research/benchmark_perception.py`) schreibt seit dem Pod-Audit über
  `research.evidence.write()` und muss am Broker neu laufen. Kein Gate-Check
  liest diese Datei, die Zahl trug also nie eine Zusicherung.
- **Trace-Befund → Optimierung:** die sequenziellen Mesh-Lookups dominieren
  (20 ms Remote-Delay × 132). Optimierung: Batch-Async (alle Lookups sofort
  feuern, Replies parallel sammeln; Responder schläft NICHT im paho-Loop-
  Thread). **Gemessen: 2,845 s → 1,012 s = 2,81× schneller** bei identisch
  132/132 validen Frames. Gate-Check `traced_pipeline` grün. (Die
  damaligen Zählstände „Gate 40/40, 305 Tests" sind überholt; aktuelle
  Zahlen ausschließlich in `ARCHITECTURE-MASTER-20260920.md`. Die
  „132/132 validen Frames" heißen seit dem Messcode-Umbau
  `serialised_frames_valid` — der Benchmark parst seinen eigenen
  f-String, die Zahl belegt Serialisierer und Parser, nicht das Modell.)
- Nächste Optimierungshebel aus den Traces: Redis-Puts batchen (132 Round-
  Trips ~0.05 s), Antwort-Stage als echter vLLM-Call mit Stream, Responder
  mit Thread-Pool statt Thread-je-Call.