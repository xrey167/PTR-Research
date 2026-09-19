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
| N4 | Mesh-Cache über Knoten | `mesh_cache` |
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