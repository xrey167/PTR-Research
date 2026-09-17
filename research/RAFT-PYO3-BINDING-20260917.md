# raft-rs als In-Process-Binding

## Entscheidung

Für den lokalen Control-Plane-Pfad ist ein PyO3-Binding sinnvoller als ein
separater Adapterprozess: der Raft-Zustandsautomat läuft ohne Socket- oder
JSON-Hop im selben Prozess. Der Python-Layer bleibt für Pod-Metadaten,
Placement, ACL, WAL/Snapshots und Netzwerk verantwortlich.

Gebaut wird `bindings/raft_binding` (`neural_pods_raft`) gegen
[`raft-rs`](https://github.com/tikv/raft-rs). Der aktuelle Smoke-Binding
validiert Konstruktion, Single-Node-Kampagne, Status und Ready-Verarbeitung.

## Sicherheits- und Dauerhaftigkeitsgrenze

`MemStorage` ist nur ein In-Memory-Testbackend. In Produktion muss jede
`Ready`-Struktur in den WAL/Snapshot-Store geschrieben werden, bevor
`RawNode.advance()` aufgerufen wird. Netzwerk-Replikation, mTLS, Fencing und
Placement bleiben außerhalb des Bindings. So wird die Rust-Komponente nicht
mit Pod-Policy vermischt.

## Build und Smoke-Test auf xrserver

```bash
cd /home/xrey/neural-pods/bindings/raft_binding
VIRTUAL_ENV=/srv/ai/workspaces/llm-lora/.venv \
  /srv/ai/workspaces/llm-lora/.venv/bin/maturin develop --release
```

Der Build ist auf dem Server erfolgreich. `RaftNode(1).campaign()` liefert den
Leader-Status `(id=1, term=1, Leader)`.

`DurableRaftNode` schreibt die vom Binding gelieferte Ready-Struktur vor dem
`ack_ready()` in den bestehenden fsync-WAL. Ein 1.000-Operationen-Smoke-Test
auf `xrserver` erreichte dabei **25.556 WAL-Operationen/s** (ca. 0,039 ms
amortisiert pro Operation, lokales SSD-WAL).

Das Binding kann protobuf-serialisierte Peer-Nachrichten über
`pending_messages()`/`step_message()` austauschen. Der 3-Knoten-Loopback-Test
`research/benchmark_raft_cluster.py` wählte einen Leader und replizierte einen
Vorschlag auf alle drei Knoten; ein Lauf mit 100 Vorschlägen ergab **100/100
Einträge auf jedem Knoten**. Das ist ein echter Raft-Pfad im Prozess, noch kein
Test über drei physische Server.

Ein zusätzlicher TCP-Loopback-Test mit demselben Protobuf-Vertrag replizierte
20/20 Vorschläge auf alle drei Knoten (**19,5 Vorschläge/s**). Dieser Wert ist
bewusst ein konservativer Baseline-Test mit einer neuen TCP-Verbindung pro
Nachricht; der Produktionspfad muss persistente Peer-Verbindungen und Batching
verwenden. Er beweist dennoch, dass die Bindung nicht auf In-Process-Aufrufe
beschränkt ist.

Messung auf `xrserver`: Election in 12 Nachrichten-Schritten, danach 100
Vorschläge in **0,00113 s = 88.796 Vorschläge/s**, mit 100/100 angewendeten
Einträgen auf Leader und beiden Followern.

## Nächster Produktionsschritt

1. `Ready.entries`, `HardState` und Snapshots atomar im PostgreSQL/WAL-Store
   sichern.
2. Nachrichten über den bestehenden mTLS-Transport an die von PD/Placement
   zugewiesenen Peers weiterleiten. Das Binding stellt dafür die aus `Ready`
   serialisierten Raft-Nachrichten als protobuf bytes bereit.
3. Erst nach bestätigter Persistenz `advance` und danach die committed entries
   in die Pod-State-Machine anwenden.
4. [`fastrace`](https://github.com/fast/fastrace)/OpenTelemetry-Spans um
   `ready`, WAL, transport und apply legen;
   Payloads bleiben redigiert, nur IDs/Revisionen/Latenzen werden erfasst.

Damit sind Bindings der schnelle lokale Kern, während der Adapter weiterhin
für Dauerhaftigkeit und heterogene Remote-Pods gebraucht wird.

Nach Umstellung auf langlebige TCP-Verbindungen (`benchmark_raft_tcp_persistent.py`) stieg der Durchsatz auf **9.909 Vorschläge/s**, weiterhin **100/100 auf jedem Knoten**. Damit ist belegt, dass der vorherige TCP-Baselinewert vom Verbindungsaufbau und nicht vom Raft- oder Binding-Kern dominiert wurde.

Der neue mTLS-Peer-Test (`benchmark_raft_mtls.py`) verwendet drei eigene, gegenseitig verifizierte Zertifikate, persistente TLS-Verbindungen und denselben protobuf-Raft-Vertrag. Auf `xrserver` wurden **100/100 Einträge auf allen drei Knoten** angewendet, mit **2.327 Vorschlägen/s**. Der Vergleich zum persistenten Klartext-TCP (**9.909/s**) quantifiziert den TLS-Overhead, ohne die Replikationskorrektheit zu verlieren.
