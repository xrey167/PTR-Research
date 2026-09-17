# Rust-Bausteine fÃ¼r Pod-Kommunikation

## Ergebnis der PrÃ¼fung

- **grpc-rs** ist ein leistungsfÃ¤higer gRPC-Core-Wrapper, bringt aber C-Core,
  Build-/TLS-KomplexitÃ¤t und eine zusÃ¤tzliche Protokollschicht mit. FÃ¼r den
  bestehenden mTLS-Transport wird er deshalb nicht ungeprÃ¼ft als Binding
  eingebaut. Ein spÃ¤terer Peer-Transport kann gRPC hinter derselben typed
  message contract nutzen; die lokale Raft-Bindung bleibt ohne RPC.
- **fail-rs** ist fÃ¼r Rust-Failpoints passend. Bis der Peer-Transport Rust
  enthÃ¤lt, ist ein Ã¤quivalenter Python-Failpoint-Layer eingebaut:
  `FailureInjector` zÃ¤hlt deterministisch, injiziert Fehler oder VerzÃ¶gerung
  und ist standardmÃ¤ÃŸig deaktiviert.
- **crossbeam** bestÃ¤tigt das Queue-Design: bounded channels, select und
  epoch-basierte Speicherverwaltung passen zu Ready-/Message-Backpressure.
  Der aktuelle Python-Batcher hat bereits bounded capacity, Deadline-Drops,
  Cancellation und Queue-Metriken. Ein Rust-Queue-Binding wird erst
  hinzugefÃ¼gt, wenn der Peer-Transport tatsÃ¤chlich Rust-seitig lÃ¤uft.

## Warum nicht blind Ã¼bernehmen?

`grpc-rs` dokumentiert selbst C-Core- und Toolchain-Voraussetzungen. FÃ¼r unser
In-Process-Binding wÃ¼rde das den gemessenen lokalen Pfad verlÃ¤ngern. Der
messbare Vorteil entsteht erst bei mehreren Remote-Pods; dort vergleichen wir
grpc-rs/tonic gegen den vorhandenen mTLS-Socket anhand von p50/p95, Fehlern,
Backpressure und Wiederholungen.

Quellen: [grpc-rs](https://github.com/tikv/grpc-rs),
[fail-rs](https://github.com/tikv/fail-rs),
[crossbeam](https://github.com/crossbeam-rs/crossbeam).

Der Quorum-Replikator unterstÃ¼tzt jetzt begrenzte exponentielle Retries und
Failpoints pro Replica. Auf `xrserver` lagen 1.000 lokale Quorum-Writes bei
**160.689 ops/s** ohne Fehler und **936 ops/s** bei einer dauerhaft
ausgefallenen Replica mit zwei Retries; beide LÃ¤ufe blieben korrekt bei
1.000/1.000 erfolgreichen Writes, weil zwei von drei Replikas verfÃ¼gbar waren.

Der wiederverwendbare `PersistentRaftClient` framed protobuf payloads mit Magic, Sequenznummer, GrÃ¶ÃŸenlimit und Manifest-SHA-256. Ein persistenter Socket-Lauf auf `xrserver` lieferte **10.000/10.000 gÃ¼ltige Frames** bei **120.178 Frames/s**. Falsche Manifest-Hashes und Ã¼bergroÃŸe Frames werden vor der Ãœbergabe an Raft abgewiesen.

Der Frame-Server ist jetzt als `PersistentRaftServer` wiederverwendbar. Der vollstÃ¤ndige Clientâ†’Server-Test Ã¼ber diesen Baustein bestÃ¤tigte **10.000/10.000 Frames** bei **113.450 Frames/s**; der vorherige Benchmarkserver ist damit aus dem Architekturpfad entfernt.

Der Server akzeptiert optional nur konfigurierte TLS-Subjects zusÃ¤tzlich zur CA-PrÃ¼fung. Ein falscher, aber von derselben CA signierter Subject wird nach dem Handshake geschlossen und erreicht den Raft-Callback nicht. Manifest-Fehler werden ebenfalls sauber verworfen. Die Regression-Suite umfasst jetzt **268 Tests** (1 Skip, 15 Warnungen).

Security-Gate: Zwei Zertifikate derselben CA wurden getestet. Der erlaubte Subject lieferte 1 Frame; nach dem Versuch eines unbekannten Subjects blieb der Callback bei **1 Frame**. Damit wurde der falsche Peer nach TLS-Handshake verworfen, auch wenn der Client den Socket-Schreibvorgang zunÃ¤chst noch als erfolgreich sehen konnte.

