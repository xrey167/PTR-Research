# Pod-Cache

Der Pod-Cache übernimmt die relevanten Namespace-Eigenschaften eines
Turbopuffer-ähnlichen Lesepfads lokal:

- Schlüssel enthalten `namespace`, `branch`, `pod_type`, sortierte `tags`,
  Query und `top_k`.
- Ein Treffer kann daher nie zwischen Pod-Typen, Tags oder Branches geteilt
  werden.
- O(1)-Lookup über einen bounded LRU mit TTL; `invalidate()` löscht gezielt
  Namespace, Branch, Typ oder Tag.
- `stats()` liefert Hit-Rate und Belegung. Pinning/Cold-Storage bleibt beim
  bestehenden LocalSearchBackend; der Result-Cache sitzt davor.

Der lokale Hot-Path-Test erreichte 412.126 Cache-Reads/s bei 10.000 Reads,
10.000 Hits und einer Initial-Missrate von 0,01 %. Das ist ein Mikrobenchmark
im selben Prozess, keine verteilte Produktionsmessung.
