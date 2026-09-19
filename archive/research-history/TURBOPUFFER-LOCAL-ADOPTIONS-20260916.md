# Sinnvolle lokale Übernahmen aus Turbopuffer

Die folgenden Bausteine passen direkt zu Neural Pods:

1. **Namespace-Metadaten**: Schema, Zeilenzahl, Branches, Read-only und
   Pinning sind bereits im LocalSearchBackend sichtbar.
2. **Copy-on-write Branches**: Pod-/Datenrevisionen bleiben unabhängig und
   können für Tests oder Re-Embedding isoliert werden.
3. **Prewarm und Cache-Temperatur**: `prewarm()` baut den Snapshot vor einem
   Burst auf und meldet `hot`; der PodCache übernimmt schnelle Resultat-Hits.
4. **Strong/Eventually-consistent Denkmodell**: Lifecycle-relevante Anfragen
   bleiben synchron und streng; eventual reads dürfen nur für nichtkritische
   Vorschauen ergänzt werden.
5. **Multi-query/RRF, Filter und Aggregationen**: der lokale Query-Vertrag
   deckt diese Suchformen ab; Typ/Tag/ACL-Filter bleiben vor dem Ranking.
6. **Recall-Gate**: ANN-Optimierungen dürfen erst nach Vergleich mit einer
   exakten Referenz aktiviert werden.

Nicht übernommen werden Anbieterabhängigkeiten, Cloud-APIs oder unbewiesene
100B-QPS-Versprechen. Die lokale Architektur bleibt selbst gehostet und
lineage-aware.

Referenzen: [Architecture](https://turbopuffer.com/docs/architecture),
[Concepts](https://turbopuffer.com/docs/concepts),
[Metadata](https://turbopuffer.com/docs/metadata),
[Recall](https://turbopuffer.com/docs/recall).
