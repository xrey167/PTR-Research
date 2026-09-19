# Multi-Hop und Parallel-Suche: Trainingscurriculum

Das Curriculum adressiert die schwächste gemessene Fähigkeit des Research-
Readers: Statement-Chaining (3/10). Es trennt abhängige Hops, die eine feste
Reihenfolge brauchen, von unabhängigen Suchästen, die parallel laufen dürfen.

## Kategorien

1. `multihop_reasoning`: 2–5 Relationen bis zum Endwert verfolgen.
2. `multihop_tool_plan`: ANN, Graph und Filter iterativ kombinieren.
3. `parallel_fanout`: unabhängige Hinweise gleichzeitig suchen und joinen.
4. `hybrid_parallel`: ANN/BM25/Metadaten parallel, danach RRF und Deduplizierung.
5. `parallel_safety`: unabhängige Arbeit parallel, abhängige Hops sequentiell.
6. `hop_validation`: vollständige Provenienz und Generation je Hop prüfen.
7. `latency_reward`: Recall erhalten und Parallelität gegen Latenz optimieren.
8. `multihop_negative`: fehlende oder widerrufene Zwischenfakten ablehnen.

Jede Kategorie enthält 50 Beispiele je Train/Dev/Test-Split, insgesamt 400 je
Split. Die Targets sind strukturierte Pläne und Ablehnungsentscheidungen; sie
enthalten keine erfundenen Unternehmensfakten.

## Lernziel

Das Modell soll nicht einfach „mehr Tools“ aufrufen. Es soll zuerst den
Abhängigkeitsgraphen erkennen, unabhängige Äste fan-outen, an Barrieren warten,
Ergebnisse zusammenführen und bei fehlender oder widerrufener Evidenz stoppen.
