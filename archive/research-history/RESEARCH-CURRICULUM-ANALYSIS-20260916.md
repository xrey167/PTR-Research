# Research-Vorhaben: Fähigkeiten und Trainingscurriculum

## Was aus dem bisherigen Vorhaben benötigt wird

Das Projekt verbindet kanonische Knowledge-Objekte, stabile Pod-Identitäten,
versionierte Generationen, einen gelernten Dragonfly-Router, lokale hybride
Suche und eine Lifecycle-Barriere. Der Reader muss deshalb nicht nur Antworten
aus einem Pod lesen, sondern Entscheidungen strukturiert ausdrücken: welches
Tool, welcher Pod-Typ, welche Generation, welche Relationen und ob ein Artefakt
noch zulässig ist.

## Abgeleitete Fähigkeiten

| Fähigkeit | Pflichtsignale | Ziel im Datensatz |
|---|---|---|
| Pod-Lifecycle | knowledge/generation/status | aktive Generation auflösen |
| Tool-Auswahl | Tool-Vertrag v2 | ANN, BM25, Filter, Graph, Registry, Calculator, Workspace |
| Kontextuelles Chunking | Heading, Tabellenkopf, leading statement | kontextstabile Blöcke |
| Statement-Chaining | Antezedenz, leaf-only, Hops | abhängige Sätze vor dem Embedding verbinden |
| Namespace/Cache | Namespace, Konsistenz, Cache-Tier, Branch | burstige Reads und sichere Updates |
| Dragonfly-Routing | Entität, Typ, Tags, Generation, Pod-Vektor | Hard-Filter plus gelernter Ranker |
| Index-Performance | LSM, Block 256, Group Commit | batched/vectorized Postings |
| Provenienz/ACL | Origin-, Knowledge-, Generation-, Artifact-Key | Revocation an der Lifecycle-Barriere stoppen |
| Modell-Pod | Adapter, Basisidentität, Generation | Pod wechseln ohne Hauptmodell neu zu verlinken |
| Evaluation | Recall, Exactness, Guardrails, Latenz, QPS | reproduzierbare Baseline-Vergleiche |

## Datensatz

`prepare_research_curriculum.py` erzeugt 500 Beispiele je Train/Dev/Test-Split,
also 50 pro Fähigkeit. Die Fälle sind synthetisch und markieren das explizit;
keine synthetische Zahl darf als Unternehmensfakt ausgegeben werden. Targets
sind kanonische JSON-Entscheidungen. Vor echtem Fine-Tuning werden diese Fälle
mit anonymisierten CQP1/J-Space- und Projektfällen ergänzt und dedupliziert.

Die Struktur folgt den Suchmaschinenprinzipien aus Wilsons Search-Engine-Bericht:
semantische Normalisierung, kontextbewusstes Chunking, Satzketten, hybride
Suche, Streaming und getrennte Index-/Queue-Schichten. [Quelle](https://blog.wilsonl.in/search-engine/)

## Nächster Trainingsschritt

1. Curriculum gegen den bestehenden Tool-Adapter als Fortsetzung trainieren.
2. Tool-Choice und Lifecycle-Guardrails separat messen.
3. Multi-Hop- und Kontextfälle gegen eine unveränderte Qwen-Baseline testen.
4. Erst danach echte anonymisierte Projektbeispiele beimischen.
