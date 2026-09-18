# Ergebnis: echtes Qwen-Modell mit versionierten LoRA-Pods

Lauf: `pilot-001`. Geraet: CPU. Laufzeit: 8.8 Minuten.

## Einordnung mit validiertem RAG-Prompt

Drei Promptvarianten wurden ausschliesslich auf den drei reservierten Validierungsfragen verglichen. Gewaehlt: `extraction`; bei gleicher Zahlenwertgenauigkeit entscheidet die kleinere Eingabelaenge.

- RAG mit gezielter Werteextraktion: **9/9 richtige Zahlenwerte** auf einem dritten, neuen Fragensatz; mittlere Latenz 1.286 s.
- Direkter strukturierter Lookup ohne LLM: **9/9 richtige Werte**; mittlere Latenz 0.035 s.

Damit ist keine Ueberlegenheit der Pods belegt. Fuer diese strukturierten Einzelwerte funktioniert ein einfacher Lookup bereits sehr gut. Der Extraktionsprompt nutzt die Frage zur Fundstellensuche und laesst das LLM anschliessend nur den Wert aus dem passenden Beleg extrahieren.

Die neun Vergleichsfragen unterscheiden sich von den vorigen zwoelf Pod-Fragen; daraus folgt kein statistisch gepaarter Leistungsvergleich. Rohdaten: [rag_comparison.json](rag_comparison.json).

## Zusaetzliche Pruefung mit explizit beleggebundenem RAG-Prompt

Frischer Prozess, gespeicherte Adapter, nochmals zwoelf neue Fragen. Keine weiteren Trainingsupdates.

| Verfahren | Exaktes Format | Richtiger Zahlenwert | Mittlere Latenz |
|---|---:|---:|---:|
| base | 0/12 | 0/12 | 1.179 s |
| grounded_rag | 0/12 | 0/12 | 1.414 s |
| always_loaded | 12/12 | 12/12 | 1.601 s |
| routed_pod | 12/12 | 12/12 | 1.622 s |

Der RAG-Prompt wurde nach dem ersten Pilotbefund geaendert, blieb aber erfolglos; die obigen neuen Fragen wurden nicht zum Training verwendet. Der erste schwache RAG-Lauf ist unten vollstaendig erhalten. Auch diese kleine Nachpruefung ist kein Produktionsvergleich.

Rohdaten: [verification.json](verification.json).

## Unbenutzte Testfragen

Drei erfundene Fakten, vier Testformulierungen pro Fakt. Strenges Exact-Match.

| Verfahren | Korrekt | Mittlere Latenz | Eingabetokens |
|---|---:|---:|---:|
| base | 0/12 | 1.147 s | 54.4 |
| qdrant_rag | 0/12 | 1.476 s | 77.1 |
| always_loaded | 12/12 | 1.601 s | 54.4 |
| routed_pod | 12/12 | 1.305 s | 54.4 |

## Lernen und Lebenszyklus

- Nach Update 24 -> 18 Tage: 12/12 Antworten korrekt (Trainings- und Testformulierungen getrennt in JSON nachvollziehbar).
- Nach Widerruf: 12/12 gepruefte Fragepfade blockiert.
- Basisgewichte unveraendert: True.
- Adapterdateien insgesamt: 10.38 MiB.
- Wiederhergestellte Generation: 3; Antwort: `24 days`.
- Ungeschuetzter alter Adapter nach Update: `24 days`.
- Mehr-Pod-Komposition: `43 days`; erwartet `53 days`.

- `stale_snapshot_rejected_after_update`: True.
- `old_cache_rejected_after_update`: True.
- `mixed_generations_rejected`: True.
- `inflight_commit_rejected_after_revoke`: True.
- `unrelated_pod_survives`: True.
- `old_identity_still_rejected`: True.

## Unerwuenschter Einfluss aktiver Adapter

| Adapter | Allgemeine Kontrollfragen korrekt |
|---|---:|
| Basis | 2/3 |
| all_facts | 0/3 |
| pod0_g1 | 0/3 |

## Grenzen

Kleiner Pilot mit echten vortrainierten Gewichten und echten Gradientenupdates. Kein Nachweis einer Ueberlegenheit gegen Produktions-RAG oder fuer 100–1000 Pods.

Die Sperrtests belegen kontrollierten Zugriff und Commit-Konsistenz. Sie belegen kein neuronales Vergessen: Ein direkt aufgerufener alter Adapter kann sein Wissen weiterhin ausgeben.

Die Latenzen gelten fuer diesen sequenziellen CPU-Lauf und warme Adapter. Keine Konfidenzintervalle, kein GPU-Test und kein Parallelitaetsbenchmark. Mehr-Pod-Verarbeitung verwendet zwei neural erzeugte Antworten als Text fuer die Komposition; keine Adapterfusion.

Einzel-Pods wurden jeweils 32 Schritte trainiert, der gemeinsame Adapter 64 Schritte. Gleiche Anzahl von Optimierungsschritten pro Fakt ist damit nicht gegeben; Trainingskosten stehen vollstaendig in report.json.

## Belege

- [Rohbericht](report.json)
- [Trainings- und Testdaten](dataset.json)
- `registry.sqlite3`: unveraenderliche Herkunftsknoten, Elternkanten und Ereignisse.
- `adapters/`: trainierte und wieder geladene Safetensors-Dateien.

Die unveraenderte Basismodell-Pruefsumme steht vor und nach dem Training im Rohbericht.
