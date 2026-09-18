# Semantik-Pilot: eine Identitaet, mehrere Formulierungen

Echtes Qwen2.5-0.5B-Instruct, zwei tatsaechlich trainierte Rank-8-LoRA-Adapter, CPU.
Die Lieferantendaten sind erfunden; es wurde kein SAP-System angesprochen.

Kanonische Identitaet: `knowledge:supplier:muller:x12:lead_time`.

| Pruefung | Ergebnis |
|---|---:|
| Generation 1, 24 Tage, untrainierte Formulierungen | 6/6 korrekt |
| Generation 2, 18 Tage, sechs Test- und acht Trainingsformulierungen | 14/14 korrekt |
| Nach Quellenwiderruf gesperrte Fragepfade | 14/14 |
| Alte Antwort nach Update am Commit gesperrt | True |
| Laufende Antwort nach Widerruf am Commit gesperrt | True |
| Falscher Principal gesperrt | True |
| Wiederherstellung als Generation 3 | 24 days |
| Basisgewichte unveraendert | True |

## Deine beiden Formulierungen

- g1: **Current delivery lead time for supplier Müller?** -> `24 days`
- g1: **X12 procurement delay from Müller GmbH?** -> `24 days`
- g2: **Current delivery lead time for supplier Müller?** -> `18 days`
- g2: **X12 procurement delay from Müller GmbH?** -> `18 days`

Alle diese Abfragen wurden ohne eingespeisten Antwortwert vom aktivierten LoRA-Pod beantwortet.

## Implementiert und geprueft

- Getrennte Retrieval-, Wissens- und Lebenszyklusdaten sowie vier explizite Schluessel.
- Kanonische Aliasaufloesung; keine Identitaetsuebernahme durch weiche Aliasse.
- Qdrant-Suche innerhalb der vorab erlaubten semantischen Menge; erneute Registry-Pruefung am Commit.
- Trainierter Kandidatenscorer mit Aehnlichkeit und strukturierten Metadaten.
- Harte Rollen, Rechte und Zeitgrenzen; weiche Annotationen tragen Quelle und Confidence.
- Relationstraversierung mit gemeinsamem Snapshot; Generationenwechsel eines Ziels macht alte Ableitungen ungueltig.

Die wiederhergestellte Ableitung traegt sowohl die Quelle der neuen Generation als auch die urspruengliche Trainingsquelle:

- `origin:a6c8b56f37261c7b8c8e449c492b040bf41c560269c0ecea670fd8593844f694`
- `origin:ff1ab609d1dd113e68ec534d31ea56de8c63f76929558a909e4b4cda9bc821dc`

## Grenzen

Der Resolver erkennt die implementierten Lieferzeitformulierungen und bekannten Firmenaliase; er ist kein universeller Sprachparser.
Die aehnliche andere Firma und der RULE-Eintrag sind textuelle Distraktoren. Nur der Ziel-Lieferant besitzt in diesem Lauf neu trainierte LoRA-Adapter.
Die Semantikfilter wurden zusaetzlich mit identischen Vektoren unterschiedlicher Firmen getestet, damit Aehnlichkeit allein den Test nicht bestehen kann.
Graphnavigation ist keine bewiesene neuronale Mehrschritt-Logik. J-Space ist nicht implementiert. Kein neuer RAG-Ueberlegenheitsnachweis und kein Gewichts-Unlearning.
Principals sind lokale Testparameter, keine authentifizierten Benutzer. Dateisystem und Registry-Aufrufer gelten als vertrauenswuerdig.

## Dateien

- [Rohbericht mit allen Antworten und Filtern](report.json)
- `registry.sqlite3`: Herkunft, Generationen, Ableitungen und Commit-Belege.
- `semantic-router.pt`: trainierter Router; `adapters/`: trainierte LoRA-Gewichte.
- [Architektur und Startbefehle](../../SEMANTIK.md)
