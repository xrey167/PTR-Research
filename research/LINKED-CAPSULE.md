# Gemeinsame Grenze für Modellverweis und Zustandskapsel

`linked_capsule.py` verbindet den vorhandenen semantischen Router mit
`NeuralSymlinks` und `PrefixCapsules`. Der Ablauf ist:

1. Semantisch routen und den ersten Abhängigkeits-Snapshot erfassen.
2. Den zum KnowledgeKey gehörenden Verweisadapter bestimmen; der aufrufende
   Modellpfad liefert dessen tatsächliche Verweisausgabe.
3. Diese Ausgabe strikt auf Identität, Cluster und aktuelle Generation prüfen.
4. Erneut routen und Übereinstimmung mit dem aufgelösten Ziel prüfen.
5. Die Kapsel inklusive Datei- und Modellhashprüfung laden.
6. Alle Routing-, Verweis-, Vektor- und Zustandsabhängigkeiten gemeinsam
   validieren; den gemeinsamen Snapshot beim Antwort-Commit erneut prüfen.

Auch der **erste** Routing-Snapshot bleibt Bestandteil der Abhängigkeiten.
Wird seine Generation während der Verweisinferenz ersetzt, wird die Anfrage
abgewiesen. Eine neue Anfrage kann danach sauber die neue Generation verwenden.
Ein falscher Modellverweis wird nicht durch den erwarteten ersetzt.

## Geschwisterartefakte und getrennte Modellstufen

Die aktuelle Integration registriert pro Generation und Reader-Modellhash eine
zusätzliche Kapsel in `capsule_variants`. Der ursprüngliche indexierte LoRA-Pod
und sein Vektor bleiben erhalten: Dragonfly v2 wurde an genau diesen Artefakten
trainiert. Die Kapsel muss dieselbe Generation als direkten Elternknoten haben.
Das erlaubt einen zusätzlichen Reader ohne heimliches Umbinden trainierter
Repräsentationen. Es löst noch nicht den Generationenwechsel ohne Routertraining.

Auf dem lokalen 16-GiB-Rechner laufen die echten Modelle nacheinander:

1. `run_linked_model.py links`: trainierter Dragonfly, echte Qwen-0.5B-Link-LoRA,
   Prüfung der tatsächlichen Ausgabe und Speicherung als Herkunftsartefakt.
2. `run_linked_model.py capsules`: erneute gelernte Auswahl, Prüfung des
   gespeicherten Nachweises einschließlich exakter Frage, Ausgabe und Adapter,
   dann echte Qwen-3B-Int8-Inferenz mit der Kapsel derselben Generation.

Der gemeinsame Antwortnachweis enthält auch das Zwischenartefakt. Dessen
Widerruf muss die daraus entstandene Antwort sperren. Dieser lokale Nachweis
ist keine kryptografische Attestierung einer entfernten Modellausführung.
Die Auswahl des Linkadapters erfolgt bereits über Dragonfly; sechs Aliasfragen
zu einem Fakt belegen keine freie Auswahl zwischen vielen Fakten im Basismodell.

## Nachweise

Die ursprünglichen vier Integrationstests bestehen: gemeinsame Receipt-Abhängigkeiten,
Update mit stabilem Link und gesperrtem altem Commit, Update während der
Verweisinferenz sowie falscher Modellverweis. Die gesamte Testsuite besteht
mit damals **104 Tests in 26,53 Sekunden** (`../runs/linked-capsule-tests.xml`).

Vier weitere Fälle prüfen Geschwisterkapseln, selektiven Nachweiswiderruf,
falsche Fragen/fehlende Adapterherkunft und generationsfremde Kapseln.
Aktueller Gesamtlauf: **111 Tests in 42,15 Sekunden**, Exit 0.

Diese Vertragstests verwenden einen kleinen echten Decoder für die Kapseln, aber
**Fixture-Verweisausgaben und einen Fixture-Encoder**. Sie belegen den
Integrationsvertrag. Der zusätzliche echte Modellversuch ist nun ausgeführt:

### `runs/linked-model-001`: echter integrierter Lauf

- Stufe 1: 6/6 tatsächliche Qwen-0.5B-LoRA-Ausgaben `LINK 1 CLUSTER 1`,
  mit trainiertem Dragonfly ausgewählt. Basisgewichte unverändert.
- Stufe 2: 6/6 tatsächliche Qwen-3B-Int8-Antworten `18` mit EOS, aus einer
  Geschwisterkapsel derselben aktuellen Generation. Basisgewichte unverändert.
- Frischer Textpräfix und gespeicherte Kapsel: im Runner sechs identische
  Tokenfolgen und vollständige Ersttoken-Logits; der separate Audit prüft alle
  sechs gespeicherten Logitpaare, tatsächliche Token-Decodierung und EOS.
- Vier Kontrollen auf Registerkopien: Nachweiswiderruf sperrt Commit,
  andere Frage und Kapsel bleiben gültig; Quellenwiderruf und Update sperren
  alte Commits. Kein neues Modelltraining für die zusätzliche Kapsel.
- Separater Audit bestanden: 64 Knoteninhalte samt Elternlisten nachgehasht,
  sechs vollständige Antwortabhängigkeiten und LoRA-/Kapseldateien geprüft.
- 111 Tests bestanden. Reader-Stufe 223,25 s inklusive Laden, Umwandlung,
  Gewichthashes, Referenzinferenz und Kontrollen. Median Anfragepfad der
  Reader-Stufe 1,453 s; LoRA-Verweisinferenz getrennt im Mittel 1,667 s.
  Diese gestufte Einzelmessung ist keine Produktions- oder RAG-Speedup-Messung.

Nachweise: `../runs/linked-model-001/{links,capsules,audit}.json`,
`reader-logits.safetensors`, `registry.sqlite3` und `source_snapshot/`.
Ausführbar mit `run_linked_model.py links --run <neuer-Ordner>`, dann
`run_linked_model.py capsules --run <derselbe-Ordner>` und
`audit_linked_model.py <derselbe-Ordner>`.

Der Audit prüft gespeicherte Evidenz, keine unabhängige erneute Modellausführung.
Die sechs bestehenden Aliasfragen sind ein Regressionstest zu einem Fakt.
Sie ersetzen nicht das weiterhin verfehlte 24-Aufgaben-Operationsgate (22/24)
oder eine neue, versiegelte Validierung.

## Weiter offen

- Größerer Kandidatenraum, zusätzliche Wissensarten und neue Validierungsfragen
  durch denselben echten Modellpfad.
- Eindeutige Frage-/Präfix-Schnittstelle und Modellidentität für jeden Reader.
- Wiederverwendbare Routingrepräsentationen ohne neues Training je Faktenwert:
  der bisherige Dragonfly bindet seine Repräsentation an die Faktengeneration.
- Originale CQP1-/BCC1-Zustandsoperatoren sowie breit angelegte Qualitäts- und
  Lifecycle-Vergleiche. Der aktuelle vollständige Präfix ist eine Kontrolle.
