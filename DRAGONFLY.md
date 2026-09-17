# Dragonfly: Router mit gelernter Pod-Repräsentation

## Aliaswissen beim Aktivieren (0.2.1)

`fit_pod` übernimmt jetzt alle **freigegebenen Aliase aus der aktuellen
Pod-Generation** automatisch ins Training. Es ersetzt Firmennamen in den
positiven Trainingsvorlagen durch die anderen freigegebenen Schreibweisen und
trainiert zusätzlich die bloßen Aliasnamen und ihre normalisierten Formen.
LLM-generierte, unbestätigte Soft-Aliase werden nicht übernommen.

Die Alias-Zuordnung steckt im trainierten Vektor `z_i` und Bias. Nach
`load_weights()` ist sie sofort nutzbar; beim Aktivieren ist kein Nachtraining
nötig. Die gespeicherte Aliasliste dokumentiert die Trainingsabdeckung. Die
Methode `recognize_alias` berechnet ihre Zuordnung aus Vektoren und verwendet
diese Liste nicht als Nachschlagetabelle.

```python
router.load_weights()
prediction = router.recognize_alias("Mueller", principal="buyer")
# prediction["subject"] == "supplier:muller"
```

Positive und negative Klassen werden beim Training gleich gewichtet. Mit
`negative_aliases=[...]` lassen sich ähnlich benannte andere Firmen explizit
abgrenzen. Eine Repräsentation wird nur aktiviert, wenn jeder freigegebene
Alias im Training einen Score von mindestens 0,8 erreicht und mindestens 90 Prozent
der negativen Trainingsbeispiele abgewiesen werden. Diese Prüfung ist
eine Trainingsabdeckung, kein Beweis für beliebige unbekannte Schreibweisen.

`recognize_alias` ist eine neuronale Diagnose mit Enthaltung unter Score 0,8
oder bei weniger als 0,15 Abstand zur nächsten Firma. Ihre Vorhersage allein
autorisiert keine Identität. Der normale Antwortpfad verwendet den trainierten
Vektor für die Fragebewertung und behält die harten Entitäts-/Rechteprüfungen.
Entfernte oder widerrufene Alias-Generationen dürfen auch nach dem Laden nicht
mehr ausgewertet werden. Neue Generationen benötigen erneut trainierte Vektoren.

Alter Schema-v1-Zustand bleibt für bisheriges Routing lesbar. Für die neue
Alias-Erkennung ist ein v2-Training nötig. Der fertige aktuelle Lauf ist
`runs/dragonfly-alias-002`:

```powershell
.\.venv\Scripts\python.exe ask_semantic.py runs/dragonfly-alias-002 "X12 procurement delay from Mueller GmbH?"
.\.venv\Scripts\python.exe train_dragonfly.py runs/semantic-003 --output runs/mein-alias-lauf
.\.venv\Scripts\python.exe verify_alias_activation.py runs/dragonfly-alias-002
```

Dieser Lauf erkennt nach erneutem Laden **7/7 Alias-Schreibweisen allein aus den
gelernten Vektoren** korrekt (sechs für Müller, eine für Müller Werke). Die
18 Routingprüfungen und sechs echten Qwen/LoRA-Antworttests bestehen weiterhin.
Rohdaten: `runs/dragonfly-alias-002/dragonfly-report.json`.
Die vollständige Suite besteht mit 82 Tests, einschließlich automatischer
Alias-Aufnahme, Erkennung ohne textbasierte Namensauflösung, Neustart,
Alias-Entfernung, Zugriffsprüfung und Ablehnung eines unzureichenden Trainings.

Implementiert ist die vereinbarte lokale Router-Idee `D(q, E_q, T_q, P_i, z_i)`.
Jeder indexierte Pod bekommt einen separat trainierten Vektor `z_i` und einen Bias.
Beim vorhandenen MiniLM-Encoder sind das **384 + 1 trainierbare Parameter pro Pod**.
Der Textencoder bleibt eingefroren; die LoRA-Antwortgewichte werden beim
Adress-Training nicht verändert.

## Berechnung und Training

Der Score ist eine Sigmoidfunktion über:

```text
12 × cosine(Fragevektor q, gelernter Pod-Vektor z_i)
 + 2 × cosine(q, indexierte Pod-Beschreibung)
 + Mittelwert der acht Metadatenmerkmale
 - gelernter Bias des Pods
```

Die Metadatenmerkmale umfassen Entität, Komponente, Prädikat, Rolle, aktuelle
Generation, Zeitgültigkeit, Zugriff und Tag-Übereinstimmung. Damit gehen `E_q`,
`T_q` und `P_i` in die Bewertung ein. Die Koeffizienten 12 und 2 sind fest;
trainiert werden `z_i` und Bias mit AdamW und binärer Kreuzentropie.

Positive und negative Trainingsfragen enthalten Adressen, keine Antwortwerte.
Während des Trainings gelten die Metadatenmerkmale absichtlich als übereinstimmend:
Das Modell muss Fragen anhand ihres Vektors unterscheiden, statt allein eine
Entitäts- oder Rechte-Markierung auszunutzen. Die Ähnlichkeit zur Pod-Beschreibung
bleibt ein echtes berechnetes Merkmal. Zur Laufzeit gelten die echten Metadaten.

Qdrant liefert Kandidaten nach den bestehenden harten Filtern. Dragonfly bewertet
diese Kandidaten; ein Score unter 0,5 führt zur Enthaltung. Scores sind keine
kalibrierten Wahrscheinlichkeiten. `learned=False` ist eine explizite
Kosinus-Baseline und wird im normalen Dragonfly-Abfragepfad nicht verwendet.

## Herkunft und Lebenszyklus

```text
Quelle → KnowledgeKey:Generation → Pod-Artefakt ─┐
                       └────────→ Indexvektor ─┼→ Repräsentation z_i → Antwort
Trainingsfragen mit eigener Herkunft ──────────┘
```

Die Repräsentation ist ein unveränderliches `router`-Artefakt in der Registry.
Der Datensatz enthält Schema, Encoder-Identität, Vektor, Bias und Trainingsmetriken.
Sein ArtifactKey schützt den serialisierten Inhalt und die Elternbeziehungen.
Beim Laden werden Hash, Schema, Dimension, Encoder-Identität und Generation geprüft.
Die Encoder-Identität wird vom Aufrufer aus dem gepinnten Modellmanifest geliefert;
sie ersetzt keine vollständige kryptographische Prüfung aller Encoderdateien.

Die Repräsentation ist an das konkrete Pod-Artefakt und den Indexvektor gebunden.
Der Antwort-Snapshot enthält auch ihren ArtifactKey. Widerruf einer Quelle, Generation,
Repräsentation oder Trainingsquelle sperrt ihre abhängigen Antworten, einschließlich
bereits gestarteter, noch nicht abgeschlossener Anfragen. Andere unabhängig
trainierte Pods bleiben gültig. Ein Generationenwechsel benötigt eine neu trainierte
Repräsentation; der Router verwendet den alten Vektor nicht stillschweigend weiter.

`fit_pod(..., training_parents=[...])` muss alle weiteren Quellen enthalten,
aus denen Trainingsfragen abgeleitet wurden. Ohne diese Eltern gelten die Fragen
als eigenständig eingebrachte, vertrauenswürdige Trainingsdaten. Der Router erntet
keine Negativbeispiele automatisch aus anderen Pods. Gemeinsame Trainingsquellen
erzeugen entsprechend gemeinsame Widerrufsabhängigkeiten.

## Lokal ausführen

Aus `C:\Users\ReyDa\neural-pods`:

```powershell
.\.venv\Scripts\python.exe train_dragonfly.py runs/semantic-003 --output runs/mein-dragonfly-lauf
.\.venv\Scripts\python.exe ask_semantic.py runs/dragonfly-002 "X12 procurement delay from Mueller GmbH?"
.\.venv\Scripts\python.exe -m pytest -q --junitxml=runs/dragonfly-tests.xml
```

Das Trainingsskript ist eine reproduzierbare Demo für das vorhandene
Müller/Müller-Werke-Szenario. Es kopiert Register, Qdrant und gespeicherte Adapter
in ein neues Verzeichnis und trainiert dort die Adressen. Das Quellregister bleibt
unverändert. Die Bibliotheksklasse `DragonflyRouter` und `fit_pod` sind nicht auf
diese beiden Namen beschränkt; die geerbte Fragenauflösung ist weiterhin der
begrenzte Lieferzeit-Parser.

## Gemessene Ergebnisse

`runs/dragonfly-002/dragonfly-report.json`:

| Prüfung | Ergebnis |
| --- | --- |
| Trainierte Repräsentationen | Zwei, jeweils 300 Optimierungsschritte |
| Auswahl ohne Entitätsfilter als Hilfestellung | 18/18 korrekt |
| Nur gelernter z-Vektor, ohne Metadatenbeitrag | 18/18 korrekt |
| Kosinus zur ursprünglichen Pod-Beschreibung | 16/18 korrekt |
| Qwen + gespeicherter LoRA nach Dragonfly-Auswahl | 6/6 korrekte Antworten |
| X99, Firmenalter, unberechtigter Principal | 3/3 gesperrt |
| Score und Repräsentations-ID nach erneutem Laden | Identisch |
| Widerruf einschließlich Repräsentation und Antwort | Bestanden; anderer Pod bleibt gültig |
| Qwen-Basisgewichte | Unverändert |

Die sechs bekannten Fragen pro Lieferant sind wiederholte Regressionen. Hinzu
kommen drei neue Fragevorlagen pro Lieferant, insgesamt 18 Routingprüfungen.
Keine dieser Testfragen ist wortgleich im Training. Der erste Lauf
`dragonfly-001` scheiterte an Alias-Schreibweisen und bleibt dokumentiert;
danach wurden die Trainingsvorlagen systematisch um Schreibweisen erweitert.
Die bekannten Testfragen sind deshalb kein unberührter finaler Benchmark.

Es werden zwei Kandidaten unterschieden, davon ein LoRA-Pod und ein textueller
Distraktor. Der direkte Vergleich ohne harte Metadatenfilter ist nur eine
Diagnose; echte Antworten verwenden weiterhin alle Filter. Diese kleinen
synthetischen Tests belegen Funktion, keine generelle Überlegenheit gegen RAG,
keine Skalierung auf viele Pods und keinen neuronalen J-Space-Operator.
