# Lokaler Anschlussversuch: gespeicherte Modellzustände

## Zweck und Herkunft

Der importierte Quantenchat-Verlauf fordert eine belastbare positive Kontrolle
für die Einspeisung von Modellzuständen. Dieser lokal neu geschriebene Versuch
prüft deshalb zuerst vollständige Präfix-Caches am vorhandenen
Qwen2.5-0.5B-Instruct. Er ist **keine Rekonstruktion des originalen
CQP1-Q0R-Codes**, verwendet nicht dessen 3B-Modell und übernimmt keine fremden
Ergebniszahlen. Forschungsabgleich: [RESEARCH-ABGLEICH.md](RESEARCH-ABGLEICH.md).

Ein solcher Cache speichert die Schlüssel- und Wertetensoren der
Aufmerksamkeitsschichten. Bei seiner Erstellung verarbeitet das Modell den
Fakttext. Bei späteren Anfragen erhält es den gespeicherten Zustand und die
neue Frage. Dies ist eine bekannte Präfix-Cache-Technik als Kontrollversuch,
kein beanspruchter neuer Wissensmechanismus.

## Implementiert

- `prefix_capsule.py`: Zustände als Safetensors speichern, SHA-256 prüfen,
  Modellidentität binden und als abgeleitete Cache-Artefakte registrieren.
- Jede Anfrage lädt einen eigenen Cache; Fortsetzungen dürfen andere Anfragen
  nicht verändern.
- Generationswechsel und Widerruf blockieren alte Zustände sowie ausstehende
  Antwort-Commits. Die unveränderte zweite Wissensidentität bleibt nutzbar.
- Neue Werte werden neu vorverarbeitet, ohne Gradienten und ohne LoRA-Training.
- `run_prefix_probe.py`: 2 synthetische Lieferanten, 4 Werte, 3 Operatoren,
  4 Kontrollpfade = 96 Antworten pro Lauf; zusätzlich eine Antwort nach Update.
- Vollständige erste Logit-Vektoren und generierte Tokenfolgen einschließlich
  EOS werden gespeichert. `audit_prefix_probe.py` zählt Ergebnisse separat nach
  und vergleicht die gespeicherten Vektoren.

Die Textreferenz berechnet denselben Präfix frisch und verwendet dieselbe
Aufteilung und Cache-Ausführung wie der gespeicherte Pfad. Ein ergänzender
Diagnoselauf vergleicht auch ungeteilte Eingabe und vollständige Tokenisierung.
Es wird nicht behauptet, dass unterschiedliche Rechenaufteilungen grundsätzlich
bitidentische Logits ergeben.

## Festgelegte Kriterien

Das strikte Gate verlangt korrekte vollständige Antworten für alle drei
Operatoren: Wert abrufen, zwei Tage addieren, mit 25 Tagen vergleichen. Ein
absichtlich anderer Zustand muss die zu seinem anderen Wert passende Antwort
liefern. Frischer und gespeicherter Zustand müssen vollständig gleiche erste
Logits und Fortsetzungen liefern. Das Update muss den neuen Wert ausgeben.

Der Nullpfad ohne Fakt ist eine zusätzliche Gegenprobe. Auch seine falschen
Antworten werden ausgewiesen. Er ist kein zulässiger Produktionspfad für eine
Anfrage, die eine gültige Wissensquelle benötigt.

## Ergebnisse und Grenzen

Die Testsuite besteht mit **94 Tests**. Vier neue Tests prüfen am kleinen echten
Decoder: identische Fortsetzungen und unabhängige Anfragen, ACL/Widerruf,
manipulierte Dateien und abweichende Modellidentität. Das ist zusätzlich zu den
unten gemessenen Antworten des vorhandenen vortrainierten Qwen-Modells.

Lauf 001 scheitert am semantischen Gate: Textreferenz und gespeicherter Zustand
erreichen jeweils **1/24**. Alle **24/24** Zustandsvergleiche sind bitidentisch.
Die falschen Zustände erreichen ebenfalls nur **1/24** gegen ihren eigenen
Sollwert. Die sechs Lifecycle-Prüfungen bestehen; der Wert nach Update wird
falsch beantwortet. Die Basisgewichte bleiben unverändert.

18 anschließende Diagnoseantworten vergleichen zwei Systemprompts und drei
Eingabepfade an drei Fragen. Pro Prompt und Frage liefern die drei Pfade jeweils
dieselbe Antwort. Damit ist ein Fehler durch Aufteilung oder Tokenisierung in
diesen Fällen nicht belegt. Der erste geänderte Prompt behebt die Fehler nicht.

Lauf 002 ist ein **explorativer Prompt-Folgeversuch am selben Datensatz**:
Ausgabeformat und Fragetyp werden direkt in der Frage angegeben. Er ist kein
unabhängiger Generalisierungstest. Lauf 001 bleibt vollständig erhalten; seine
Kriterien und Ergebnisse werden nicht nachträglich umgedeutet.

| Prüfung in Lauf 002 | Ergebnis |
| --- | --- |
| Frische Textreferenz | 12/24 |
| Gespeicherter Zustand | 12/24 |
| Davon Wertabfrage | 8/8 |
| Davon Addition von zwei Tagen | 4/8 |
| Davon Vergleich mit 25 Tagen | 0/8 |
| Falscher Zustand gegen seinen eigenen Sollwert | 12/24 |
| Antwortwechsel bei falschem Zustand | 16/24 |
| Nullpfad: korrektes UNKNOWN ohne Fakt | 0/24 |
| Bitgleiche erste Logits und Fortsetzungen frisch/gespeichert | 24/24 |
| Update 18 auf 24 ohne Training | Antwort 24, korrekt einschließlich EOS |
| Lifecycle-Prüfungen | 6/6 |
| Basisgewichte unverändert | Ja |
| Median gespeicherter Pfad, Laden/Hashprüfung und Inferenz | 0,509 s |
| Median frischer Textpräfix und Inferenz | 1,053 s |
| Rohe Zustandstensoren pro Fakt | 1.327.104 Bytes |
| Striktes wissenschaftliches Gate | Nicht bestanden |

Die Laufzeiten stammen aus festen, nicht randomisierten Einzelwiederholungen
je Frage auf der CPU; sie belegen keinen belastbaren allgemeinen
Geschwindigkeitsvorteil. Der Nullpfad halluziniert Antworten und darf nicht zur
Umgehung einer fehlenden Wissensquelle verwendet werden.

Gesamtumfang dieser Fortsetzung: **212 echte Modellfortsetzungen**
(zweimal 96 Kontrollantworten, zweimal eine Updateantwort, 18 Diagnoseantworten).
Die beiden getrennten Audits bestätigen jeweils die 96 gespeicherten
Kontrollantworten und 24 vollständigen Logit-/Tokenvergleiche. Sie führen die
Lifecycle-Prüfung oder Gewichtsprüfung nicht unabhängig erneut aus.

Belege:

- [Lauf 001](../runs/prefix-interface-001/results.json),
  [Audit 001](../runs/prefix-interface-001/audit.json)
- [Promptdiagnose](../runs/prefix-prompt-diagnostic-001/results.json)
- [Lauf 002](../runs/prefix-interface-002/results.json),
  [Audit 002](../runs/prefix-interface-002/audit.json)
- [Ausgeführter Quellstand für Lauf 002](../runs/prefix-interface-002/source_snapshot/manifest.json)
- [94 bestandene Tests](../runs/prefix-interface-tests.xml)

Nächste begründete Hürde: eine Textreferenz, die die Operationen zuverlässig
beherrscht, vorzugsweise am zum Originalforschungsstand passenden Modell.
Quotientenkompression oder ein trainierter Zustandscompiler lassen sich aus
diesem gescheiterten Kontrollgate noch nicht als tragfähig ableiten.

Die Technik speichert vollständige Zustände pro Fakt. Sie belegt weder
kompakte übertragbare Quotienten noch die BCC1-O-Bindung, Mehr-Hop-Komposition,
Neuheit oder Überlegenheit gegenüber starkem RAG. Der bewährte lokale
Dragonfly-/LoRA-Pfad bleibt ein eigener Versuchsstrang.

## Wiederholen

```powershell
.\.venv\Scripts\python.exe research/run_prefix_probe.py --output runs/prefix-neu
.\.venv\Scripts\python.exe research/run_prefix_probe.py --typed-format --output runs/prefix-typed-neu
.\.venv\Scripts\python.exe research/audit_prefix_probe.py runs/prefix-typed-neu
.\.venv\Scripts\python.exe -m pytest -q
```

Ein Ausgabeordner muss neu sein. Die Skripte überschreiben keine bisherigen
Versuchsläufe. Protokoll und Teilresultate werden während der Ausführung
gesichert; `status=completed` bedeutet abgeschlossene Messung und ist getrennt
von `scientific_gate_passed`.
