# Qwen2.5-3B auf dem lokalen Windows-Rechner

## Ausführung und Prüfbarkeit

Der Download der veröffentlichten Revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1` ist vollständig und gegen die beiden
veröffentlichten Gewichts-SHA-256 geprüft. Die exakte Revision des früheren
ChatGPT-Projekts bleibt ohne dessen Originalmanifest unbestätigt.

Die aktuelle Maschine hat frisch gemessen **15,65 GiB physischen RAM**. Die
frühere Annahme von 32 GB war falsch. Der Float32-Versuch wurde nach starkem
Speicherdruck beendet, der BF16-Versuch nach umfangreicher Auslagerung und nur
einer fertigen Kapsel. Diese Läufe bleiben erhalten.

`cpu_int8.py` verwendet die installierte PyTorch-Implementierung dynamischer
Int8-Linear-Schichten mit oneDNN. Es trainiert keine Gewichte. Embeddings werden
in BF16 gespeichert und liefern FP32-Aktivierungen; die übrigen Berechnungen
verwenden Float-Aktivierungen. Die ursprünglichen Qualitätskriterien bleiben
gleich, die Numerik ist aber ausdrücklich eine **eigene quantisierte Variante**.
Dies ist keine bitgleiche Reproduktion des ursprünglichen 3B-Versuchs.

Der Modellfingerabdruck umfasst auch gepackte Quantisierungsgewichte, Skalen,
Nullpunkte und Bias. Diese wären in `named_parameters()` allein nicht enthalten.
Der Hash unveränderter Float-Modelle bleibt kompatibel. Vier kleine
Implementationsgrenzen sind geprüft: Float-Hash-Kompatibilität, BF16-Hash,
kompletter quantisierter Decoder-Forward und Erkennung geänderter gepackter
Gewichte. Gesamtsuite vor Start des vollständigen Laufs: **107 bestanden**.

PyTorch 2.14 warnt, dass die verwendeten Quantized-Tensor-APIs künftig entfallen.
Der aktuelle Versuch verwendet die vorhandene, getestete Version; daraus folgt
keine Kompatibilitätszusage für spätere PyTorch-Versionen.

## Einordnung des abgebrochenen Int8-Laufs 003

Der private Speicherwert stieg beim Aufbau auf etwa 10 GB. Dieser Wert ist nicht
mit dem tatsächlich residenten Arbeitsspeicher gleichzusetzen. Der Prozess wurde
wegen des beobachteten Speicherdrucks gestoppt; die erst anschließend ausgewerteten
Puffer zeigten jedoch bereits **60 fertige Modellantworten**. Die frühere Aussage,
dass damit keine nutzbare Inferenz möglich sei, war zu weitgehend.

Ein separater Teil-Audit prüft Text-Decodierung, EOS und Sollwerte der erhaltenen
Antworten: Text und gespeicherter Zustand jeweils 14/15, absichtlich anderer
Zustand 15/15 gegen seinen eigenen Wert, Nullkontrolle 7/15. Vollständige rohe
Logits wurden in dieser alten Skriptversion erst am Ende gesichert und fehlen.
Deshalb ist **kein vollständiger Audit** dieses abgebrochenen Laufs möglich.

## Verbesserte Datensicherung in Lauf 004

Nach jeder Vierergruppe werden vollständige Logits und Ergebnisse über temporäre
Dateien und atomaren Dateiaustausch gespeichert. Frische RSS-/Private-/Seitenfehler-
und verfügbare-RAM-Messungen begleiten die Zwischenstände. Ein alter Messpunkt
ersetzt keine aktuelle Fortschritts- oder Speicherprüfung.

Der Lauf wiederholt dieselben 96 Kontrollantworten, anschließend Update und
Lifecycle-Prüfungen. Er ist eine Wiederholungs-/Vergleichsmessung am bestehenden
Datensatz, kein neuer unabhängiger Generalisierungsnachweis.

```powershell
.\.venv\Scripts\python.exe research/run_prefix_probe.py --model models/qwen3b --dtype bfloat16 --quantization dynamic_int8 --typed-format --output runs/3b-int8-neu
.\.venv\Scripts\python.exe research/audit_prefix_probe.py runs/3b-int8-neu --tokenizer models/qwen3b
```

Quellen und Rohdaten:

- `models/qwen3b/download-manifest.json` im Projektverzeichnis
- [Teil-Audit 003](../runs/prefix-interface-3b-003/partial-audit.json)
- [Ergebnisse 004](../runs/prefix-interface-3b-004/results.json)
- [107 Tests](../runs/cpu-int8-tests.xml)

Ein kleineres Gewichtsformat belegt allein weder niedrigen Gesamtspeicherverbrauch
noch höhere Geschwindigkeit oder Antwortqualität. Dafür sind die tatsächlichen
Messungen und Gegenproben maßgeblich.

## Vollständiger Abschluss von Lauf 004

Der Prozess ist mit Exit 0 beendet. Der separate Auditor bestätigt alle 96
Kontrollantworten anhand von vollständigen Rohlogits, Token-Decodierung, erstem
Greedy-Token und EOS. Alle 24 Vergleiche frisch/gespeichert sind bitidentisch.

| Prüfung | Ergebnis |
| --- | --- |
| Frische Textreferenz | 22/24 |
| Gespeicherte Zustandskapsel | 22/24 |
| Wertabfrage | 8/8 |
| Addition von zwei Tagen | 7/8 |
| Vergleich mit 25 Tagen | 7/8 |
| Anderer Zustand gegen seinen eigenen Sollwert | 22/24 |
| Antwortwechsel durch anderen Zustand | 22/24 |
| Nullkontrolle: UNKNOWN ohne Fakt | 16/24 |
| Update 18 auf 24 ohne Training | 24, einschließlich EOS korrekt |
| Lifecycle-Prüfungen | 6/6 |
| Gewichte einschließlich gepackter Int8-Gewichte unverändert | Ja |
| Median Zustandskapsel | 1,375 s |
| Median frischer Textpräfix | 2,578 s |
| Gesamter Lauf einschließlich Laden, Quantisierung und Prüfungen | 346,09 s |
| Striktes wissenschaftliches Gate | Nicht bestanden |

Die zwei Fehler sind konkret erhalten: Für Vost mit 18 Tagen liefert die
Addition von zwei Tagen 30 statt 20; bei Vost mit 42 Tagen wird die Frage nach
mehr als 25 Tagen mit NO beantwortet. Beide Fehler treten bereits in der
gleichwertigen Textreferenz auf.

Die Messung ist kein kontrollierter allgemeiner Geschwindigkeitsvergleich und
kein starker RAG-Benchmark. Auch die Verbesserung gegenüber 12/24 des 0.5B-Laufs
ist ein Vergleich zweier Modell-/Numerikvarianten am selben kleinen Datensatz,
keine unabhängige Generalisierungsvalidierung.

[Vollständiger Audit](../runs/prefix-interface-3b-004/audit.json).
