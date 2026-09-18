# Erneuter Gesamttest mit Messung

**Ergebnis: bestanden.** Produktionscode, Konfiguration und gespeicherter Ausgangslauf wurden nicht geaendert. Alle Modellantworten wurden neu erzeugt.

## Umgebung und Methode

- Windows, Intel Core i5-13420H (8 Kerne, 12 logische CPUs), 15.7 GiB RAM.
- Lokale CPU-Inferenz mit vier PyTorch-Threads; Qwen2.5-0.5B-Instruct, gespeicherter Rank-8-LoRA, MiniLM und Qdrant Local.
- Ausgangspunkt: `runs/dragonfly-alias-002`, Paketversion 0.2.1. Mutierende Pruefungen verwenden temporaere DB-/Qdrant-Kopien.
- Sequentielle Abfragen, vorgewaermte Modelle; RSS-Speicherstichprobe alle 50 ms. P95 ist das empirische 95. Perzentil dieser kleinen Stichprobe.
- Drei frische Prozessstarts einschliesslich Imports, Modellladen, Hashpruefung, Routing und Antwort. Der Betriebssystem-Dateicache wurde nicht geleert; das ist kein Kaltstart nach einem Neustart des Rechners.

## Korrektheit

| Pruefung | Ergebnis |
| --- | --- |
| Vollstaendige pytest-Suite | 82 Tests, 0 Fehler, 0 Ausfuehrungsfehler |
| Alias-Erkennung | 35/35; sieben verschiedene Aliase, je fuenf Wiederholungen |
| Dragonfly mit harten Filtern | 54/54; 18 verschiedene Fragen, je drei Wiederholungen |
| Kosinus mit denselben harten Filtern | 54/54 |
| Echte Qwen/LoRA-Antworten | 18/18; sechs verschiedene Fragen, je drei Wiederholungen |
| Nicht unterstuetzte/unerlaubte Fragen | 4/4 gesperrt |
| Widerruf von Quelle, Generation, Repr?sentation und Trainingsquelle | Vier Szenarien bestanden; laufende Commits gesperrt, anderer Pod gueltig |
| Erneutes Training der beiden Pod-Vektoren | Beide erfolgreich; danach 7/7 Aliase erkannt |
| Vollstaendige frische CLI-Prozesse | 3/3 korrekte Antworten |
| Lineage-Audit | 33 Knoten und 3 Adapterartefakte verifiziert |
| Basisgewichte | Hash vor/nach Inferenz identisch |
| Wheel | 8 enthaltene Quelldateien stimmen byteweise mit dem Arbeitsverzeichnis ueberein |

## Warme Laufzeiten

| Vorgang | Stichproben | Median | P95 |
| --- | ---: | ---: | ---: |
| Neuronale Alias-Erkennung | 35 | 16.8 ms | 22.1 ms |
| Dragonfly-Routing mit Filtern | 54 | 22.5 ms | 29.0 ms |
| Kosinus-Routing mit Filtern | 54 | 18.2 ms | 26.1 ms |
| Antwort insgesamt: Routing, Adapter-Hash, Qwen, Commit | 18 | 930.8 ms | 972.5 ms |
| Davon Qwen-Generierung | 18 | 892.4 ms | 939.1 ms |
| Davon Commit | 18 | 6.3 ms | 17.6 ms |

Die mittlere Differenz zwischen Dragonfly und gefiltertem Kosinus betraegt hier 3.2 ms. Nach den harten Filtern bleibt fuer jede Frage genau **ein** Kandidat. Deshalb zeigt dieser Test keinen zusaetzlichen Auswahlvorteil des neuronalen Scorers im normalen Antwortpfad.

Der separate Diagnosevergleich ohne Entitaetsfilter ergibt weiterhin **18/18** fuer Dragonfly und **16/18** fuer Kosinus. Diese Diagnose wurde nicht zum Ausliefern von Antworten verwendet.

## Laden, Training und Speicher

- Vollstaendiger Prozessstart: 30.02 s, 34.08 s, 35.83 s; Median **34.08 s**.
- Encoder laden nach Bibliotheksimport: 0.495 s.
- Dragonfly/Qdrant aktivieren bei geladenem Encoder: 0.175 s.
- Qwen laden: 6.145 s; zusaetzliche Basis-Hashpruefung: 3.316 s.
- Adapter pruefen/laden: 0.135 s.
- Neues Pod-Vektortraining, je 300 Schritte inklusive Beispiels-Embeddings und Persistenz: 1.215 s / 0.639 s. Kein erneutes Training der Qwen-LoRA-Gewichte.
- RSS nach Encoder/Router: 538 MiB; nach Qwen/Adapter: 1982 MiB.
- Hoechster abgetasteter RSS im warmen Benchmarkprozess: 2522 MiB (2.46 GiB).
- Hoechster abgetasteter Prozessbaum-RSS der drei frischen CLI-Prozesse: 3035 MiB (2.96 GiB).
- 18 Antwortanfragen verbrauchten 65.59 CPU-Sekunden in 16.76 Sekunden Wandzeit, entsprechend rund 3.91 ausgelasteten CPU-Kernen.

## Grenzen und Nachweise

Dies sind wiederholte Regressionen an zwei Lieferanten, einem antwortenden LoRA-Pod und einem textuellen Distraktor. Sie messen weder grosse Pod-Mengen noch Parallelbetrieb. Die kurzen Antworten sind kein belastbarer Durchsatzbenchmark fuer lange LLM-Ausgaben. Ein gleich hoher Score auf diesem Datensatz beweist keine allgemeine Gleichwertigkeit der Router.

Das erste ad-hoc-Messskript uebergab die Trainingsnamen versehentlich durch eine ASCII-PowerShell-Pipeline. Die vorherigen warmen Messungen lasen ihre Fragen korrekt aus UTF-8-JSON und sind davon unabhaengig. Nur das betroffene neue Training und die folgenden Pruefungen wurden mit korrigierter Kodierung wiederholt; die Trainingsvorlagen wurden vor Ausfuehrung exakt mit dem gespeicherten Katalog verglichen. `initial-attempt.json` bewahrt den fehlgeschlagenen Versuch.

Die automatische Ausfuehrungsrichtlinie blockierte das Aufraeumen des ersten temporaeren Testordners mit `blocked by policy`. Der Ordner wurde belassen. Dies betrifft nicht die abgeschlossenen Testresultate.

Rohdaten: `measurements.json`, `pytest.xml`, sowie die stdout/stderr-Logs der einzelnen Unterprozesse. Wheel-SHA256: `0ccfda74e4e318a20a1ed9a54cf8bf662d460c38d38feb3288cb2ac3e3c70df0`.
