# Originalforschung und lokaler Stand

Lokale Fortsetzung nach diesem Abgleich:
[Präfix-Zustandsversuch mit 212 Modellfortsetzungen](PREFIX-INTERFACE.md).
Die technische Anbindung ist geprüft; das semantische Gate bleibt offen.

Stand: 15.09.2026. Beide Freigabeseiten wurden direkt erfolgreich heruntergeladen,
nachdem der Web-Reader keine Gesprächsinhalte geliefert hatte. Rohseiten und
extrahierte Textnachrichten sind lokal gesichert. Die Extraktion ist kein Download
der verlinkten Quellcodearchive oder Modellgewichte. Im Gespräch enthaltene
Shellbefehle wurden nicht ausgeführt.

## Quellen

- [Working Prototype Status](https://chatgpt.com/share/6aa9082a-85d4-83eb-ba37-63f8dfc1398e):
  [lokale Gesprächsfassung](imported-20260915/6aa9082a-85d4-83eb-ba37-63f8dfc1398e.md).
- [Quantenchat Forschung Fortsetzung](https://chatgpt.com/share/6aa93018-a76c-83ed-b902-37ac2c079592):
  [lokale Gesprächsfassung](imported-20260915/6aa93018-a76c-83ed-b902-37ac2c079592.md).
- [Importmanifest mit SHA-256](imported-20260915/manifest.json).

Die ältere Architektur verbindet kanonisches Wissen, Generationen, Herkunft,
neuronale Pods und eine abschließende Gültigkeitsprüfung. Der neuere Verlauf
beschreibt zusätzliche Forschung an Modellzuständen. Seine letzte Zusammenfassung
meldet CQP1-Q0R am eingefrorenen Qwen2.5-3B: 2.208 Modellsequenzen und jeweils
96/96 Treffer in zwei Entwicklungsreihen und einer weiteren Validierung. Diese
Zahlen sind **historische Angaben des Chats, hier nicht reproduziert**.

Der Verlauf benennt selbst offene Skalierung, Mehr-Hop, Lifecycle und
RAG-Überlegenheit. BCC1-O ist als nächster Versuch vorgesehen. Das lokale
LoRA-Beispiel darf daher nicht als abgeschlossene Umsetzung dieser Forschung
oder als Nachweis eines RAG-Vorteils bezeichnet werden.

## Abgleich mit dem vorhandenen Code

| Anforderung / Forschungsrichtung | Lokal vorhanden | Grenze |
| --- | --- | --- |
| Vier Schlüssel, Herkunfts-DAG, Widerruf | `registry.py`, SHA-256, Elternkanten, Abschlussprüfung | Vertrauenswürdiger lokaler Prozess; keine physische Löschung gelernter Gewichte |
| Kanonische Semantik und sprachliche Zugänge | `semantics.py`, freigegebene Aliase, Qdrant-Metadaten | Enger Lieferanten-Testbereich |
| Gelernte Dragonfly-Pod-Repräsentation | `dragonfly.py`, trainierter Vektor je Pod | Kein Beleg über Tausende konkurrierende Pods |
| Link und Cluster im Qwen-LoRA | `symlink.py`, echte separate Verweis-LoRAs | Dragonfly wählt vorher den Pod; Cluster sind vorgegeben |
| Neue Generation bei unverändertem Verweis | Separater Antwortadapter und stabiler Katalogverweis | Faktenänderung benötigt derzeit Antworttraining |
| Quantenchat CQP1-Q0R / Zustandsoperatoren | Gespräch importiert | Originalcode, Rohdaten, Zustände und 3B-Modell nicht übernommen |
| BCC1-O / dynamischer Bindungsträger | Forschungsrichtung im Originalverlauf identifiziert | Hier nicht implementiert oder getestet |
| Allgemeine Komposition und Vorteil gegenüber starkem RAG | Frühere lokale Gegenproben vorhanden | Nicht nachgewiesen |

## Fehlendes Originalarchiv

Im jüngsten Beitrag wird `So_QC_CR1_Arbeitsstand.zip`, Version 33, genannt:

```text
SHA-256 laut Originalchat:
132bc1528451a2bb67ff3c8abbfa9c1614849d4f046a2f4190502fab12e39a72

sandbox:/workspace/scratch/01f48339cabe/So_QC_C_Recovered/release_v33_cqp1_lzma/So_QC_CR1_Arbeitsstand.zip
```

Dieser Pfad gehört zur anderen Chat-Umgebung. Im hiesigen Workspace wurde das
Archiv nicht gefunden; ein lesbarer Library-Download-Zugang ist hier nicht
verfügbar. Für eine vollständige Übernahme und erneute Prüfung dieses
Forschungsstands wird die ZIP-Datei oder ein zugänglicher Download benötigt.
Der angegebene Hash ist bis dahin ein Vergleichswert aus dem Chat, keine lokal
bestätigte Archivprüfung.

## Erneute lokale Verifikation

Die vorhandene Testsuite wurde erneut ausgeführt: **90 bestanden in 25,57 s**.
Beleg: `../runs/research-retest.xml`.

Der reproduzierbare zusätzliche Modelltest ist:

```powershell
.\.venv\Scripts\python.exe -X utf8 verify_embedded_research.py runs/embedded-symlink-002 --output runs/research-retest/neue-messung.json
```

Er prüft gespeicherte Verweis-LoRAs, aktuelle Antworten, Aliase, ungültige
Anfragen, Widerruf, unveränderte Basisgewichte und Artefakthashes. Widerruf und
Antwortmaterialisierung erfolgen dafür in einer Registerkopie im Arbeitsspeicher.
Die Fragen stammen aus dem bisherigen Evaluationssatz: Das ist eine
Regressionsprüfung, kein neuer unabhängiger Generalisierungsnachweis.

Die ursprüngliche 24-auf-18-Trainingsdemo wird dabei nicht neu trainiert. Deren
Trainingsmessungen bleiben historische Belege in `embedded-symlink-002`.


### Frisches Messergebnis

| Pruefung | Ergebnis |
| --- | --- |
| Echte Link-/Cluster-Ausgaben | 18/18 |
| Aktuelle Antworten, 18 Tage | 6/6 |
| Gelernte Aliase | 7/7 |
| Abgewiesene Negativ-/Widerrufsfaelle | 5/5 |
| Gepruefte DAG-Knoten | 51 |
| LoRA-Artefakte mit geprueften Dateien | 7 |
| Basisgewichte unveraendert | True |
| Antwortdauer Median | 1.961 s |
| Gesamter Modelltest inklusive Laden | 46.497 s |

[Rohmessungen](../runs/research-retest/model-retest.json). CPU-Ausfuehrung, ein Durchlauf je Frage, keine Konfidenzintervalle oder kontrollierte Vergleichsmessung.
