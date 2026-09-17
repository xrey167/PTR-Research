# Im Qwen-LoRA eingebettete semantische Verweise

## Verifizierter Lauf: 0.3.0

Der fertige Lauf liegt in `runs/embedded-symlink-002`.

| Nachweis | Ergebnis |
| --- | --- |
| Gesamte Testsuite | 90 Tests bestanden |
| Echte Qwen-Link-/Cluster-Ausgaben | 18/18 korrekt, drei getrennte Verweis-LoRAs |
| Echte Antworten ueber den Verweis | 6/6 mit 24 Tagen, nach Update 6/6 mit 18 Tagen |
| Linkgewichte und Cluster nach Werteupdate | Identisch; Link-Adapter nicht neu trainiert |
| Alte Snapshots und widerrufene Zielquelle | Gesperrt |
| Qwen-Basisgewichte | Vor und nach Training/Inferenz identisch |
| Herkunfts-/Integritaetspruefung | 51 Knoten und sieben LoRA-Artefakte verifiziert |
| Separater CLI-Prozess | `LINK 1 CLUSTER 1` und anschliessend `18 days` |

Das Training der drei Link-LoRAs dauerte 43,3 / 85,1 / 77,6 Sekunden; das Training
des neuen 18-Tage-Antwort-LoRAs 45,8 Sekunden. Der vollstaendige Demo-Lauf dauerte
343 Sekunden. Eine Antwort ueber beide Modellaufrufe dauerte in den zwoelf
Messungen im Median **2,67 Sekunden**. Der einzelne frische CLI-Prozess brauchte
20,1 Sekunden. Das sind CPU-Messungen dieses kleinen Beispiels, kein gepaarter
Leistungsbenchmark gegen den frueheren Ein-Aufruf-Pfad.

Rohdaten: `symlink-report.json` und `verification.json` im Laufverzeichnis;
Gesamttests: `runs/embedded-symlink-tests.xml`.

```powershell
.\.venv\Scripts\python.exe ask_semantic.py runs/embedded-symlink-002 "X12 procurement delay from Mueller GmbH?"
```

Der Pod erhaelt einen echten, separat trainierten Qwen-LoRA-Verweisadapter.
Dieser lernt zu Alias- und Frageformulierungen eine Ausgabe wie:

```text
LINK 1 CLUSTER 1
```

Die beiden Zahlen kommen bei der Inferenz aus den Modellgewichten. Weder die
Sollzahlen noch der Faktenwert werden dem Verweis-Prompt als Kontext mitgegeben.
Der System-Prompt legt nur das Ausgabeformat fest. Die kanonischen IDs bleiben
im vertrauenswuerdigen Katalog hinterlegt; die kleinen Integer sind dessen
persistente lokale Modellvokabeln, keine globalen kryptographischen Identitaeten.

## Verbindung von Semantik, Pod und Generation

```text
Alias/Frage
   -> Dragonfly waehlt die passende Wissensidentitaet
   -> Qwen + Verweis-LoRA erzeugt LINK n CLUSTER m
   -> Katalog prueft Identitaet und Cluster
   -> stabiler KnowledgeKey wird auf die aktuelle Generation aufgeloest
   -> deren aktueller Antwort-LoRA erzeugt den Faktenwert
   -> gemeinsamer Snapshot und Commit
```

Beispiel: `LINK 1` bleibt derselbe trainierte Verweis, wenn sich der Faktenwert
von 24 auf 18 Tage aendert. Es wird ein neuer Antwort-Pod fuer die neue Generation
bereitgestellt. Der Verweis-LoRA wird dabei nicht neu trainiert. Ein bereits
laufender Snapshot der alten Fakten-Generation bleibt trotzdem gesperrt.

Das ist eine kontrollierte logische Indirektion, kein Betriebssystem-Symlink
und kein physischer Speicherzeiger im Tensor. Der verifizierte Katalog bleibt
fuer die aktuelle Generation notwendig. Die Zuordnung wird nach dem ersten
Modellaufruf erneut geprueft; ein falscher oder unvollstaendiger Verweis fuehrt
zum Abbruch. Es gibt keine Ersetzung einer falschen Modellantwort durch die
erwartete Link-ID.

## Was bedeutet Cluster?

Der Semantic Compiler liefert einen deterministischen `semantic_cluster`:

```text
ClusterKey = SHA256(schema, domain, type, predicate, role)
```

Die Felder werden eindeutig als kanonisches JSON serialisiert. Beispielsweise
koennen Lieferzeit-Fakten verschiedener Firmen im selben Cluster liegen,
waehrend ein Regel-Pod einen anderen Cluster erhaelt. Das sind vorgegebene
semantische Gruppen. Qwen lernt ihre Zugehoerigkeit; die Gruppen werden nicht
durch unbeaufsichtigtes Clustering entdeckt.

Jeder Verweisadapter lernt die Cluster-ID seines eigenen Pods. Er ist kein
universeller Suchrouter: Dragonfly aktiviert zuvor den passenden Pod. Die
Clusterkennung wird beim Aufloesen mit den aktuellen harten Metadaten verglichen.

## Herkunft und Widerruf

Der Katalog fuehrt fuer jede Wissensidentitaet eine eigene versionierte
Identitaetsbeschreibung. Sie enthaelt KnowledgeKey, ClusterKey, Subjekt, Komponente
und freigegebene Aliase, aber keinen veraenderlichen Faktenwert.

- Der Verweis-LoRA haengt von dieser Identitaetsgeneration, deren tatsaechlichen
  Quellen und seinen Trainingsdaten ab.
- Der Antwort-LoRA haengt von der aktuellen Fakten-Generation ab.
- Der Antwort-Snapshot enthaelt beide Ableitungen und die Dragonfly-Repräsentation.
- Cluster- oder Alias-Aenderungen sperren den alten Verweis und erfordern neues
  Verweistraining. Ein reines Werteupdate erhaelt den Verweis.
- Widerruf einer Identitaetsquelle sperrt den abgeleiteten Verweis. Widerruf der
  aktuellen Faktenquelle verhindert das Aufloesen auf diesen Fakten-Pod.
- Mitglieder eines Clusters haben getrennte Ableitungen. Der Widerruf eines
  Lieferanten hebt nicht automatisch die Rechte des anderen Lieferanten auf.

Die Quellen der Identitaetsbeschreibung werden aus den gueltigen Hard-Metadaten
abgeleitet. Die Quelle bleibt als Elternknoten erhalten; auch bei einem spaeteren
Werteupdate kann ihr Widerruf einen neuen Identitaetsnachweis notwendig machen.

## Lokal trainieren und pruefen

```powershell
.\.venv\Scripts\python.exe train_symlink_lora.py runs/dragonfly-alias-002 --output runs/mein-symlink-lauf
.\.venv\Scripts\python.exe ask_semantic.py runs/mein-symlink-lauf "X12 procurement delay from Mueller GmbH?"
.\.venv\Scripts\python.exe -m pytest -q
```

Die Demo trainiert drei Verweis-LoRAs: zwei Lieferanten-Fakten mit gemeinsamem
Cluster und einen Regel-Pod in einem zweiten Cluster. Danach trainiert sie einen
neuen Antwort-LoRA fuer 18 Tage und prueft die Wiederverwendung desselben
Verweisadapters. Der neue Lauf befindet sich in einem eigenen Verzeichnis;
Ausgangsregister und Ausgangsgewichte werden nicht veraendert.

Die Verweis-LoRAs verwenden Rank 8, 48 Schritte und Lernrate 0,0003; der
Antwort-LoRA verwendet standardmaessig 32 Schritte und Lernrate 0,003.
Basisgewichte bleiben eingefroren. Rohdaten, Trainingsverluste, echte
Modellausgaben und Laufzeiten stehen in `symlink-report.json`.

Der erste Entwicklungsversuch mit Lernrate 0,003 fuer den Verweis war instabil
und wurde wegen einer unvollstaendigen Ausgabe abgebrochen. Er bleibt in
`runs/embedded-symlink-001` als fehlgeschlagener Lauf dokumentiert.

## Grenzen

Der zweistufige Pfad benoetigt einen zusaetzlichen Modellaufruf und ist daher
langsamer als die bisherige direkte Pod-Antwort. Die Demonstration prueft einen
kleinen synthetischen Katalog; sie belegt keine Skalierung auf viele Cluster,
kein allgemeines Reasoning ueber Linkketten und kein Unlearning aus kopierten
Gewichten. Training von beliebigen neuen Verknuepfungen benoetigt weiterhin
explizite Identitaets- und Quellenangaben.
