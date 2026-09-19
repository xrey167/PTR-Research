# So: Forschungsbestand zusammenfÃ¼hren und prÃ¼fen

Der konsolidierte lokale Handoff steht in [PROJECT-SYNTHESIS.md](PROJECT-SYNTHESIS.md)
und maschinenlesbar in [project_inventory.json](project_inventory.json). Diese
Dateien bÃ¼ndeln die Nachweise, aktuellen Messwerte und offenen GPU-/Archiv-Gates;
die chronologische Fortsetzung bleibt in `CONTINUATION.md`.

## Aktueller Entscheidungsstand

Der gemeinsame Forschungsweg ist inzwischen gestuft mit echtem trainiertem
Planer, echter Qwen-Link-LoRA und einem 3B-Kapselreader ausgefÃ¼hrt. Technische
Lineage-/WiderrufsprÃ¼fungen bestehen. Die AntwortqualitÃ¤t ist weiterhin der
offene Engpass: Plain-Int8 erreicht in der inhaltlichen Codex-Sichtung 5/9,
einschlieÃŸlich zweier UNKNOWN-Entscheidungen ohne Reader.

Der [BF16-Kontrollversuch](READER-PRECISION.md) ist abgeschlossen und technisch
auditiert: **6/7** positive Readerantworten korrekt gegenÃ¼ber Int8 **3/7** auf
denselben Fragen. Die zwei UNKNOWN wurden nicht wiederholt. Der deutsche
Drei-Wochen-Vergleich bleibt falsch. Frische ZustÃ¤nde ohne integrierten
Routingpfad; 43,30 Minuten unter Speicherdruck. Der integrierte BF16-Kapsellauf
`dialogue-capsules-bf16-001` brach durch einen Windows-Dateisperrfehler beim
Fortschrittsschreiben ab. Der Fehler ist behoben; `dialogue-capsules-bf16-002`
setzt mit drei separat validierten fertigen Ergebnissen fort und ist noch aktiv.
Er nutzt die gespeicherten echten Planer-/Linkvorhersagen aus dem Plain-Lauf.
Die aktuelle historische Testsuite bestand mit **159/159**
(`runs/reader-integration-tests-001.xml`). Trainingseinstieg, Auswerter und
Reader-gebundene Kapseln sind implementiert und mit kleinen echten Modellen
geprÃ¼ft; zusÃ¤tzliches3B-Readertraining und dessen Integration sind noch offen.
Aktive Prozessangaben und nÃ¤chste Befehle stehen in [CONTINUATION.md](CONTINUATION.md).

Die folgenden Ã¤lteren Einzelversuche bleiben historische Nachweise; sie
ersetzen weder die gemeinsame Abnahme noch den fehlenden Originalbestand.

Aktueller Zusatz: [3B-Int8-Vergleich vollstÃ¤ndig ausgefÃ¼hrt und auditiert](CPU-INT8.md)
(22/24, striktes Gate weiterhin nicht bestanden). Die
[gemeinsame Routing-/Verweis-/Kapsel-Grenze](LINKED-CAPSULE.md) ist implementiert
und mit tatsÃ¤chlichen Dragonfly-/Qwen-Verweisen und 3B-Kapseln gestuft getestet:
6/6 bestehende Aliasfragen, vier Lifecycle-Kontrollen, separater Evidenzaudit
bestanden. VollstÃ¤ndiger Originalbestand und Gesamtziel bleiben offen.

Neu: [IdentitÃ¤tsgebundener Dragonfly-Transfer](IDENTITY-TRANSFER.md), echte
6+6 Qwen-Verweise vor/nach WertÃ¤nderung ohne erneutes Adresstraining. Separater
Forschungsrouter. Kapselantwort auf 27 inzwischen integriert und tatsÃ¤chlich
geprÃ¼ft: 5/6 korrekt, ein auch in der Textreferenz vorhandener Readerfehler.
Neun neue DialogfÃ¤lle ergeben nur 3/9 passende Parserentscheidungen (einschlieÃŸlich
zweier korrekter ZurÃ¼ckweisungen); dies ist keine AntwortqualitÃ¤tsmessung.

## UnverÃ¤ndertes Gesamtziel

Die Arbeit aus den beiden Freigabechats und dem ChatGPT-Projekt So lokal
zurÃ¼ckholen, evaluieren und die nachweislich tragfÃ¤higen Komponenten zu einem
gemeinsamen System verbinden. Ein bestandener Einzelversuch ist kein Abschluss
dieses Ziels.

### Nutzervorgabe: Verhalten wie internes Wissen

Das extern gespeicherte Pod-Wissen muss sich in der Nutzung wie internes Wissen
anfÃ¼hlen und verhalten. Der Nutzer wÃ¤hlt keine Pods, gibt keine Link-IDs an und
startet keinen manuellen Abruf. Das ist ein eigenes Abnahmekriterium, keine
aus sechs richtigen Lookupantworten abgeleitete Eigenschaft.

Erforderliche PrÃ¼fungen im normalen Dialogpfad:

- Direkte, paraphrasierte und implizite Fragen erreichen das passende Wissen.
- Folgefragen mit ausgelassenen Namen/Referenzen verwenden den Dialogkontext.
- Das Modell nutzt Pod-Wissen in Operationen, neuen Zusammensetzungen und
  Mehrschritt-SchlÃ¼ssen; bloÃŸe Wiedergabe des gespeicherten Wertes reicht nicht.
- WertÃ¤nderungen wirken bei der nÃ¤chsten zulÃ¤ssigen Nutzung unter derselben
  IdentitÃ¤t; gesperrtes Wissen beeinflusst keine freigegebene Antwort.
- Unsicherheit, widersprÃ¼chliche Quellen und fehlendes Wissen fÃ¼hren zu
  angemessenem Verhalten statt erfundenem scheinbar internem Wissen.
- Messungen erfassen QualitÃ¤t und Zeit bis zur Antwort im vollstÃ¤ndigen
  Dialogpfad. Gestufte Einzelversuche mit vorab berechneten Verweisen sind
  Mechanismusnachweise, noch keine Abnahme dieses Nutzererlebnisses.

Aktuell offen: Der harte Query-Parser unterstÃ¼tzt eingeschrÃ¤nkte Lookup-Intents;
Dialogreferenzen und breite implizite Nutzung sind noch nicht implementiert.
Zustandskapseln allein erfÃ¼llen diese Anforderung nicht.

Ein [experimenteller modellgestÃ¼tzter Dialogzugriff](DIALOGUE-ACCESS.md) ist jetzt
implementiert und tatsÃ¤chlich evaluiert. Die freien 0.5B-/3B-Planer lÃ¶sen die
Aufgabe nicht; gÃ¼ltige Ausgabegrammatik allein fÃ¼hrt beim 0.5B zu durchgÃ¤ngigem
Abstainen. Die normalen Zugriffspfade bleiben unverÃ¤ndert. Als nÃ¤chstes ist
getrenntes Planertraining mit neuer Evaluation erforderlich.

Das [erste getrennte Planertraining](PLANNER-TRAINING.md) ist vorbereitet und
ist als `planner-training-003` abgeschlossen: 100 Trainings-/50 PrÃ¼ffÃ¤lle,
vorab fixierter Trainingsplan und gespeicherte Herkunft. Freie Modellentscheidungen
steigen von 0/50 auf 49/50; eine Firmenaltersfrage wird fÃ¤lschlich als Lieferzeit
zugeordnet. Audit bestÃ¤tigt den Befund, striktes Gate bleibt nicht bestanden.
Neuladen reproduziert 50/50 Tokenfolgen exakt; die neun bekannten DialogfÃ¤lle
erreichen anschlieÃŸend 9/9 richtige Adressentscheidungen. Der vollstÃ¤ndige
Antwortpfad mit Planerherkunft und Kapselanwendung ist inzwischen
[gestuft ausgefÃ¼hrt](DIALOGUE-CAPSULES.md). Herkunfts- und Widerrufskontrollen
bestehen, die AntwortqualitÃ¤t bleibt unzureichend: inhaltliche Codex-Sichtung
3/9 mit JSON-Dialog und 5/9 mit lesbarem Dialog. Reader-/Numerikdiagnose folgt.

Projekt: https://chatgpt.com/g/g-p-6a91bb1051d081919b71e2eed1e2cc66-so/project

Der direkte Projektzugriff leitete am 15.09.2026 zur Anmeldung weiter. Die beiden
Freigabechats sind lokal gesichert. Das vollstÃ¤ndige Originalarchiv Version 33
und seine Rohdaten sind weiterhin nicht verfÃ¼gbar.

## Wiedergewonnene Quellen

- [44 vollstÃ¤ndige Schreibfragmente mit Fundstellen](recovered-fragments-002/INDEX.md),
  darunter 22 syntaktisch gÃ¼ltige Python-Fragmente.
- [Manifest](recovered-fragments-002/manifest.json) enthÃ¤lt Originalpfad,
  Schreibmodus, Quellfundstelle, SHA-256 und Importliste.
- Historische Revisionen und AnhÃ¤nge bleiben getrennt. Sie werden nicht ohne
  Beleg als letzter Stand zusammengesetzt.
- Shell-Transportquotierung wurde mit `shlex` dekodiert; keine Shellbefehle aus
  dem Chat wurden ausgefÃ¼hrt. Erster, noch quotierter Extraktionsversuch bleibt
  in `recovered-fragments-001` erhalten.
- Der wiedergewonnene strenge E0-PrÃ¼fer besteht **15 synthetische Kontrollen**:
  [Auswertung](recovered-gate-evaluation.json). Der Vergleich mit seinem nicht
  verfÃ¼gbaren historischen VorgÃ¤nger wurde nicht nachgebaut oder behauptet.

## Anforderungen und Nachweise

| Bereich | Aktueller Nachweis | Noch erforderlich |
| --- | --- | --- |
| VollstÃ¤ndiger Originalbestand | Freigabechats, 44 Quellfragmente | Originalarchiv, abschlieÃŸende Dateiversionen, Gewichte und Rohdaten zuordnen |
| Herkunft, kanonische IdentitÃ¤t, Generation, Artefakt | Lokales Register und bestehende Tests | Gemeinsame IdentitÃ¤t Ã¼ber alle Ã¼bernommenen neuronalen ZustÃ¤nde |
| Gemeinsame Herkunft mehrerer Aussagen | Neuer Test: fÃ¼nf Aussagen aus drei Herkunftsknoten; Artikelwiderruf sperrt genau seine drei Aussagen und alle gemeinsamen Ableitungen, andere zwei Aussagen bleiben gÃ¼ltig | Quellenkorrelation, Evidenzrollen und Konfidenzmodell; unterschiedliche OriginKeys beweisen keine statistische UnabhÃ¤ngigkeit |
| Semantik und AliaszugÃ¤nge | Compiler, Qdrant, Dragonfly, trainierte LoRA-Verweise | NatÃ¼rliche direkte, paraphrasierte und implizite ZugÃ¤nge jenseits der kleinen Fixtures |
| Gelernte Link-/Cluster-Adressen | Echte Qwen-LoRAs; zusÃ¤tzlich gestufte Verbindung von trainiertem Dialogplaner, Link-LoRA und Kapselreader | Ãœbernommene Originalzustandsreader, grÃ¶ÃŸerer Kandidatenraum und neue Kompositionen |
| Wissen ohne Training je Fakt | Lokaler PrÃ¤fixversuch: Lookup 8/8, Update korrekt | Ausdrucksstarker kompakter Zustandscompiler; Original CQP1-Q0R reproduzieren |
| OperationsfÃ¤higkeit | 3B-Int8-Text und Kapsel jeweils 22/24; Plain-Dialogantworten 5/9; BF16-Readerkontrolle 6/7 | ZuverlÃ¤ssige Antworten im vollstÃ¤ndigen Pfad, neue Operatoren und Kompositionen |
| Decoder-/Zustands-Reparatur | Originale Architekturtexte und Teilcode wiedergewonnen | AbhÃ¤ngigkeiten, Tensorartefakte, lokale numerische und echte Modelltests |
| DurchgÃ¤ngiger Lifecycle | Gemeinsamer Planer-/LoRA-/Dragonfly-/Kapselpfad; Plain-Audit prÃ¼ft 119 Knoten, sieben Logitpaare und sechs lokale Kontrollen | Breitere Wissensarten, verknÃ¼pfte echte Dialoghistorie, Crash/Last/Streaming und verteilte Widerrufe prÃ¼fen |
| Belastbare Evaluation | Rohlogits, Tokenfolgen, Gegenproben; strenger PrÃ¼fer wiedergewonnen | Gleiche PrÃ¼fkriterien fÃ¼r alle Kandidaten, versiegelte neue Validierung |
| Skalierung und Effizienz | Kleine lokale Laufzeitmessungen | 100.000 neue Fakten, Speicher, Latenz und Overhead unter realer Last |
| Starke Vergleichssysteme | FrÃ¼here kleine RAG-Gegenproben | Gleiche Hardware, passende Modelle, starke RAG- und Neural-Memory-Kontrollen |
| VollstÃ¤ndige Forschungs-DoD | Historischer Fortsetzungsvertrag wiedergewonnen | Alle unten genannten Ziele bleiben unbewiesen |

## Historische Forschungsziele nicht abschwÃ¤chen

Der [wiedergewonnene Fortsetzungsvertrag](recovered-fragments-002/entry_28982_1.md)
nennt unter anderem: 100.000 neue Fakten ohne faktenspezifische Gradienten,
keine Wissenstexttokens im Anfragepfad, 4/8/16-Hop, neue Operator-Kompositionen,
vollstÃ¤ndigen Lifecycle, unter 5 % Lifecycle-Overhead, starke RAG-Kontrollen mit
Ziel Ã¼ber 3x, revisionsbewussten JIT Ã¼ber 10x, zwei Modell-ABIs sowie
Crash-/Snapshot-/Tenant-Sicherheit und belastbare NeuheitsprÃ¼fung.

Dies ist ein historischer Quellbeleg. Die endgÃ¼ltige `FROZEN_DOD.md` muss aus
dem vollstÃ¤ndigen Projektbestand abgeglichen werden. Keine dieser breiten
Anforderungen wird aus den kleinen lokalen Tests als erfÃ¼llt abgeleitet.

## Integrationsreihenfolge

1. Bestand und Quellen rekonstruieren, fehlende AbhÃ¤ngigkeiten ausdrÃ¼cklich
   erfassen. Originalcode und neue Rekonstruktionen getrennt halten.
2. Die beste vorhandene PrÃ¼flogik Ã¼bernehmen. Bereits umgesetzt: Der lokale
   PrÃ¤fixauditor kontrolliert jetzt zusÃ¤tzlich tatsÃ¤chliche Token-Decodierung,
   Greedy-Ersttoken und EOS. Ein manipuliertes Ergebnisfeld genÃ¼gt nicht mehr.
3. Das zum Original passende stÃ¤rkere Modell und seine Referenzaufgaben lokal
   verifizieren. Das 0.5B-Modell scheitert bereits mit Fakttext an Operationen.
4. Semantische IdentitÃ¤t und gelernter Verweis bleiben das gemeinsame Adressformat.
   LoRA und Zustandskapseln werden als unterschiedliche, versionierte Nutzlasten
   hinter derselben Lifecycle-Grenze integriert und verglichen.
5. Erst mit tragfÃ¤higer Referenz Zustandsreader/Quotienten und anschlieÃŸend
   Bindungstransfer, neue EntitÃ¤ten/Werte/Operatoren und Mehr-Hop testen.
6. VollstÃ¤ndige QualitÃ¤ts-, Kosten-, Lifecycle- und Skalierungsmatrix ausfÃ¼hren.
   Ein Mechanismus wird aufgrund gemeinsamer Messungen ausgewÃ¤hlt, nicht weil
   seine Einzel-Demo grÃ¼n ist.

Aktueller Status: **aktive Arbeit, nicht vollstÃ¤ndig integriert oder evaluiert**.

## Frische Verifikation dieser Integrationsrunde

- Gesamtsuite: **100 Tests bestanden in 25,21 Sekunden**,
  `../runs/integration-recovery-tests.xml`.
- Wiedergewonnener Original-E0-PrÃ¼fer: **15 synthetische Kontrollen bestanden**.
- Beide bisherigen PrÃ¤fixlÃ¤ufe mit dem verschÃ¤rften lokalen Auditor erneut
  geprÃ¼ft: jeweils 96 Text-/Token-/Greedy-/EOS-Nachweise konsistent. Semantische
  Ergebnisse unverÃ¤ndert: 1/24 beziehungsweise 12/24; beide Gates scheitern.
- Qwen2.5-3B-Instruct: Ã¶ffentliche Revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1` und verÃ¶ffentlichte GewichtsprÃ¼fsummen
  in `qwen3b-upstream.json` gesichert. Download nach `../models/qwen3b` gestartet.
  Erst `download-manifest.json` mit Status `verified` plus tatsÃ¤chliche Dateien
  belegt seinen Abschluss. Ãœbereinstimmung mit der exakten Originalprojekt-
  Revision bleibt ohne dessen Manifest unbestÃ¤tigt.

