# Aktueller Stand: typed Reader-LoRA registriert, Integration offen

## Verifiziert am 15.09.2026

Die alten CPU-/Session-Hinweise weiter unten sind historische Handoffs. Der
aktuelle Stand ist:

- GPU-Host `xrey@xrserver`: echter Qwen2.5-3B-LoRA-Lauf, 46 Updates, Basis
  unverändert, Adapter verändert.
- Lokaler Search-Backend ohne Turbopuffer-API: SQLite-WAL, Branches,
  Pin/Unpin, BM25/Regex/Vector-Fusion, ACL/Provenienzfilter und parallele
  iterative Such-Episoden.
- Typed Pods: `context`, `math`, `model`; jeweils mit eigenem Vertrag und
  gemeinsamem Origin/Knowledge/Generation/Artifact-Lineage.
- Reader-Rohqualität: 79/92 auf dem eingefrorenen Testsplit. Die typisierte
  Math-Guard-Audit korrigiert denselben Split reproduzierbar auf 92/92.
- Typed-Pod-Wochen-Diagnose: 16/24 roh, 24/24 nach deterministischer Guard;
  die Rohfehler bleiben als Modellgrenze dokumentiert.
- Typed-Curriculum-Adapter `reader-lora-gpu-004` ist als Model-Pod in
  `runs/reader-model-registry-004.sqlite3` registriert; Reader-Identität und
  Adapter-/Protokollhash stimmen mit dem Trainingsreport überein.
- Gesamtsuite: 172 Tests bestanden, zwei bekannte Dependency-Warnings.

Belege und Maschinenstatus stehen in `research/PROJECT-SYNTHESIS.md` und
`research/project_inventory.json`.

Lokaler Handoff konsolidiert: research/PROJECT-SYNTHESIS.md beschreibt den
gemeinsamen Origin->Knowledge->Generation->Dragonfly/Symlink->Planner->Link-
>Reader->Lifecycle-Pfad, Nachweise und Grenzen. research/project_inventory.json
ist maschinenlesbar validiert (schema neural-pods-project-inventory:v1,
status assembled_not_complete, 5 offene Gesamtanforderungen). INTEGRATION-PLAN.md
verlinkt beide Dateien. Das ersetzt keine GPU-Ausfuehrung, sondern macht den
aktuellen Projektstand und die naechsten Schritte reproduzierbar.

Der erste echte3B-Reader-Preflight wurde nach mehr als20Minuten ohne einen
abgeschlossenen Optimizerschritt beendet (nur1Forward, CPU ca1274s, private
ca7.95GB). Ergebnis ist runs/reader-training-preflight-001/host-limit.json mit
status=aborted_host_limit; kein Trainings-/Qualitaetsergebnis. Der Prozess
27116 wurde beendet, weil der lokale CPU/Paging-Host keinen verwertbaren Schritt
lieferte. research/train_reader.py wurde danach auf einen begrenzten
Ein-Schritt-Preflight (--probe-microbatches1) und laufenden30s-Monitor geaendert;
Pycompile und tests/test_reader_training_run.py (1pass) bestanden. Ein erneuter
3B-Preflight wird erst auf geeigneter GPU/Host sinnvoll ausgefuehrt.

19:07-19:08 weitere Livepruefung des Preflights: PID27116 seit18:48 aktiv,
CPU ca1210s, private memory ca7.96GB, Responding=true. Report bleibt beim
ersten backward (forward_calls1, steps[]); keine Exception, kein terminaler
Status. Der neue Monitor ist erst fuer kuenftige Prozesse wirksam.

Trainer-Fix: research/train_reader.py hat nun einen daemon Monitor, der waehrend
langem Backward alle30s elapsed/RSS/verfuegbaren Speicher schreibt. Pycompile
bestanden. Der bereits laufende PID27116 wurde vor dieser Aenderung geladen und
nutzt sie nicht; sein Prozesszustand (responsive, CPU ca1096s, private ca7.92GB)
bleibt die autoritative Messung.

15.09.2026 weitere Preflight-Beobachtung: PID27116 bleibt responsive und
CPU-Zeit stieg auf ca1006s; privater Speicher ca7.92GB. Report weiterhin
phase=preflight, forward_calls=1, steps=[] (Heartbeat wird erst nach dem
langen ersten backward geschrieben). Kein Fehler/Abbruch festgestellt.

Naechste Beobachtung des echten Reader-Preflights: PID27116 weiterhin
responsive, BF16/Qwen3B CPU, ca7.93GB private memory, CPU-Zeit ca829s;
forward_calls=1 und steps=[] im letzten atomaren Report, da der erste
Rueckwaertslauf noch laeuft. Keine zweite Modellinstanz gestartet und kein
Trainingsergebnis behauptet. Nach terminalem Status sind die Messwerte zu
pruefen; erst dann darf full train oder eine begrenzte Folgeprobe starten.

Der integrierte Fortsetzungslauf ist jetzt terminal completed und der aktuelle
Audit meldet audit_passed=true: 9 Readerfaelle (7 positive, missing_part und
unsupported_metric UNKNOWN), 127 verifizierte Graphknoten, 43 Quellstaende,
7 vollstaendige Cached/Fresh-Logitpaare, 3 uebernommene Rows sowie alle sechs
Planner/Request/Link/Fact-Widerrufskontrollen bestanden. Antwortqualitaet ist
kein automatischer Gate; der german_followup bleibt semantisch falsch (Ja/27
statt Nein bei drei Wochen). Audit meldet full_research_goal_complete=false.

Nach Abschluss und Audit des integrierten Laufs (Session1532, Exit0; 9 Faelle,
127 Knoten, controls bestanden) wurde die echte 3B-Reader-Speicherprobe gestartet:
PID27116, output runs/reader-training-preflight-001. Eingaben/Modellhash verifiziert,
29.933.568 LoRA-Parameter, BF16 CPU, laengster Prompt143 Tokens. Status bleibt
preflight waehrend des ersten Rueckwaertslaufs; Prozess reagiert und CPU-Zeit steigt,
privater Speicher ca7.93GB. Noch kein Optimizerschritt abgeschlossen. Nicht
abbrechen oder parallel einen weiteren3B-Prozess starten.

Neu: research/reader_capsule.py verbindet volle Reader-Identitaet und
registrierte LoRA-/Trainingslineage direkt mit jedem KV-Artefakt.
Tiny-Qwen/LoRA-KV-Test Session51215 Exit0,1pass31.45s: frischer/cached
Decode mit exakten Tokens/Logits, skalierungsbedingte Sperre, Trainingrevoke
sperrt Kapsel/Commit, unabhaengiger Lieferantenfakt bleibt gueltig.
Manuell gesetzte kleine LoRA-Fixture, KEIN3B-Training und keine Qualitaetsmessung.
Vollhash an Nutzungsgrenzen ist teuer, kein Performanceanspruch.
Bestehender grosser Lauf nutzt weiter seinen alten eingefrorenen Quellstand.

GANZE Testsuite TERMINAL: Session31002 Exit0,159passed,2bekannteWarnungen,
82.00s. JUnit runs/reader-integration-tests-001.xml. Das deckt den aktuellen
Code einschliesslich Training/Evaluator/ReaderCapsules ab, nicht echte3B-
Trainingsqualitaet oder das Gesamtziel. Session1532 weiterhin live,
zuletzt german_followup/cached_decode,193Forwardcalls,2196.02s.
followup_reference inzwischen fertig, korrekte27Tage; sechs fertige Rows,
davon drei aus validiertem Altlauf uebernommen. Rest German und2UNKNOWN,
danach Endhash/Audit. Keine Speicherdruck bedingte Neustarts.

Neuester Stand: research/evaluate_reader.py implementiert echte getrennte
Basis-/Adapter-Evaluation fuer eingefrorene dev/test-Splits. Noch KEINE3B-
Ausfuehrung. BF16/eager identisch fuer beide Varianten, Vollidentitaet nach
Adapterreload erforderlich, Outputs/Tokens/ersteLogits/Hashes/Zeit gespeichert;
ExactMatch ausdruecklich Formatmetrik. Keine Runtime-Lineage-Behauptung.
Trainingsrunner setzt Adapter-Inferenzkonfiguration vor Save/Identitaet.
Tiny-Qwen-Train/Reload-Test jetzt auch mit kompletter Identitaetsgleichheit:
Session10877 Exit0,1pass35.11s. Evaluator-End-to-End mit echtem TinyQwen plus
Qwen-ByteTokenizer: Session67363 Exit0,1pass29.09s. Getrennte Sentinel-Tokens
pruefen, dass Target und Assessment nicht in Prefix/Suffix gelangen.
Frueher Evaluatortest scheiterte: AutoTokenizer rekonstruierte generischen
WordLevel als QwenBPE und lieferte leere Segmente. Test nutzt jetzt echten
QwenTokenizer; Evaluator lehnt leere Segmente explizit ab.
Lauf1532 weiter live: followup_reference/fresh_reference,161Forwards,
1817.67s zuletzt. Fuenf fertige Rows (3validiert uebernommen,2neu);
followup_buffer korrekt29Tage. Nicht neu starten. Nach terminal auditieren,
dann Deadline-Diagnose bzw.3B-Trainingsspeicherprobe. Details und Befehle in
READER-TRAINING-RUN.md. Die unten genannte Session64227 ist aelter.

Neuer Fortschritt: research/train_reader.py ist jetzt ein ausfuehrbarer
Preflight-/Trainingseinstieg; Anleitung research/READER-TRAINING-RUN.md.
Der echte eingefrorene Bundle-Check lief erfolgreich (360 Train,46 Updates),
--help ebenfalls. Tests/test_reader_training_run.py: Session64227 Exit0,
1 passed in34.91s (PEFT-Warnung zu fehlender Config beim synthetischen Modell).
Vier reale Tiny-Qwen/LoRA-Updates pruefen Schedule, partielle Fenster,
unveraenderte Basis, veraenderte Adapter und exakt gleiche Reload-Logits.
Erster Lauf scheiterte an automatischem Attention-Backend beim Reload;
explizites eager auf beiden Seiten behebt den Vergleich, ohne Toleranzlockerung.
3B-Preflight und Training NICHT gestartet. Der Trainingsrunner speichert
Status trained_not_evaluated, noch keine Registry-/Kapselintegration.
Session1532 weiter live gepollt; zuletzt followup_buffer/fresh_reference,
125 Forwardcalls,1416.99s. buffer fertig mit korrekten29Tagen.
Die alte Aussage unten "Trainingseintrittspunkt offen" ist damit ueberholt;
Speicherprobe, echtes3B-Training und Evaluation bleiben offen.

Trainingsinputs jetzt eingefroren: research/prepare_reader_training.py wurde
ausgeführt nach runs/reader-training-inputs-001. Status prepared_not_trained,
360/92/92,46Optimizerupdates bei2Epochen,3Warmupupdates,letztesFenster8.
Fünf Input-/Konfig-/Modellmanifestdateien und acht relevante Quellstände mit
SHA-Hashes;13Dateien anschließend nachgeprüft. Paketversionen und feste
Auswahl-/Bewertungspolitik gespeichert. Große Modellgewichte wurden bei dieser
leichten Vorbereitung ausdrücklich nicht erneut gehasht. Vollständiger
Trainingseintrittspunkt, Gerätespeicherprobe und Training weiterhin offen.

Vorbereitung LoRA-Readeridentität: research/reader_identity.py bindet neben
Gewichtshash auch Modell-/Adapterkonfiguration und tatsächliche Skalierung,
Aktivierung/Deaktivierung/Mergeflags der LoRA-Layer. Echter kleiner Qwen/PEFT-
Test bestanden: Session3912 Exit0,1/1 in40,66s. Halbierte Skalierung und
deaktivierter Adapter ändern Identität bei unverändertem Gewichtshash.
Noch NICHT in bestehende Kapselläufe integriert; für zukünftigen trainierten
Reader vorgesehen. Keine allgemeine ABI-/Backendattestation. Lauf1532 bleibt
aktiv, zuletzt direkt gepollt ohne neue fertige Antwort.

Fortschrittsfix jetzt auch in probe_reader_precision.py/Diagnosemodus
übernommen (Syntaxprüfung bestanden). tests/test_progress_json.py hat nun
einen echten Windows-Test: offener Lesehandle verhindert replace, Timer schließt
ihn, Retry schreibt erfolgreich. Alle3 Tests bestanden in0,331s; kein Mock
im Windows-Sperrtest. Das reproduziert die Fehlerklasse, identifiziert aber
nicht den konkreten damaligen Leser/Prozess. Lauf1532 weiter aktiv.

## Aktive Fortsetzung: Session1532 / PID23476

Neuer Run runs/dialogue-capsules-bf16-002 gestartet mit:
python -X utf8 research/run_dialogue_capsules.py reader --reader-input plain
--reader-precision bf16 --resume-from runs/dialogue-capsules-bf16-001
--run runs/dialogue-capsules-bf16-002.
Session1532 direkt gepollt, Modell lädt, reader.json running/load, PID23476.
NICHT erneut starten. Altlauf95788 bleibt terminal Exit1.

Übernahme vor Modellladung validiert durch research/resume_dialogue.py:
drei fertige Receipts, sechs Logittensoren, Kapseldateien, Herkunfts-/Quellhashes,
Token/Text/EOS und exakte Linkzeilen. Separater tatsächlicher Audit
research/audit_resume_source.py: Session58444 Exit0,3Rows/6Tensoren;
vier manipulierte Kopien (Reihenfolge/Text/Token/Modellhash) abgewiesen.
Ergebnis in Altlauf/resume-validation.json. Restlauf beginnt ab buffer;
fertige Fälle werden nicht neu inferiert. Exakter Modellhash wird nach Laden
gegen Altlauf geprüft, bestehende BF16-Kapsel erneut validiert/gebunden.

Neuer Run bewahrt failure.json, ursprüngliche Logits und Quellsnapshot separat
als resume-source-* auf. Abschließender audit_dialogue_capsules.py verifiziert
auch deren Hash, unveränderte übernommene Zeilen/Tensoren und alten Snapshot.
Auditänderung nach neuem Quellsnapshot; noch nicht auf fertigem Restlauf geprüft.
Wichtige Grenze: Altlauf hatte keinen abschließenden Gewichtsvergleich wegen
Abbruch; neue Metadaten kennzeichnen dies ausdrücklich. Neuer Endhash beweist
nicht rückwirkend den fehlenden Endhash des Altprozesses.

## WICHTIG: Integration jetzt terminal fehlgeschlagen, nicht mehr live

Session95788/PID29648 endete mit Exit1 beim Windows-Replace der Fortschrittsdatei
(PermissionError/WinError5). Kein Modell-/Logitvergleichsfehler festgestellt.
Fehler während fresh_reference/buffer bei169 Forwards,2282,49s. Drei fertige
Fälle (direct,planning,deadline) bleiben erhalten. reader.json ist wegen des
Replacefehlers veraltet/running. reader.json.tmp enthält failed; unverändert
als failure.json gesichert. Alte Live-Angaben unten gelten nicht mehr.
Keine eigenen großen Modelljobs laufen aktuell. NICHT blind denselben Run
überschreiben oder alle drei fertigen Fälle erneut ausführen.

Fix vorbereitet: research/progress_json.py mit bounded PermissionError-Retry
(20 Versuche,50ms) und Erhalt der .tmp-Datei bei dauerhaftem Fehler.
run_dialogue_capsules.py nutzt ihn; reine Forward-Heartbeats tolerieren nach
Retry weiterhin gesperrte Fortschrittsdateien und zählen die Ausfälle.
Zwei gezielte unittest-Tests bestehen (temporäre Sperre/bleibende Sperre).
Noch kein neuer integrierter Modelllauf mit diesem Fix. Nächste Arbeit:
Fortsetzung in neuem Run mit verifizierter Übernahme der drei fertigen
Receipts/Logits und der vorhandenen BF16-Kapsel implementieren; tatsächliche
Failure-Datei als terminale Quelle verwenden. Offene Fälle ab buffer ausführen.

Reader-Promptbau neu: research/reader_prompt.py, drei leichte Tests bestanden
(keine Target-/Assessment-Leaks, Assistant+EOS-Maske, keine verdeckte Evidenz).
Echter lokaler Qwen3B-Tokenizer ohne Modell geprüft: Session80758 Exit0,
544 Datensätze, maximale Länge154 Tokens, alle maskierten Antworten dekodieren
exakt zur Sollantwort. Kein Readertraining durchgeführt.

Readerdaten erweitert: aktuell360 train/92 dev/92 test. Hinzugekommen sind
followup_lookup/followup_buffer mit zahlenfreier History und stale_followup
mit altem Wert aus demselben Split. Aktuelle Evidenz bestimmt Sollantwort;
kein generationsübergreifender Zahlenleak durch History in andere Splits.
Drei unittest-Datenprüfungen bestanden in0,017s, einschließlich explizitem
52aktuell/56alt->52Tage-Beispiel. Synthetischer Dialog, keine echten Receipts
und noch kein Modelltraining. Ältere264/68/68-Angaben unten sind Vorversion.

Readerdaten vorbereitet in research/reader_training_data.py:264 train/68 dev/
68 test, disjunkte Supplier/Parts/Grundwerte und splitabhängige Formulierungen.
Deutsch/Englisch; Lookup,+4Puffer, Tages-/Wochenfristen short/equal/long,
fehlendeEvidenz mit UNKNOWN und identischer positiver Gegenfrage. Keine
Müllerfälle. Doppelte Missing-Eingaben proGrundwert entfernt. Zwei leichte
unittest-Prüfungen bestehen (Disjunktheit, eindeutigeInputs, Gegenpaare,
literal vorgegebene Sollantworten/Einheiten). Noch nicht als Trainingsrun
eingefroren und nicht trainiert/evaluiert. Kein Dialoghistory-Training in
diesem ersten Datensatz (history=[]); breitere Daten weiterhin erforderlich.

Reader-Trainingsprimitive nachgeprüft: Output/Logits/Loss werden nun vor dem
nächsten Microbatch explizit freigegeben. Neuer weakref-Lebensdauertest plus
bisherige Autogradtests bestanden:4/4, Session82480 Exit0,9,64s.
Kein Training des großen Readers; keine Änderung am laufenden BF16-Runner.
Session95788 weiter aktiv, cached_decode/buffer,134 Forwards,1850,61s.

Neuer Fortschritt für den geplanten Readertrainer:
research/reader_training.py enthält accumulation_windows und
token_normalized_backward (single process, Assistantmasken, verschobene
Next-Token-Labels, Tokenzahl des tatsächlichen Fensters als Nenner).
Kein kompletter Trainer und nicht in den laufenden Modellversuch eingebaut.
Zwei leichte Fenster-unittests bestanden. Drei echte Autogradtests mit kleinem
FP64-ToyDecoder bestanden, Session62564 Exit0,11,99s: Gradienten und Loss
stimmen mit Gesamtbatch bei ungleichen Antwortlängen und kürzerem Fenster
überein; leere Supervision wird abgewiesen. DDP/Mixed-Precision-Training/
Checkpointing und tatsächliches Qwen-LoRA-Training dadurch nicht validiert.
Großer BF16-Lauf weiterhin Session95788/PID29648, zuletzt cached_decode/buffer,
125 Forwards,1752,83s. Kein Neustart; drei Fälle vollständig abgeschlossen.

Neueste Nachrichten: Nutzer hat die bubblewrap/socat-Meldung ausdrücklich zum
Ignorieren freigegeben; keine Installation/WSL-Einrichtung durchführen.
Anschließend Unsloth-LoRA-Guide als Arbeitsgrundlage geliefert. Abgleich und
Präzisierungen in research/LORA-TRAINING-REVIEW.md, noch untrainierter
Reader-Kandidat in research/reader-lora-candidate.json (JSON validiert).
Rank16/Alpha32, alle7 Attention/MLP-Module,lr2e-4,2Epochen,effektiveBatch16,
Assistant-only, tokennormalisierte Akkumulation vorgesehen; kein neuer Trainer
oder Modelltraining dadurch bereits durchgeführt. Bestehende Adapter unverändert.

Session95788 zuletzt direkt gepollt: dritter Fall deadline abgeschlossen:
`No, because the delivery lead time is 27 days, which is longer than 20 days.`
Kapsel-/Referenzgleichheit und Commit im Runner bestanden. Gesamtjob läuft weiter.

**Aktiv: Session95788, PID29648**, `runs/dialogue-capsules-bf16-001`.
Befehl: `python -X utf8 research/run_dialogue_capsules.py reader --reader-input plain --reader-precision bf16 --copy-from runs/dialogue-capsules-plain-001 --run runs/dialogue-capsules-bf16-001`.
Session direkt gepollt, Modellgewichte werden geladen; reader.json meldet
running/load. Nicht neu starten. Dieser Live-Abschnitt hat Vorrang vor allen
älteren Statusangaben unten. Keine anderen eigenen Modelljobs parallel starten.

Neuere direkte Beobachtung derselben Session: running/cached_decode,
Fall deadline,63 abgeschlossene Forwards,974,72s. BF16-Kapsel bereits kompiliert:
cache:c85b9da46e4b135d310313a9413e671e95b076b65cdee1484ea75dc000ea4ddc,
gebunden an aktuelle27-Tage-Generation. Erster vollständiger Fall direct:
`27 days`, fresh_equal=true (ganze erste Logits und Tokenfolge), gemeinsam
committet als answer:b52dcd5daed4d507d224ae0359a705376df70e37fd22b9fd5704a823b572567e.
Zweiter vollständiger Fall planning: `Yes, 27 days. The delivery lead time for
component X12 from Mueller GmbH is 27 days.`, fresh_equal=true, committet.
Beide bisherigen Antworten inhaltlich korrekt (Yes bei planning überflüssig).
Alle weiteren Fälle noch offen; abschließender Audit erst nach terminalem Ende.
36 gesicherte Quelldateien und Manifest nativ hashgeprüft. Abschließender
audit_dialogue_capsules.py prüft nun ebenfalls Snapshotdateien samt Manifest
(optional für ältere Runs); py_compile bestanden, Audit noch nicht ausgeführt.
Diese Auditänderung entstand nach dem Snapshot, Runnerquellen unverändert.

Parallel vorbereitet: research/reader_deadline_cases.py liefert12 kontrollierte
Diagnosefälle (Deutsch/Englisch x Tage/Wochen x Frist21/27/28 bei Fakt27).
Enthält den originalen deutschen Fehlerfall; ausdrücklich Diagnose, kein
unabhängiger Holdout. History nur Identität, Sollwerte separat unter assessment.
Zwei leichte unittest-Prüfungen bestanden (0,003s), ohne Torch/Modellimports:
python -m unittest discover -s tests -p test_reader_deadline_cases.py -v.
Noch keine Modellinferenz auf diesen12 Fällen.
Diagnoserunner nun vorbereitet: probe_reader_precision.py --diagnostic-deadlines
--output runs/reader-deadline-diagnostic-001. Verwendet denselben BF16-Reader,
Fakt27, Präfix/Suffix und max64; speichert die12 Fälle vor dem Modellstart,
Quellsnapshot und Hashmanifest. Nur reader_input geht in den Prompt.
audit_reader_precision.py erkennt diesen Modus und prüft12 statt7 Sequenzen;
kein vorgetäuschter Int8-Paarvergleich. Beide Syntaxprüfungen bestanden,
neuer Modus noch nicht real ausgeführt/auditiert. Erst nach Ende von Session95788
starten, um konkurrierende BF16-Modelle im knappen RAM zu vermeiden.
Diese neuen Dateien sind nicht Teil des laufenden Integrationssnapshots.

Zusätzlicher leichter Provenienztest bestanden: tests/test_provenance_fanout.py,
unittest1/1 in0,015s. Fünf agentartige Textartefakte aus drei Originalquellen,
davon drei aus gleichem Artikel; Aggregation->Pod->Cache->Antwort hat genau
drei Roots. Artikelwiderruf sperrt exakt seine Ableitungsclosure, andere zwei
Aussagen bleiben zulässig. Herkunft bleibt für Audit erhalten. Kein Modelltest
oder Beweis statistischer Quellenunabhängigkeit; kein Core-Code verändert.

Der aktuelle Runner sichert vor Modellstart research/*.py und neural_pods/*.py
unter source_snapshot/reader samt SHA-Manifest. reader.json enthält dessen Hash,
PID, Speicherwerte, aktuelle Phase und Anzahl abgeschlossener Modellforwards.
Instrumentation nach letzter Gesamtsuite ergänzt; py_compile bestanden.
Die 137er-Suite war vor dieser reinen Fortschritts-/Snapshotinstrumentierung.

Dieser Lauf übernimmt die echten gespeicherten Planer-/Linkvorhersagen und
deren lokale Herkunft aus dem abgeschlossenen Plain-Lauf. Er trainiert oder
berechnet die Links nicht erneut. Er erzeugt echte BF16-Kapseln, lädt sie im
gemeinsamen Routing-/Linkpfad, vergleicht jede Ausgabe mit frischer Referenz
und committet mit der gesamten lokalen Lineage. Sieben Readerfragen plus
zwei validierte UNKNOWN. Bekannter deutscher Fehler bleibt im Datensatz.
Nach regulärem Exit0: research/audit_dialogue_capsules.py auf diesem Run,
Inhaltsbewertung separat, keine Ableitung des Gesamtziels aus grünen Audits.

## Abgeschlossene BF16-Kontrolle und Regressionstests

Dieser Abschnitt ersetzt die historischen Live-Angaben weiter unten.
Session32388/PID16092 ist regulär mit Exit0 abgeschlossen: 2598,17 Sekunden,
Gewichte unverändert. NICHT wieder starten. Kontrollaudit58472 Exit0:
sieben exakte Eingabesequenzen, Token/Text/EOS/Logitkonsistenz, Runner-Snapshot
und Dateihashes verifiziert. Inhaltliche Codex-Sichtung in
runs/reader-bf16-control-001/content-review.json: **6/7**, gegenüber Int8 **3/7**
auf denselben sieben Readerfragen. Frühere Int8 5/9 enthalten zwei UNKNOWN,
die im BF16-Kontrolllauf nicht wiederholt wurden.

Ein Fehler bleibt: german_followup antwortet `Yes, 27 days...` auf die Frage,
ob drei Wochen reichen. Erwartet Nein, da 27>21. Richtiger Zahlenwert genügt
nicht. Alle sechs anderen Fälle korrekt; häufig überflüssiges Yes.
followup_buffer enthält bereits27 in der History, daher kein isolierter
Pod-Kausalnachweis. Vollständiger Bericht: research/READER-PRECISION.md.

Regressionstests nach Ende des Modells: Session37728 Exit0, **137 passed**,
eine bekannte Torch-Quantisierungswarnung, 30,26s,
runs/bf16-variants-tests.xml. Einschließlich neuem echten kleinen
FP32/BF16-Kapselvariantentest: modellabhängige Auswahl und selektiver Widerruf
bestanden. Keine aktuellen eigenen Modell-/Audit-/Testjobs laufen.

Noch offen: tatsächlicher integrierter BF16-Kapselrun; Runner und Audit sind
vorbereitet (Befehl unten). Vor Ausführung aktuelle Quellen sichern. Der
Kontrolllauf beweist weder gespeicherte BF16-Kapselintegration noch den
vollständigen Forschungsauftrag. Deutscher Entscheidungsfehler muss zusätzlich
behoben und auf getrennten Daten geprüft werden; BF16 allein reicht nicht.

## Historische Zwischenstände des nun abgeschlossenen Kontrolllaufs

Aktiver Prozess am 15.09.2026: Session **32388**, Python-PID **16092**,
`research/probe_reader_precision.py --output runs/reader-bf16-control-001`.
Zuletzt Prozess direkt bestätigt und dieselbe Session gepollt: laufend,
99 abgeschlossene Modellforwards, Fall `followup_buffer`, rund 1489 Sekunden.
Nicht wegen langsamer Beobachtung neu starten. Dieser Abschnitt hat Vorrang
vor den älteren Terminalangaben unten.

Nun vier abgeschlossene Antworten: direct `27 days`; planning
`Yes, 27 days. The delivery lead time for component X12 from Mueller GmbH is 27 days.`
deadline `No, because the delivery lead time is 27 days, which is longer than 20 days.`
buffer: 29 Tage, korrekte Addition 27+2 und korrekte Identität, EOS;
587,80 Sekunden. Tatsächlicher vollständiger Text im report.json.
Inhaltliche Zwischensichtung in research/READER-PRECISION.md: vier korrekt,
noch kein vollständiger Qualitäts-/Evidenzaudit.
Sieben positive Fragen insgesamt, gleiche segmentierte Eingaben und max64
wie beim Plain-Int8-Reader. Nur BF16-Reader geladen; frische KV-Zustände.
Kein vollständiger Lifecycle-/Routinglauf und noch kein Gesamtergebnis.
Starker Speicherdruck, etwa 7,4 GB private Prozessbytes und stark schwankender
physischer Speicher. Keine fremden Nutzerprozesse beenden.

Nach bestätigtem Prozessabschluss:
`python -X utf8 research/audit_reader_precision.py runs/reader-bf16-control-001`.
Audit wurde um Runner-Snapshothash, Tokenlimit und Evidenzdateihashes ergänzt,
ist noch nicht ausgeführt. Danach tatsächliche Antworten nach derselben
inhaltlichen Rubrik wie Plain-Int8 bewerten und Vergleich dokumentieren.
Gewichts-/Token-/Eingabekonsistenz beweist keine Antwortqualität.

Turbopuffer mit Nutzer besprochen: mögliche austauschbare Kandidatensuche;
keine Migration beauftragt/ausgeführt. Registry bleibt Lifecycle-Autorität.
Ein Suchbackendwechsel behebt nicht den aktuellen Reader-/Speicherengpass.

Vorbereitet, noch NICHT mit Modell ausgeführt: run_dialogue_capsules.py hat
`--reader-precision bf16` (Default bleibt int8). BF16 erzeugt pro Generation
eine neue modellhashgebundene Kapsel und bindet sie als Variante; gemeinsamer
prepare/commit plus frische Referenz bleiben enthalten. Audit prüft zusätzlich
direkte Generationsherkunft, Modellhash und BF16-Kompilierungszuordnung.
py_compile für Runner und beide Audits bestanden, noch keine Regressionstests
oder BF16-Integration durchgeführt. Erst nach Ende des aktiven Readerprozesses:
Kontrollaudit und inhaltliche Sichtung, dann angemessene Regressionstests.
Neu in tests/test_linked_capsule.py, noch nicht ausgeführt:
test_precision_variants_select_own_state_and_revoke_independently verwendet
zwei echte kleine Qwen-Modelle (FP32/BF16), gemeinsamen Retrievalindex und
getrennte Kapseln. Prüft modellabhängige Auswahl und selektiven Widerruf:
gesperrtes BF16 darf nicht auf FP32 ausweichen, FP32 bleibt verwendbar.
Möglicher Folgelauf (neues Verzeichnis, vorher Existenz prüfen):
`python -X utf8 research/run_dialogue_capsules.py reader --reader-input plain --reader-precision bf16 --copy-from runs/dialogue-capsules-plain-001 --run runs/dialogue-capsules-bf16-001`.
Wegen BF16-Speicherdruck kann auch dieser Lauf sehr langsam sein; nicht parallel
zum aktuellen Modell starten. Neue Runnerquellen vor dem Lauf sichern.

## Vorheriger abgeschlossener Integrationslauf

Diese Runde hat konkrete Fortschritte erzielt. Gesamtziel aktiv. Eigene Jobs
terminal: Links78920, JSON-Reader56638, JSON-Audit66739, Plain-Reader32298,
Plain-Audit49003, Tests42433 alle Exit0. Keine Wiederstarts dieser Runs.

- research/adopt_lineage.py: atomare, hashgeprüfte Übernahme einer lokalen
  Quell-DAG-Snapshotkette. ACL/Revocation/Head-Konflikte geprüft; keine spätere
  Synchronisation mit Quellregister. Destination ist lokale Versuchsautorität.
- Sieben Importtests plus neuer UNKNOWN-Snapshottest. Gesamtsuite136 Tests
  in22,18s bestanden, runs/dialogue-capsules-tests.xml.
- DialogueAccess hat jetzt PlannerDeferred(InvalidState) für echte UNKNOWN-
  Entscheidungen samt konsumiertem Snapshot/Plannerinput. Bestehende Caller
  können weiter InvalidState fangen; neuer Runner persistiert UNKNOWN-Lineage.
- run_dialogue_capsules.py links: echter trainierter Planner, echte Link-LoRA,
  übernommene Plannerherkunft, synthetischer Request-Origin, Planerproof und
  Linkproof in gemeinsamem Register. Sieben ADDRESS2/LINK1CLUSTER1 und zwei
  tatsächliche UNKNOWN. Keine erzwungenen Antworten/Adressen.
- reader: sieben echte Qwen3B-Int8-Kapselantworten auf ursprüngliche Dialogfragen,
  zwei UNKNOWN ohne Readerinferenz. max64Tokens. Gemeinsame Receipts enthalten
  Planer-/Input-/Link-/Router-/Generations-/Kapselherkunft. Basis unverändert.
- runs/dialogue-capsules-001 (JSON-Dialog): technischer Audit112 Knotenhashes,
  sieben Logitpaare und sechs Clonekontrollen bestanden. Inhaltlich nur3/9
  laut begründeter Codex-Sichtung: followup_buffer, missing_part,unsupported_metric.
  Andere Antworten verändern X12 zuX11/X1, widersprechen Fakten oder antworten
  nicht. buffer ohneEOS bei64Tokens. Richtiges Zahlvorkommen allein zählt nicht.
- --reader-input plain --copy-from runs/dialogue-capsules-001 auf neuem Run
  runs/dialogue-capsules-plain-001: gleiche Links/Fragen/Modelle/Kapseln/Tokenlimit,
  nur Dialogdarstellung lesbar. Technischer Audit119 Knoten,7Paare,6Kontrollen.
  Inhaltlich5/9: direct,followup_buffer,followup_reference und beideUNKNOWN.
  Planung behauptet Ankunft zumBestelldatum,deadline falsche Begründung,
  bufferX1 stattX12, deutscheAntwortNo+3weeks statt27Tage. Alle siebenEOS.
  Reader229,62s. Inhaltliche Bewertungen mit Reporthash in content-review.json;
  Codex-Sichtungen, keine unabhängigen Menschen/automatischen Qualitätsaudits.
- Audits in audit_dialogue_capsules.py prüfen auch lokale Widerrufe: Training,
  einzelne Anfrage, Link,Fakt sperren Commit; Anfrage selektiv, Planerwiderruf
  erhält unabhängige Payloads. Import ist KEIN verteiltes Revokationssystem.
- DIALOGUE-CAPSULES.md, INTEGRATION-PLAN.md ergänzt, Quellsnapshots in beidenRuns.

Nächste wichtige Diagnose: dieselben Readerfragen mit BF16-Modell ohne Int8
vergleichen. Noch nicht bewiesen, dass Modellfähigkeit statt Quantisierung die
Fehler verursacht. Vorige BF16/Float32Runs wurden unter RAM/Pagingdruck vor
brauchbarer vollständiger Bewertung gestoppt. Keine pauschale Unmöglichkeit
ableiten. Möglichst NUR Reader laden, keinen MiniLM/Planner zusätzlich; exakt
gleiche segmentierte Präfix-/Suffixdarstellung aus plainRun verwenden. Erst
wenige vordefinierte Fälle/Counterfactuals, aktuellen Prozess/RAM/Phase messen;
nicht allein wegen früherem privaten Speicherwert stoppen. Eigene Prozesse
immer anhandIdentität prüfen, Benutzerprozesse unangetastet lassen.

Persistente produktive Dialogreceipts, normale laufende App, großeKataloge,
MehrHop, Originalarchiv, 100k-no-gradient/ABI/Speedup/Neuheits-DoD bleiben offen.
Planer50er Test weiterhin49/50; neue Datensätze für weitere Generalisationsclaims.

# Historischer Stand: Planner trainiert, auditiert und neu geladen

Diese Runde hat konkrete Fortschritte erzielt. Gesamtziel bleibt aktiv.
Keine eigenen laufenden Modell-/Testprozesse: Training42170, Replay40942,
Tests62468 alle Exit0. Alle weiter unten genannten laufenden Zustände historisch.

- runs/planner-training-003 abgeschlossen: 300 Schritte, 540.672 trainierbare
  LoRA-Parameter, Basis unverändert. 1.053,03 s inkl. Baseline und Testinferenz.
- Freie Ausgabe vor/nach Training: 0/50 ->49/50. Separater
  audit_planner_training.py bestanden: echte Tokens/EOS/Ziele, Trennung,
  Dateihashes/Registry. Strenges Generalisierungsgate bleibt false.
- Einziger Fehler test:age:3: "How many years has Lurena Supply been operating?"
  erzeugt ADDRESS2 statt UNKNOWN (Lieferzeitadresse). Gültige Syntax, echter
  Intentfehler. Alle anderen neun Familien jeweils5/5 richtig.
- Finaler Adapter runs/planner-training-003/adapter/dialogue_planner,
  ArtifactKey lora:69e7ce869d9fb52dfd7fb790e92ff710398ef25b564f94fd37a414d5d4625a82.
  Herkunft im training-run/registry.sqlite3 über capability:dialogue-address-planner
  und Trainings-/Modellursprünge. Modellbasehash in protocol/payload61d18c9f...
- Neu research/verify_planner_adapter.py lädt nach Registry-/Datei-/Basisprüfung
  tatsächlich neu. runs/planner-reload-001: alle50 Tokenfolgen exakt reproduziert,
  dann9/9 richtige Adressentscheidungen auf den alten Dialogfällen hinter
  DialogueAccess. Siebenmal ADDRESS2, X99/Preis tatsächlich UNKNOWN.
  Keine bloß vom Guard kaschierten falschen Modelleingaben. Basis unverändert.
  Replay84,71s, Quellsnapshot nach Terminal unverändert gesichert.
- Gesamtsuite128 Tests in23,83s bestanden, runs/planner-training-tests.xml.
- PLANNER-TRAINING.md, DIALOGUE-ACCESS.md, INTEGRATION-PLAN.md aktualisiert.

Nächste Integration: echte Planner-LoRA -> aktuelle stabile Wissensadresse ->
echte Link-LoRA -> aktuelle Kapsel -> tatsächliche Antwort/Operationen, mit
gemeinsamer Snapshot-/Commitgrenze und vollständiger Planner-/Dialogherkunft.
Aktueller Replay prüft Adressen, keine fachlichen Antworten. Planerherkunft liegt
noch im separaten Trainingsregister; nicht einfach externe IDs ohne Lineage in
Faktenreceipts einfügen. Isolierter Versuch darf verifizierten Trainings-DAG
kopieren, muss dann klare lokale Autorität haben; das wäre keine verteilte
Widerrufssynchronisation. Rückfragen/UNKNOWN dürfen keinen falschen Fakt nutzen.

Die neun alten Dialogfälle sind Regression, nicht neue Validierungsstichprobe.
Neuer50er Prüfsatz ist jetzt ebenfalls bekannt. Breite implizite Nutzung,
persistente Dialogreceipts, neue Wissensarten, große Kataloge, Mehr-Hop, noch
ein Readerfehler (27,5/6), zwei Operatorfehler (22/24), Originalarchiv und
ursprüngliche Forschungs-DoD bleiben offen. Keine Gesamtziel-Abschlussbehauptung.

# Historischer Stand: echtes Planner-LoRA-Training läuft (003)

Diese Runde: konkreter Fortschritt und zuletzt verifiziert laufender Prozess.
Gesamtziel bleibt aktiv. NICHT neu starten: Session **42170** gehört zu
`research/train_dialogue_planner.py --output runs/planner-training-003`.
Letzter bestätigter Stand: Schritt 64/300, elapsed 258,68 s, läuft weiter.
Mit write_stdin auf dieselbe Session warten; bei Handleproblem erst Prozesse
und report.json prüfen. Datei allein ist kein Beleg für lebenden Prozess.

## Neuer Code und Daten

- research/planner_training_data.py: 100 eindeutige Trainings- und 50 eindeutige
  Prüfeingaben. Getrennte Firmennamen/Teile/Formulierungen. Kataloggrößen 2/3,
  gleiche Zielhäufigkeit pro Position innerhalb jeder Kataloggröße.
- UNKNOWN 40 %, Adressen 1/2/3 train 24/24/12, test 12/12/6. Adresse 3 ist
  seltener, weil Zweierkataloge sie nicht besitzen. Keine Faktenwerte.
- Unbekannte Firmen/Teile haben positive Gegenproben mit derselben Frageform.
  Acht Grundfamilien plus zwei Gegenprobenfamilien, zunächst nur Englisch.
- tests/test_planner_training_data.py besteht, prüft Trennung, Gegenproben,
  Positionsbalance und Ausschluss der neun alten Regressionstestfragen.
- train_dialogue_planner.py: echte Qwen0.5B-PEFT-LoRA q/v, Rank8, lr0.0003,
  3 fixe Epochen =300 Schritte, assistant-only loss, 4 CPU-Threads. Basismodell
  per PodModel.frozen_hash geprüft. Freie Baseline vor Training, Nachherprüfung
  erst nach allen Epochen, keine checkpoint selection anhand Testdaten.
- 003 Baseline 0/50. Training bei letztem Poll64; keine Trained-Qualität behaupten.
- dataset.json und protocol.json samt Hash vor Run gespeichert, Quellsnapshot
  aus sieben Dateien während Baseline. Adaptercheckpoint je Epochengrenze;
  finaler Adapter wird mit Trainings-/Modellherkunft in eigener Registry erfasst.
- audit_planner_training.py vorbereitet, NOCH NICHT ausgeführt: sobald 003
  completed und Prozess terminal, Audit auf denselben Run ausführen.
  Prüft gespeicherte Tokens/EOS/Ziele/Splits/Artefakte, kein unabhängiges Training.
- PLANNER-TRAINING.md enthält Versuch und Grenzen. Volltests nach Abschluss
  des CPU-Trainings laufen lassen; zuletzt127 vor neuem Datensatztest.

## Erhaltene abgebrochene Piloten

- 001: Session27078 Exit1, eigene Prozesse33564/46140 nach exakter
  Kommandozeilenprüfung beendet; 28 Basisantworten,0 Trainingsschritte.
  Unbekannte Firmennamen enthielten unerwünschten Hinweis "Unknown".
- 002: Session53181 Exit1, eigene Prozesse35892/32088 geprüft/beendet;
  32 Basisantworten,50 Schritte. Datencheck zeigte Positionsbias/fehlende
  ADDRESS3-Testziele und fehlende Verfügbarkeitsgegenproben. Kein fertiger Adapter.
- Beide report.json explizit status stopped mit Grund, Daten bleiben erhalten.
  Kein Löschen/Resume daraus. 003 startet erneut von unveränderter Modellbasis.

## Nächste Schritte nach Abschluss

1. Terminalstatus und report prüfen, dann audit_planner_training.py RUN.
2. Tatsächliche Vorher-/Nachherwerte nach Aufgabenfamilie beurteilen; positives
   Generalisationsgate nur bei allen50 korrekt. Auch dann keine internen
   Wissensfähigkeiten/Antwortqualität/Produktionsreife daraus ableiten.
3. Adapter-Neuladen und neun bestehende Dialogregressionen hinter DialogueAccess
   als separaten Integrationsversuch prüfen; erst danach sinnvoll Kapselpfad.
4. Deutsche Dialoge, größere Kataloge, zusätzliche Wissensarten, Dialog-Lineage,
   Antwort-/Operatorfehler, Originalarchiv und breite DoD bleiben offen.

# Historischer Stand: Dialogzugriff implementiert, freie Planer scheitern

Konkreter Fortschritt dieser Runde; Gesamtziel aktiv. User-Anforderung weiterhin
internes Wissensverhalten, nicht bloße Lookupabnahme.

- Neu research/dialogue_access.py: wertfreier aktueller Katalog nach Registry-
  ACL/Generation/Lineage-Gates. Modell schlägt ADDRESS n oder UNKNOWN vor.
  Exakte Syntax/Kataloggrenzen/Teilecodes/bekannte Lieferantennamen geprüft,
  vorherige USER-Beiträge können Referenzen liefern. Kanonische Lookupfrage
  geht durch IdentityDragonfly; ursprüngliche Frage/History bleiben erhalten.
- Snapshot hält ALLE Katalogabhängigkeiten, weil Modell alle Metadaten sah.
  Noch keine Plannerproof-/Dialogreceipt-Persistenz, keine Antwortgeneration,
  kein vollständiger Link-LoRA/Kapselpfad. Soft-Intent kann weiterhin falsch sein.
- tests/test_dialogue_access.py sieben Fälle bestanden. Gesamtsuite 127 Tests
  in 24,21 s bestanden, Session 96481 Exit 0; XML runs/dialogue-access-tests.xml.
- run_dialogue_access.py reale 0.5B-Float32 oder --reader-3b Int8 mit
  unverändertem Neun-Fälle-Dialogvertrag. Optional --constrained erlaubt nur
  Tokenpfade gültiger Adressen oder UNKNOWN, keine richtige Zieladresse erzwungen.
- dialogue-access-001: 0/9 Modellentscheidungen korrekt, 0/9 Syntax, 2/9 nach
  Zurückweisung. Session 35686 Exit 0. Basis unverändert.
- dialogue-access-3b-001: 0/9 Modellentscheidungen, 1/9 nach Guards, 1/9 Syntax.
  Preisfrage erzeugt ADDRESS 2 (falsch: Lieferzeit). Session 33155 Exit 0.
  Quellsnapshot während Lauf gesichert. Kein zuverlässiger Intent-Compiler.
- dialogue-access-constrained-001: 9/9 gültige Syntax, überall UNKNOWN,
  daher 2/9 Modell/Guard korrekt, sieben positive Fälle scheitern.
  Session 70087 Exit 0. Kein interner Wissenszugriff dadurch belegt.
- audit_dialogue_access.py mit Rusttokenizer prüft echte Token->Text/EOS und
  getrennte Modell-/Guardmetriken. Alle drei Audits erfolgreich; bedeutet
  Konsistenz der Fehlresultate, nicht bestandene wissenschaftliche Aufgabe.
- DIALOGUE-ACCESS.md und INTEGRATION-PLAN.md ergänzt.

Nächste sinnvolle Arbeit: echtes getrenntes Training einer Planungs-LoRA auf
wechselnden wertfreien Katalogen, Dialogreferenzen und expliziten Negativfällen.
Keine weiteren Promptvarianten auf den neun bekannten Testfällen als neue
Generalisation ausgeben. Trainings-/Testnamen, Teile, Templates und Katalog-
reihenfolgen sauber trennen; neue versiegelte Evaluation. Vor Integration
semantische Ablehnung von Preis/Alter/unbekannten Firmen/Teilen sicher messen.
Gemeinsame Kapselantwort zuletzt weiter 5/6, Operatorset weiter 22/24.
Originalarchiv/Forschungs-DoD/normaler Dialogpfad bleiben offen; keine Blockade,
solange solche unabhängige Arbeit möglich ist.

# Historischer Stand: neue Kapselantwort real getestet, 5/6

Diese Runde hat konkrete Fortschritte erzielt, Gesamtziel aktiv. Keine eigenen
Modelljobs mehr laufend: Linksession 46159, Readersession 24147, Audit 17781
alle Exit 0. Testsession 14937 ebenfalls Exit 0: 17 relevante Integrationstests
in 11,35 s bestanden, runs/identity-linked-tests.xml. Gesamtsuite zuletzt 120.

- run_linked_model.py akzeptiert jetzt abgeschlossene identity-transfer-Quellen
  mit --model-manifest des Originalmodelllaufs und lädt IdentityDragonfly.
- runs/identity-linked-001 ist die neue Integration aus identity-transfer-002.
  Linkstufe 6/6 echte neue Qwen-Ausgaben korrekt, anschließend echter 3B-Reader.
- Neue 27-Kapsel: 5/6 richtige Antworten. Deutsche Frage
  "Wie lange braucht Muller GmbH fuer X12?" gibt "Fact: Muller GmbH has a
  delivery lead time of" aus, ohne EOS innerhalb 12 Tokens. Lookup-Gate false.
  Textreferenz ist token-/logitidentisch: kein Unterschied durch Cacheladen.
- Vier Lifecyclekontrollen bestanden, Gewichte unverändert. Reader 154,37 s,
  Median 1,132 s. Einzelmessung, kein Speedup-/Servingnachweis.
- audit_linked_model.py bestätigt 74 Knotenhashes, sechs Receipts/Dateihashes,
  sechs vollständige Logitpaare, zählt 5/6. Quellsnapshot aus 18 Dateien im Run.
- Neues internal-knowledge-cases.json: neun nicht trainierte Dialogfälle für
  direkte/indirekte Fragen, Operationen, Folgefragen und unbekanntes Wissen.
- probe_access_contract.py ausgeführt: Parser-only 3/9 Zielentscheidungen,
  davon eine richtige direkte Auflösung und zwei korrekte Zurückweisungen.
  Sechs indirekte/kontextabhängige Zugriffe scheitern vor der Modellausführung.
  Keine Antwortqualitätsmessung/ACL- oder ANN-Prüfung in dieser Diagnose.
- IDENTITY-TRANSFER.md und INTEGRATION-PLAN.md mit tatsächlichem Ergebnis ergänzt.

Nächste funktionale Arbeit: Dialog-/Intentauflösung, die implizite Zugriffe und
Referenzen auf gültige aktuelle Wissensadressen ermöglicht, ohne die harte
ACL-/Quellen-/Entitäts-/Generationsprüfung aufzuweichen. Die neuen Fälle nicht
einfach durch harte Frage->Antwort-Tabellen grün machen. Tatsächlichen Modellpfad
prüfen; direkte Fragen allein erfüllen Nutzerziel nicht. Der Readerfehler bei
27 und die früheren zwei Operatorfehler bleiben offen. Keine Verbesserung der
Outputqualität durch bloße Extraktion des Registry-Wertes als Modellleistung
ausgeben. Originalarchiv, breite DoD und Produktionspfad bleiben offen.

# Historischer Stand: Identitätsadressen über Wertänderung getestet

Diese Zielrunde hat Fortschritt erzielt; Gesamtziel aktiv. Nutzer ergänzt:
Das gesamte System muss sich am Ende wie internes Wissen anfühlen und verhalten,
auch wenn es extern in Pods liegt. INTEGRATION-PLAN.md enthält dazu explizite
Abnahmen: implizite Zugriffe, Dialogreferenzen, Anwendung/Komposition, Lifecycle,
Unsicherheit und vollständiger Nutzerpfad. Keine bloße Lookup-Abnahme daraus machen.

- Neu: research/identity_dragonfly.py, separate Forschungsvariante, kein
  Umetikettieren alter v2-Gewichte. Frisches Training aus wertfreiem Deskriptor
  und expliziten Beispielen; Herkunft an NeuralSymlinks-Identität gebunden.
- learned_dependencies enthält zusätzlich aktuelle Pod-/Vektorabhängigkeiten,
  auch für die geerbte advisory Aliasdiagnose. Aktuelle ACL/Version bleibt hart.
- Neun Tests: Wertwechsel/reload/exakte Repräsentation, Alias/Cluster/ACL,
  selektiver Widerruf ursprünglicher Identitätsquelle, neuer Faktenquelle,
  Trainingsquelle, Repräsentation sowie Hashmanipulation.
- Gesamtsuite 120 bestanden in 23,36 s, Session 50853 Exit 0,
  runs/identity-transfer-tests.xml.
- Echter Lauf runs/identity-transfer-002: zwei Identitätsadressen trainiert,
  sechs echte Qwen-Linkausgaben vor und sechs nach 18 -> 27 korrekt. Keine
  Trainingsaufrufe nach Update; RepresentationKey/LoRA/Score/reload gleich,
  alte Commits gesperrt, Qwen-Basis unverändert. Session 18811 Exit 0.
- Erstlauf 001 scheiterte wegen fehlender lifecycle-Metadaten im Runner-Update,
  Session 46110 Exit 1. Korrigiert; fehlgeschlagenen Run erhalten.
- Audit_identity_transfer.py bestanden: 61 DAG-Knoten, zwölf Textausgaben,
  Receipt-/Adapterherkunft und Dateien. Keine unabhängige Inferenz-/Tokenprüfung.
- Snapshot aus 14 Quelldateien im Run; IDENTITY-TRANSFER.md beschreibt Grenzen.

Nächster konkreter Integrationsschritt: echte Qwen3B-Kapsel mit neuem Wert 27
hinter IdentityDragonfly + tatsächlichem Link auflösen und Antwort prüfen.
Run 002 hat noch einen textuellen Test-Pod für 27; KEINE neue Antwort-LoRA oder
neue Kapselantwort behaupten. Run_linked_model.py verwendet bisher v2-Router
und fixe Ausgangsquelle; für Identitätstransfer sachgerecht erweitern oder
neuen Runner erstellen. Der neue Router ist noch nicht im normalen CLI aktiv.
Originalarchiv und breite Forschungs-DoD bleiben offen; meaningful work möglich.

# Historischer Stand: echte gestufte Integration abgeschlossen

Diese Zielrunde hat konkrete Fortschritte erzielt. Gesamtziel bleibt aktiv.
Keine eigenen laufenden Modellprozesse: Tests Session 89165, Reader 19517,
Audit 34281 alle Exit 0. Private Projektdateien bleiben unzugänglich; die
verfügbaren Tools enthalten keine ChatGPT-Projekt-/Browser-Sitzungsanbindung.
Keine erneute Einzelfragen-/Dateiliste vom Nutzer verlangen.

## Neue Nachweise

- runs/linked-model-001: echte gelernte Dragonfly-Auswahl und Qwen-0.5B-LoRA
  erzeugen 6/6 richtige Verweise; separates Qwen-3B-Int8 liest 6/6 mal 18.
- Beide Basisgewichte unverändert, Reader 223,25 s gesamt; Median Readerpfad
  1,453 s, Linkinferenz separat durchschnittlich 1,667 s. Kein Servingbenchmark.
- Geschwisterkapseln pro Generation/Modellsha erhalten originale LoRA-/Vektor-
  Artefakte. Deshalb bleibt der tatsächlich trainierte Dragonfly nutzbar.
- Persistierter LinkPrediction-Nachweis prüft Frage, tatsächlichen Text und
  Adapterherkunft; gemeinsamer Commit enthält Nachweis, Routing, LoRA, Kapsel.
- Vier Kontrollen bestanden: proof/source/update blockieren, proof selektiv.
- Separater audit_linked_model.py: 64 DAG-Knoten nachgehasht, sechs Antworten
  aus Tokens verifiziert, sechs vollständige Logitpaare gleich, Dateihashes
  und Receipt-Lineage geprüft. Keine unabhängige Inferenzwiederholung.
- 111 Tests bestanden in 42,15 s. Voller Code-Snapshot im Run vorhanden.
- research/LINKED-CAPSULE.md und INTEGRATION-PLAN.md aktualisiert.

## Nächster wesentlicher Schritt

Dragonfly v2 hängt weiterhin an Faktengeneration und exaktem Primärartefakt.
Die zusätzliche Geschwisterkapsel löst nur den Readerwechsel, NICHT den
Wertwechsel ohne neue Gradienten. Identity-basiertes Adresstraining muss an
separate, wertfreie Identitätsherkunft gebunden werden; alte tatsächlich
faktenabhängige v2-Repräsentationen dürfen nicht einfach umetikettiert werden.
NeuralSymlinks besitzt bereits separate stabile Identitäten; deren Semantik,
Alias-/Cluster-/ACL-Änderungen und Herkunft sind als Ausgangspunkt zu prüfen.
Danach echter Update-/Reload-/Widerrufstest ohne Router-/Linktraining je Wert.

Breiter bleiben 22/24 statt 24/24 Operatorqualität, neue versiegelte Fälle,
Mehr-Hop, 100k neue Fakten ohne Faktgradienten, CQP1/BCC1 Originaloperatoren,
RAG-/ABI-/Crash-/Tenant-/Neuheitsnachweise offen. Die sechs bestehenden
Lookupfragen belegen keine davon. Fehlendes Version-33-Archiv bleibt eine
Quellenlücke; keine Blockiert-Markierung solange unabhängige Arbeit möglich ist.

# Historischer Stand (durch obigen Stand ersetzt)

# Aktueller Stand: 3B-Lauf vollständig beendet und auditiert

**Vorherige Zielrunde: Fortschritt. Keine laufenden Modellprozesse dieser Runde.**
Session 26603 (`prefix-interface-3b-004`) endete mit Exit 0; Audit-Session 46181
ebenfalls. Frühere Sessions 70543 und 99276 wurden nach Identitätsprüfung bewusst
beendet; sie sind nicht mehr laufend. Alle weiter unten genannten Handles sind
historisch und dürfen nicht als aktuelle Arbeitsaufträge gelesen werden.

- `research/CPU-INT8.md`: vollständiger Bericht und Grenzen.
- 3B-Int8: 96 Kontrollantworten plus Updateantwort. 22/24 Text und Kapsel;
  24/24 komplette Logit-/Tokenpaare identisch, 6/6 Lifecycle, Gewichte unverändert.
- Fehler: Vost 18 plus 2 ergibt 30; Vost 42 >25 ergibt NO. Striktes Gate false.
- 107 Tests bestanden; anschließend Checkpoint-Sicherung im Runner ergänzt und
  mit dem vollständigen 3B-Lauf plus separatem Audit tatsächlich ausgeführt.
- 3b-003 enthält 60 echte Teilantworten, separat token-/EOS-geprüft. Keine
  vollständigen Rohlogits: kein voller Audit möglich. Der Stopp wurde auf
  Grundlage früherer Speichermessungen vorgenommen; die spätere brauchbare
  Inferenz war bereits weiter fortgeschritten. Diese Einordnung ist korrigiert.
- Physischer RAM 15,65 GiB. Private/virtuelle Bytes nicht mit residentem RAM
  verwechseln. Frische Gruppen-Checkpoints zeigen tatsächlich ca. 3 GiB RSS.

## Nächste konkrete Integration

1. Bestehenden erfolgreichen Lauf `runs/embedded-symlink-002` und
   `research/linked_capsule.py` prüfen. Der gemeinsame Integrationsvertrag ist
   getestet, der echte trainierte Verweis-/Dragonfly-/Kapselpfad noch nicht.
2. Echte Qwen-LoRA-Verweise und gelernte Dragonfly-Adressen mit aktuellen
   Kapseln in einer isolierten Registerkopie verbinden; alle Abhängigkeiten
   einschließlich des ursprünglichen Routingstands bis zum Commit erhalten.
3. Wegen RAM möglichst keine zwei großen Modelle gleichzeitig laden. Getrennte
   echte Modellstufen sind für den Integrationsnachweis möglich, müssen aber
   samt Zwischenergebnis-Lineage transparent ausgewiesen werden und belegen
   keine Produktionslatenz.
4. Der bisherige Dragonfly verlangt neue Repräsentationen je Faktengeneration;
   Wiederverwendung ohne faktenspezifische Gradienten ist noch offen. Ebenso
   semantisches Gate, neue Namen/Werte/Operatoren, Mehr-Hop, Skalierung und starke
   Vergleichssysteme. Ziel nicht auf bestandene Teiltests reduzieren.

Eine neue asynchrone Textfrage nach direktem Download-Link/lokalem Pfad des
Version-33-Archivs wurde gestellt. Kein Zugriff auf private Projektdateien;
Freigabechats und 44 Quellfragmente sind vorhanden. Weiter unabhängige Arbeit
leisten; das Gesamtziel bleibt aktiv.

# Historische Checkpoints (durch obigen Stand ersetzt)

Vorherige Runde: **Fortschritt**. 44 Chat-Quellfragmente zurückgewonnen, darunter
22 gültige Python-Dateien; Original-Prüflogik mit 15 Fällen evaluiert; ihre
Text-/Token-/Greedy-/EOS-Anforderungen im lokalen Auditor integriert und mit
beiden gespeicherten Modellläufen geprüft. Gesamtsuite 100/100.

## Aktualisierung: Download abgeschlossen, Modellversuch läuft

Qwen2.5-3B ist vollständig vorhanden: `models/qwen3b/download-manifest.json`
meldet `verified`, beide veröffentlichten Gewichts-SHA-256 wurden verglichen.
Downloadsession 11171 ist mit Exit 0 beendet (286,91 s).

Neue Integration: `research/linked_capsule.py` verbindet Routing, Verweis und
Kapsel mit einem gemeinsamen Snapshot einschließlich des ersten Routingstands.
Vier neue Integrationstests bestanden; Gesamtsuite **104/104 in 26,53 s**.
Details und Grenzen: `research/LINKED-CAPSULE.md`.

**Aktuell laufende Unified-exec-Session: `70543`.**

```text
python research/run_prefix_probe.py --model models/qwen3b --dtype bfloat16 --typed-format --output runs/prefix-interface-3b-002
```

Zuletzt bestätigte der Tool-Poll die weiter laufende Session. Gewichte geladen,
noch keine fertige Fragegruppe; Ergebnisstatus `running`. Nicht neu starten,
sondern dieselbe Session prüfen. Quellstand liegt im Lauf unter `source_snapshot`.

Der erste Float32-Versuch `prefix-interface-3b-001`, Session 26422, wurde nach
direkter Identitätsprüfung des eigenen Prozesses PID 26916 bewusst beendet:
aktuell **15,65 GiB physischer RAM**, 98,1 % belegt, nur 0,30 GiB verfügbar.
Exit 1 bestätigt. Ergebnisse ausdrücklich als `interrupted_resource_pressure`
gespeichert, keine vollständigen Modellantworten. Keine verlorene Session oder
bloße Beobachtungs-Zeitüberschreitung als Fehler interpretiert.

BF16 wurde ergänzt; tatsächliche Gewichtsbytes werden dtype-unabhängig gehasht.
Erste Logits werden für die Audits verlustfrei von BF16 nach FP32 erweitert.
Die 15 betroffenen Prefix-/Link-/Evidenztests bestehen nach dieser Änderung.
Der frühere Hinweis auf 32 GB in Gesprächszusammenfassungen ist für die aktuelle
Maschine falsch; physische Kapazität wurde frisch mit psutil gemessen.

## Historischer Downloadhandle (nicht mehr laufend)

- Skript: `research/download_qwen3b.py`
- Ziel: `models/qwen3b`
- Unified-exec-Session: `11171`
- Zuletzt beobachteter Hauptprozess: Windows PID `50040`, Start 15.09.2026 14:14:50.
- Letzter Tool-Poll bestätigte eine weiter laufende Session, keinen Abschluss.
- Diese Notiz ist **keine aktuelle Prozessbestätigung**. Bei Wiederaufnahme
  zuerst genau diese Session pollen beziehungsweise den Prozesszustand prüfen.
  Nicht wegen fehlender Fortschrittsausgabe einen zweiten Download starten.

## Nächste Arbeit

1. Session **70543** prüfen, denselben BF16-Lauf beobachten und Ergebnis nach
   tatsächlichem Abschluss auditieren. Speicher-/Fortschrittslage prüfen; keine
   weiteren schweren Modellprozesse parallel starten.
2. Ergebnisprüfung: `python research/audit_prefix_probe.py runs/prefix-interface-3b-002 --tokenizer models/qwen3b`.
   Referenzqualität und striktes Gate vollständig auswerten, beide 3B-Läufe
   einschließlich des Ressourcenabbruchs erhalten.
3. Modellrevision, Quellstand, Rohdaten und Ressourcen aufzeichnen. Das Ziel ist
   weiterhin die volle Zusammenführung, nicht nur eine bessere Präfix-Demo.
4. Anhand des [Integrationsplans](INTEGRATION-PLAN.md) fehlende Originalmodule
   beschaffen oder ausdrücklich als neue Rekonstruktionen entwickeln. Bestehende
   LoRA-Verweise, Semantik und Dragonfly müssen mit der ausgewählten
   Zustandsrepräsentation unter einer gemeinsamen Lifecycle-Grenze verbunden werden.

Der private ChatGPT-Projektlink führt hier zur Anmeldung. Freigabechats und
44 Fragmente sind zugänglich; vollständiges Version-33-Archiv, endgültige
FROZEN_DOD und Originalrohzustände fehlen weiterhin. Das blockiert nicht alle
lokalen Anschlussarbeiten und rechtfertigt derzeit keinen Zielstatus `blocked`.
