# Reader-Präzision: abgeschlossener Vergleich

Stand 15.09.2026. Lauf `runs/reader-bf16-control-001`, Session 32388,
PID 16092, regulär Exit0. **Technischer Audit bestanden; inhaltlich 6/7.**
2598,17 Sekunden Gesamtlaufzeit (43,30 Minuten), Modellgewichte unverändert.
Audit-Session 58472 ebenfalls Exit0. Der Lauf darf nicht neu gestartet werden.

## Fragestellung

Der vollständige Plain-Dialogpfad mit Qwen2.5-3B und dynamischer Int8-
Quantisierung erreicht in der bisherigen inhaltlichen Codex-Sichtung 5/9.
Dieser Kontrolllauf prüft dieselben sieben positiven Readerfragen mit BF16,
denselben segmentierten Token-Eingaben und demselben Greedy-Limit von 64 Tokens.
Die zwei UNKNOWN-Fälle werden nicht erneut ausgeführt. Es werden keine alten
Int8-KV-Zustände in das BF16-Modell geladen.

## Tatsächlich beobachtet

Inhaltliche Codex-Sichtung, kein unabhängiger menschlicher Qualitätsaudit:

| Fall | BF16-Ausgabe | Inhaltliche Bewertung |
|---|---|---|
| direct | `27 days` | Korrekte Dauer. |
| planning | `Yes, 27 days. The delivery lead time for component X12 from Mueller GmbH is 27 days.` | Richtige Dauer und Identität; überflüssiges Yes ist ein Stilproblem. |
| deadline | `No, because the delivery lead time is 27 days, which is longer than 20 days.` | Richtige Entscheidung und Begründung. |
| buffer | `The delivery lead time for Mueller GmbH X12 with a two-day buffer is 29 days. Yes, because adding 2 days to the original 27 days equals 29 days.` | Korrekte Addition und Identität, EOS erreicht; überflüssiges Yes. |
| followup_buffer | `Yes, because adding two extra days to the 27 days lead time results in 29 days.` | Korrekte Anwendung auf die Folgefrage ohne erneut genannte Identität; überflüssiges Yes, EOS erreicht. |
| followup_reference | `Yes, 27 days. The supplier Mueller GmbH has a delivery lead time of 27 days for component x12.` | Richtige Referenzauflösung und Dauer; History enthält keinen Zahlenwert. Überflüssiges Yes. |
| german_followup | `Yes, 27 days. The delivery lead time for component X12 from Müller GmbH (alias Mueller GmbH) is 27 days.` | Falsch: Auf die Frage, ob drei Wochen reichen, muss Nein folgen. Die richtige Zahl 27 behebt diese falsche Entscheidung nicht. |

Int8 war beim direkten Fall ebenfalls korrekt. Bei Planung behauptete es,
X12 müsse 27 Tage vorher produziert werden, um zum Bestelldatum einzutreffen.
Bei deadline widersprach es sich mit `20 days is longer than the 12 days`.
Diese zwei Fehler treten in den bisher gesehenen BF16-Antworten nicht auf.
Alle sieben Fälle abgeschlossen. Auf denselben sieben positiven Readerfragen
steigt die inhaltliche Codex-Wertung von **Int8 3/7 auf BF16 6/7**. Die früher
genannten Int8 5/9 enthalten zusätzlich zwei UNKNOWN ohne Reader. Diese wurden
hier nicht erneut ausgeführt und werden nicht in die BF16-Quote eingerechnet.
Der deutsche Fall bleibt falsch. Der kleine bekannte Satz ist kein neuer
unabhängiger Generalisierungsbenchmark.

Die drei gemessenen Fallzeiten betragen rund 213,29 / 332,55 / 292,70 Sekunden.
Der vierte Fall `buffer` benötigte 587,80 Sekunden.
`followup_buffer` benötigte 317,07 Sekunden.
Der BF16-Prozess hält etwa 7,4 GB private Bytes bei starkem Speicherdruck.
Das ist eine Messung auf dem aktuell ausgelasteten Rechner, kein belastbarer
isolierter Durchsatzvergleich zwischen BF16 und Int8.

## Grenzen und nächster Nachweis

Die History von `followup_buffer` enthält bereits die Assistenzantwort
`27 days.`. Eine richtige Antwort von 29 Tagen kann daher aus dem Dialog allein
entstehen; dieser Fall ist ein Antwortqualitätsvergleich, kein isolierter
Nachweis eines kausalen Pod-Zugriffs. `followup_reference` und `german_followup`
enthalten in der History nur Identität/Bestellkontext, keinen Zahlenwert.
Für die spätere Lifecycle-Abnahme müssen zusätzlich veraltete Dialogwerte
gegen eine neuere Generation und widerrufene Antwortabhängigkeiten geprüft
werden. Vorgegebene synthetische History ersetzt keine echte Receipt-Kette.

Der Kontrolllauf verwendet frische Präfixzustände und hat keinen integrierten
Planner-/Router-/Lifecycle-Pfad. Auch gute Antworten beweisen weder das gesamte
Nutzerziel noch die historischen CQP1-/Mehrschritt-/Skalierungsbehauptungen.
Die Int8-Variante unterscheidet sich außerdem durch ihre numerische Umsetzung
(quantisierte Linear-Layer und weitere Dtype-Anpassungen); dieser Versuch
isoliert nicht einzelne Layer oder einen einzelnen Rundungsmechanismus.

`research/audit_reader_precision.py` hat erfolgreich geprüft: die
Referenzdatei, genaue Eingabetokens, Runner-Snapshot, gespeicherte erste Logits,
Greedy-Ersttoken, Token/Text/EOS-Konsistenz und das Tokenlimit. Alle sieben
Tokenfolgen unterscheiden sich von Int8. Die separate Inhaltsbewertung nach
derselben Rubrik wie Plain-Int8 liegt in `content-review.json` samt Reporthash.
Ein anschließender BF16-Kapsellauf benötigt neu erzeugte, zum BF16-Modellhash
passende Kapseln und erneut die integrierten Lifecycle-Kontrollen.

## Abgrenzung des verbliebenen Fehlers

Im bestehenden integrierten Linknachweis für `german_followup` wählt der
Planer die richtige Adresse; der BF16-Kontrollreader nennt anschließend den
richtigen Wert27. Falsch ist die Entscheidung zur Frist. Das Planertraining
in `planner_training_data.py` trainiert ausdrücklich nur Adressen/UNKNOWN,
keine Zahlenantworten oder Wochenvergleiche. Eine Verbesserung seines
Adressscores ist daher kein Nachweis einer Behebung dieses Readerfehlers.

Noch auszuführende Diagnose: denselben Fakt und Dialogkontext konstant halten,
Fragesprache Deutsch/Englisch und Fristdarstellung Tage/Wochen kreuzen;
zusätzlich erfüllbare, unerfüllbare und exakt gleiche Fristen prüfen. Das
trennt Sprach-, Einheiten- und Entscheidungsprobleme besser als eine einzelne
umformulierte Fehlerfrage. Diese Diagnose ist noch nicht ausgeführt und
ersetzt nicht spätere getrennte Validierungsdaten oder die integrierte Abnahme.
