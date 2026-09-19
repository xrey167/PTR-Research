# Training einer allgemeinen Planungs-LoRA

## Abgeschlossenes Training und Audit

`planner-training-003` ist abgeschlossen: **0/50 vor Training, 49/50 danach**.
540.672 LoRA-Parameter wurden in 300 Schritten trainiert; Basisgewichte blieben
unverändert. Gesamtdauer inklusive Vorher-/Nachher-Inferenz: 1.053,03 Sekunden.
Der unabhängige Evidenzaudit zählt dieselben Ergebnisse aus 100 gespeicherten
Tokenfolgen mit EOS nach und prüft Datentrennung, Adapterdateien und Registry-
Hashes. Er wiederholt weder Training noch Inferenz.

Alle fünf Fälle je Familie bestehen für direkte Fragen, Planung, Puffer,
Folgefragen, Preis sowie unbekannte/vorhandene Firmen und Teile. Firmenalter
erreicht 4/5. Fehler: `How many years has Lurena Supply been operating?`
erzeugt `ADDRESS 2` statt `UNKNOWN`. Das ist eine tatsächliche Intentverwechslung
mit gültiger Syntax, kein bloßer Formatfehler. Das feste 50/50-Gate bleibt false.

Der Adapter ist unter `adapter/dialogue_planner/` gespeichert und als
`lora:69e7ce869d9fb52dfd7fb790e92ff710398ef25b564f94fd37a414d5d4625a82`
mit Trainings- und Modellherkunft registriert. Der separate Neuladeversuch
`planner-reload-001` besteht: alle 50 Tokenfolgen exakt gleich. Anschließend
erreicht derselbe geladene Adapter 9/9 korrekte Adressentscheidungen hinter
DialogueAccess. Sieben positive Fragen ergeben tatsächlich `ADDRESS 2` für
Müller GmbH, die Fragen zu X99 und Preis tatsächlich `UNKNOWN`. Darunter sind
indirekte Planung, Puffer, Dialogreferenzen und eine deutsche Folgefrage.
Diese neun Fälle waren bereits bekannt, aber nicht in den Trainingsdaten.
Sie sind Regression, keine weitere unabhängige Generalisierungsstichprobe.

Der Neuladeversuch dauert 84,71 Sekunden, Basisgewichte unverändert. Er erzeugt
noch keine fachlichen Antworten, Rechnungen oder persistierten Planer-Nachweise
im gemeinsamen Faktenregister. Die Trainingsherkunft liegt im separaten
Trainingsregister; ihre Einbindung in die gemeinsame Lifecycle-Grenze fehlt noch.

## Festgelegter Versuch

`planner-training-003` verwendet Qwen-2.5-0.5B und eine neue LoRA mit Rang 8
auf q_proj/v_proj. Basisgewichte bleiben eingefroren. Drei Epochen mit 100
Beispielen ergeben 300 vorab festgelegte Optimierungsschritte, Lernrate 0,0003.
Die gespeicherten Epochen-Checkpoints werden nicht nach Testergebnissen gewählt.

Der vollständige Datensatz und sein Hash werden vor dem ersten Modelllauf
gespeichert. 100 eindeutige Trainings- und 50 eindeutige Prüfeingaben verwenden
getrennte Firmen, Teile und Formulierungen. Kataloge haben zwei oder drei
Einträge, wechselnde Reihenfolgen und gleich häufig ausgewählte Positionen
innerhalb jeder Kataloggröße. Adressen 1/2 treten insgesamt häufiger auf als 3,
weil kleinere Kataloge keine dritte Adresse besitzen.

Aufgaben: direkte Anfrage, Ankunftsplanung, Puffer, Dialogreferenz, Preis,
Firmenalter sowie unbekannte Teile/Firmen. Zu unbekannten Teilen/Firmen existieren
positive Gegenproben mit derselben Frageform und tatsächlich vorhandenen Daten.
40 % sind UNKNOWN-Ziele. Keine Faktenwerte werden als Trainingseingabe genutzt.

Die Vorher-/Nachher-Prüfung verwendet freie Ausgabe ohne erzwungene Grammatik.
Vollständige Tokenfolgen einschließlich EOS werden gespeichert. Der spätere
Audit zählt aus diesen Tokens und den vorab gespeicherten Zielen nach.
Die Planungs-LoRA erhält einen Registry-Eintrag mit Trainings-/Modellherkunft.

## Datenkorrekturen vor dem endgültigen Vergleich

- `planner-training-001`: nach 28 Basisantworten und vor jedem Trainingsschritt
  gestoppt. Unbekannte Firmen hatten einen unbeabsichtigten Namenshinweis
  (`Unknown`). Er wurde entfernt; Teilantworten bleiben erhalten.
- `planner-training-002`: nach 32 Basisantworten und 50 Trainingsschritten
  gestoppt. Die Datensatzprüfung zeigte eine Positionsverzerrung, fehlende
  ADDRESS-3-Prüffälle und fehlende Gegenproben. Kein fertiger Adapter oder
  bestandener Vergleich wird daraus behauptet.
- `003` startet wieder von der unveränderten Basis mit dem korrigierten
  Datensatz. Die Änderungen erfolgten ohne Auswahl anhand trainierter
  Prüfergebnisse. Dennoch ist dies ein kleiner synthetischer Entwicklungsversuch,
  keine unabhängig durchgeführte externe Validierung.

## Grenzen

Der erste Trainingsvertrag ist englisch und auf Lieferzeitadressen beschränkt.
Freie neue Wissensarten, größere Kataloge, deutschsprachige Dialoge, Mehr-Hop,
persistente Dialogherkunft und der vollständige Kapselantwortpfad bleiben offen.
Ein bestandener Adresstest wäre kein Nachweis korrekter Rechnung oder Antwort.
Die neuen Katalogeinträge des Prüfsatzes wurden ohne weiteres Training
zugeordnet. Der erste Befund ist positiv, aber auf diesen kleinen synthetischen
Bereich beschränkt; die Intentverwechslung bleibt ein Gegenbeleg vollständiger
Zuverlässigkeit. Das Set ist nach der Auswertung ein bekannter Regressionstest.
