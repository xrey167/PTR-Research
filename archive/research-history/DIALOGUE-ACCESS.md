# Modellgestützter Dialogzugriff auf Wissensadressen

**Neuer Trainingsbefund:** Die [Planungs-LoRA](PLANNER-TRAINING.md) erreicht
49/50 auf dem getrennten synthetischen Prüfsatz; nach Neuladen 50/50 identische
Tokenfolgen und 9/9 richtige Adressentscheidungen auf den unten beschriebenen
alten Dialogfällen. Damit ist der zuvor gescheiterte Zugriff deutlich verbessert.
Eine Firmenaltersfrage im neuen Prüfsatz bleibt falsch; die vollständige
Planer-/Link-/Kapselantwort und gemeinsame Planerherkunft fehlen weiterhin.

`dialogue_access.py` ergänzt einen experimentellen Zugriffspfad über dem
Identitätsrouter. Ein eingefrorenes Modell erhält den Dialog und einen
wertfreien Katalog aktuell zulässiger Wissensadressen. Es schlägt `ADDRESS n`
oder `UNKNOWN` vor. Keine der neun Abnahmefragen wird als Trainingsbeispiel
verwendet; der Prompt erklärt die allgemeine Adressaufgabe.

Vor dem Modellaufruf werden aktuelle Pod-, Vektor- und Repräsentationsabhängigkeiten
geprüft. Danach prüft der Wrapper exakte Ausgabesyntax, Kataloggrenzen, explizite
alphanumerische Teilekennungen und bekannte Lieferantennamen. Referenzen können
auf frühere Nutzerbeiträge zurückgehen; Assistant-Text begründet keine explizite
Entitätszuordnung. Eine intern erzeugte kanonische Lookupfrage führt die
ausgewählte Identität durch den bisherigen Dragonfly-/Qdrant-Pfad.

Diese Lookupfrage ist eine interne Brücke zum bestehenden Parser. Sie beweist
nicht, dass Dragonfly bereits aus der ursprünglichen indirekten Frage routet.
Die ursprüngliche Frage und der Dialog bleiben für eine spätere Antwortstufe
erhalten. Es werden keine Faktenwerte aus dem Register als Antworten ausgegeben.

## Herkunft und Grenzen

Weil das Planungsmodell den gesamten zulässigen Katalog sieht, hält der Snapshot
alle darin verwendeten Abhängigkeiten fest. Eine Änderung während der Planung
sperrt die Anfrage. Das ist konservativer als rein selektive Faktenspeicherung;
ein großer Katalog sowie dessen Änderungsrate müssen später vermessen werden.

Die Modellentscheidung bleibt weiche Semantik. Der Wrapper garantiert weder
allgemeine Spracherkennung noch das Erkennen jedes unbekannten Firmennamens,
jeder Intentverwechslung oder Prompt-Injektion. Derzeit beschränkt er sich auf
einzelne Lieferzeitadressen. Mehrere Entitäten, andere Wissensarten und freie
Operatorpläne sind noch nicht unterstützt.

Der Test übergibt Dialogbeiträge als Eingabedaten. Ein produktiver, über mehrere
Anfragen persistierter Dialog muss frühere Assistant-Antworten mit ihren
Receipts verknüpfen und nach Updates/Widerrufen erneut prüfen. Dies fehlt noch,
ebenso der durchgängige Planner-/Link-LoRA-/Kapsel-/Antwort-Commit-Pfad.

## Erste Messung

`dialogue-access-001`, eingefrorenes Qwen-0.5B-Float32:

- 0/9 korrekte Modellentscheidungen, 0/9 gültige Ausgabesyntax.
- 2/9 korrekte Resultate nach Zurückweisung: beide Negativfälle werden nur
  wegen ungültiger Ausgaben abgelehnt. Kein Nachweis semantischen Abstainens.
- 9/9 echte Ausgaben mit EOS, separat mit dem Originaltokenizer nachgeprüft.
- Basisgewichte unverändert. Keine Antwortgenerierung oder Link-LoRA in diesem
  Versuch. Der eingefrorene 3B-Int8-Vergleich verwendet denselben Dialogvertrag.

Sieben Vertragstests prüfen Dialogreferenz, wertfreien Katalog, Teile-/Namens-
widerspruch, ungültige Modelladresse, Update während Planung und ACL vor dem
Modellaufruf. Sie verwenden Fixture-Vorhersagen und ersetzen keinen Modelltest.

## Weitere Gegenproben

| Variante | Korrekte Modellentscheidungen | Korrekt nach Guards | Gültige Syntax |
| --- | --- | --- | --- |
| 0.5B Float32 frei | 0/9 | 2/9 | 0/9 |
| 3B Int8 frei | 0/9 | 1/9 | 1/9 |
| 0.5B Float32 mit erlaubten Ausgabefolgen | 2/9 | 2/9 | 9/9 |

Alle drei Läufe sind abgeschlossen und separat anhand tatsächlicher Tokenfolgen
auditiert. Im 3B-Lauf erzeugt die Preisfrage fälschlich `ADDRESS 2`, also die
Lieferzeitadresse. Die expliziten Teile-/Namensprüfungen verhindern diesen
Intentfehler nicht. Dieser Planer darf deshalb nicht als zuverlässiger
Produktionszugriff oder als harter semantischer Compiler behandelt werden.

Der eingeschränkte 0.5B-Lauf erzeugt für jede Frage `UNKNOWN`. Die Einschränkung
erlaubt ausschließlich Tokenpfade zu gültigen Adressen oder `UNKNOWN`; sie
setzt keine richtige Adresse ein und beweist keine semantische Fähigkeit.
Die zwei korrekten Negativfälle entstehen hier durch tatsächliches Abstainen,
aber alle sieben positiven Fälle bleiben ungelöst.

Nachweise: `../runs/dialogue-access-001`, `../runs/dialogue-access-3b-001`,
`../runs/dialogue-access-constrained-001`, jeweils `protocol.json`, `report.json`
und `audit.json`. Der 3B-Lauf hat zusätzlich einen während der Ausführung
gesicherten Snapshot der unveränderten Laufzeitquellen. Der Audit prüft Tokens,
EOS und Nachzählung, keine unabhängige erneute Inferenz oder Antwortqualität.

Der inzwischen durchgeführte erste Trainingsversuch prüft, ob eine Planungsfähigkeit aus
Dialogreferenzen und wechselnden wertfreien Katalogen zu richtigen Adressen
oder begründetem Abstainen generalisiert. Trainingsnamen, Teile, Katalogreihen-
folgen und Sprachmuster müssen von einer neuen versiegelten Evaluation getrennt
sein. Die neun inzwischen ausgewerteten Fälle sind nur noch Regressionstests.
Weder eine feste Frage-/Adressentabelle noch eine reine Ausgabegrammatik
erfüllt die verlangte Nutzung wie internes Wissen.
