# Gelernte Wissensadresse über Wertänderungen hinweg

## Ergebnis

`runs/identity-transfer-002` führt echtes Training mit dem eingefrorenen lokalen
MiniLM-Encoder und echte Qwen-0.5B-Link-LoRA-Inferenz aus. Zwei neue
Identitätsadressen werden einmal gelernt. Anschließend wird der Müller-Fakt
von 18 auf 27 geändert, ohne weitere Router- oder LoRA-Trainingsaufrufe.

- Sechs tatsächliche Qwen-Linkausgaben vor und sechs nach dem Update stimmen.
- RepresentationKey und LoRA-ArtifactKey bleiben exakt gleich.
- Auswahlscore und Repräsentation sind nach Neuladen unverändert.
- Der alte Antwort-Snapshot wird beim Commit abgewiesen.
- Qwen-Basisgewichte bleiben unverändert. Gesamtlauf 24,97 Sekunden.
- Separater Audit: 61 Knoteninhalte mit Elternlisten nachgehasht; zwölf
  Linktexte, Receipt-Abhängigkeiten und Adapterdateien geprüft.
- Gesamtsuite: 120 Tests in 23,36 Sekunden, Exit 0;
  `../runs/identity-transfer-tests.xml`.

Der erste Lauf `identity-transfer-001` scheiterte im Update-Hilfscode, weil
die Gültigkeitsmetadaten nicht mitgegeben wurden. Er bleibt unverändert als
Fehlversuch erhalten; der zweite Lauf übernimmt die bestehenden Metadaten.

## Mechanismus und Herkunft

`identity_dragonfly.py` ist eine separate Forschungsvariante. Sie übernimmt
keine alten generationengebundenen Gewichte. Ihre Initialisierung verwendet
ausschließlich den kanonischen Identitätsdeskriptor: Subjekt, Komponente,
vertrauenswürdige Aliase, Wissensidentität und typisierten Cluster. Faktenwert,
konkreter Pod und dessen Vektor fließen nicht in diese Initialisierung ein.
Adressfragen und negative Beispiele werden als eigenes Training dokumentiert.

Die Repräsentation hängt von der separaten Identitätsgeneration, deren
tatsächlichen Quellen und gegebenenfalls zusätzlichen Trainingsquellen ab.
Ein Widerruf dieser Quellen sperrt die Adresse weiterhin. Das System behauptet
nicht, eine geänderte Herkunft durch identische Metadaten ersetzen zu können.

Zur Laufzeit bleiben aktueller Fakten-Pod und Vektor zusätzliche verpflichtende
Abhängigkeiten. Die wiederverwendbare Adresse erteilt keine Berechtigung.
Alias-/Clusteränderungen verlangen neues Identitätstraining; aktuelle ACL und
Gültigkeit werden unabhängig geprüft. Das gilt auch für Aliasdiagnosen.

## Zusätzlicher echter Kapsellauf

`runs/identity-linked-001` übernimmt den trainierten Identitätsrouter und die
Generation mit Wert 27 aus `identity-transfer-002`. Sechs neue echte Qwen-
Linkausgaben werden unter gemeinsamer Herkunft gespeichert; anschließend liest
Qwen-3B-Int8 eine neu kompilierte Geschwisterkapsel derselben Generation.

- 6/6 Verweise korrekt; 5/6 Readerantworten korrekt `27` mit EOS.
- Fehler bei `Wie lange braucht Muller GmbH fuer X12?`: Ausgabe
  `Fact: Muller GmbH has a delivery lead time of`, dann Tokenlimit erreicht.
  Der unveränderte Lookup-Test ist somit **nicht bestanden**.
- Frischer Textpräfix und Kapsel liefern im Runner jeweils dieselben Tokens
  und Ersttoken-Logits. Der Fehler ist auch in der Textreferenz vorhanden.
- Vier Lifecycle-Kontrollen bestanden, beide Basisgewichte unverändert.
- Separater Evidenzaudit bestätigt 74 Knotenhashes, sechs Receipt-Herkunftsketten,
  Dateihashes und sechs vollständige Logitpaare; zählt unabhängig 5/6 Antworten.
- Reader-Stufe 154,37 s einschließlich Laden/Umwandlung/Hashprüfungen;
  Median Reader-Anfragepfad 1,132 s. Kein belastbarer Servingbenchmark.

Die 27-Kapselantwort wurde damit tatsächlich getestet. Vollständige
Antwortzuverlässigkeit ist noch nicht erreicht. Ein größerer Tokenrahmen allein
würde die bereits falsche Ausgabeform nicht nachträglich korrekt machen.

## Grenzen und nächster Schritt

Der ursprüngliche Transferlauf belegt Adressauflösung und besitzt einen
textuellen Test-Pod. Der zusätzliche Kapsellauf erweitert diesen Nachweis wie
oben beschrieben. Der Transfer-Audit wiederholt keine Inferenz und prüft mangels Roh-Tokenfolgen der
Linkstufe keine unabhängige Tokendecodierung. Die zwölf Ausgaben beziehen sich
auf sechs bestehende Lookupfragen, nicht auf eine neue Validierungsmenge.

Die ursprüngliche Dragonfly-v2-Klasse bleibt generationengebunden. Die neue
Forschungsvariante ist noch nicht im normalen CLI-/Dialogpfad aktiviert.
Neue Entitäten benötigen weiterhin eigenes Adresstraining: Das historische
Ziel von 100.000 neuen Fakten ohne faktenspezifische Gradienten ist nicht belegt.

Die Nutzervorgabe bleibt: Pod-Wissen soll sich wie internes Wissen verhalten.
Dazu müssen der gemeinsame Readerpfad, implizite Zugriffe, Folgefragen,
Operationen und Mehrschritt-Schlüsse funktionieren. Ein stabiler Verweis ist
ein Baustein dieses Verhaltens, nicht dessen vollständiger Nachweis.

`internal-knowledge-cases.json` enthält neun neue, nicht trainierte Dialogfälle.
Die erste Parserdiagnose in `identity-linked-001/access-parser-diagnostic.json`
erfüllt nur drei Zielentscheidungen: direkte Frage und zwei Zurückweisungen.
Sechs indirekte/kontextabhängige Zugriffe scheitern. Dabei wird keine Antwort
generiert; 3/9 ist ausdrücklich keine Modellqualitätsmetrik. Der bestehende
Parser besitzt keine Dialogschnittstelle. Dies ist die nächste funktionale
Lücke auf dem Weg zum geforderten Nutzerverhalten.
