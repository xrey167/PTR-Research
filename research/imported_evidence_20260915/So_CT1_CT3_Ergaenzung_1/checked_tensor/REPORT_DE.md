# CT1–CT3: geprüfte Tensorberechnung und Kosten einer algebraischen Kontrolle

## Ergebnis und Geltungsbereich

Diese Fortsetzung liefert einen implementierten, eng begrenzten Tensorprüfer
und zwei algebraische Prüfvarianten. **Keine neue Modellantwort und keine
Architekturüberlegenheit sind nachgewiesen.** Die volle ursprüngliche DoD
bleibt bindend. Die Schritte ersetzen weder den eingefrorenen echten Backbone,
100.000 neue Fakten, natürliche Fragen und neue 4/8/16-Hop-Operatoren noch
Lifecycle-/JIT-/RAG-/ABI-/Neuheitsnachweise am selben vollständigen System.

Der neueste Hauptstand ist Version 11. Drei tatsächliche Materialisierungs-
versuche scheiterten mit HTTP 502. Er konnte in diesem Lauf nicht gelesen oder
geändert werden. Das Ergänzungspaket verwendet die im Gespräch vorliegende
TC1-Formel und CC1-Beschreibung; es ist keine Reproduktion ihrer archivierten
Originaldaten. Modellgewichte, torch und transformers sind ebenfalls nicht
verfügbar. Die hier verwendeten Tensoren wurden eigens für mathematische
Kontrollen erzeugt und sind vollständig beigefügt.

## CT1: Zwischen gültiger Revision und korrekter Berechnung unterscheiden

CC1s dokumentierte Vertrauensgrenze bleibt wesentlich: Ein freier Compiler kann
fertige Bytes unter einer aktuellen Revision deklarieren. Ein Hash beweist,
welche Bytes vorliegen; er beweist nicht, dass diese Bytes das Ergebnis des
angegebenen Programms und seiner Inputs sind.

CT1 beschränkt das Programm auf eine Datenbeschreibung mit vier Operationen:
Input, Addition, Subtraktion, Matrixprodukt. Formen, numerische Typen,
Vorwärtsreferenzen, verwendete Eingaben und endliche Werte werden geprüft.
Ein Prüfer berechnet das Ergebnis unter dem autoritativen Snapshot erneut und
bindet Programmdigest, Arithmetik-ABI, Eingabemanifest, Epochen und Ausgabedigest.

Geprüfte Formel: **B + S + R + C(Y−X)**. C ist hier ein expliziter Input.
Seine Herleitung als R X⁺ wird **nicht** mitbewiesen. Der vollständige Compiler
müsste auch diese Herleitung und die ursprünglichen Quellrevisionen prüfen.
Die aufrufende Instanz muss Programm und Snapshot autoritativ vorgeben; ein
vom Produzenten selbst erfundener Snapshot wäre keine solche Autorität.

Nach zwei Entwicklungsseeds wurden Code und Protokoll versiegelt. Die
Validierung umfasst 12 neue Seeds × 3 Tensorformen = **36 mathematische Fälle**:

- sämtliche Ergebnisse stimmen elementweise mit unabhängiger rationaler
  Arithmetik überein;
- 36/36 alte Ausgaben mit aktualisierten Metadaten werden abgelehnt;
- eine Kontrolle, die lediglich den neuen Manifest-/Ausgabedigest vergleicht,
  akzeptiert alle 36 dieser Fälle;
- aktuelle Ausgaben werden akzeptiert; undeclarierte Inputs, alte Belege nach
  Epochenerhöhung sowie sechs strukturelle Negativkontrollen werden abgelehnt;
- bei ABA sind unveränderte Ausgabebytes unter einem neu berechneten Beleg
  zulässig, während der alte Beleg unzulässig bleibt.

Der unabhängige Prüfer rekonstruiert aus gespeicherten Ganzzahl-/Zweierbruch-
daten sowohl ursprüngliche als auch geänderte Ausgaben. Das ist ein separater
Codepfad desselben Assistenten, keine externe Replikation.

CT1 führt keine beliebigen Python-Programme aus. Es ist deshalb auch keine
Sandbox, die beliebigen Python-Code zuverlässig auf unerlaubte Dateizugriffe
überwacht. Es ist ein beschränkter Datenauswerter mit reproduzierbarer Rechnung.
Es beweist keine kausale Entstehung einer bereits extern erzeugten Datei;
es prüft deren Gleichheit mit dem Ergebnis des autorisierten Ausdrucks.

## CT2: billige Zufallsprüfung scheitert an kleinem Rang

Die bekannte Freivalds-Idee prüft ein Matrixprodukt mit Zufallsvektoren statt
vollständiger Multiplikation. CT2 setzt sie auf exakt skalierte Ganzzahlen
modulo 2³¹−1 um. Eingabe-/Ausgabebounds verhindern, dass ein anderer Integer
durch Addition des Moduls fälschlich als identisch gilt. 36 gültige Fälle,
36 Einzelfehler und 36 Alias-Kontrollen bestehen die jeweilige Prüfung.

40 binäre Prüfvektoren führen für eine vorher fixierte falsche Matrix zu einer
theoretischen Fehlannahmegrenze ≤2⁻⁴⁰. Das ist keine gemessene Fehlerquote.
Die öffentlichen Testseeds dienen Reproduktion, nicht adversarialer Sicherheit.
Für einen solchen Einsatz müssten unabhängige geheime Challenges erst nach
Festlegung der Produzentenausgabe gezogen werden.

Der Operationenzähler zeigt: Für die TC1-Form m=15,k=6,d=256 braucht diese
Variante etwa 9,49× so viele MAC-Terme wie das direkte Produkt; für einen
einzigen Paarterm 46,82×. Damit ist **diese** Prüfvariante als billigere
TC1-Alternative ausgeschieden. Es handelt sich um eine algebraische Rechnung,
nicht um gemessene CPU-/GPU-Latenz.

Eine nachträgliche Codeprüfung ergänzte explizite Grenzen für die erlaubten
Tensorformen. Erstimplementierung und damalige Ergebnisse sind als
`check_product_initial.py` und `cert_results_initial/` erhalten; die geschlossene
Formenvalidierung wurde danach ausgeführt. Das sind Entwicklungswiederholungen.

## CT3: zwei Feldvektoren — Bankprüfung hilft, Anfragenprüfung nicht

CT3 verwendet die ebenfalls bekannte Alternative zweier gleichverteilter
Vektoren aus dem endlichen Körper. Die theoretische Fehlannahmegrenze sinkt
unter den genannten Voraussetzungen auf p⁻² < 2⁻⁶¹. Auch diese 36 Kontrollen
verwenden dieselben Entwicklungsdaten und sind keine unabhängige neue Domäne.

| Aktive Paarterme m, k=6, d=256 | Direkt: MAC-Terme | Prüfprodukt: MAC-Terme | Verhältnis |
|---|---:|---:|---:|
| 1 | 1.536 | 3.596 | 2,341 |
| 3 | 4.608 | 4.644 | 1,008 |
| 15 | 23.040 | 10.932 | 0,474 |

Die drei Quellen des bisherigen Ansatzes aktivieren höchstens drei Paarterme
gleichzeitig. Die günstigere Bilanz für **alle 15** darf daher nicht als
Inferenzgewinn einer einzelnen Anfrage ausgegeben werden. Selbst die
optimistische Kernrechnung spart dort nichts. Modulararithmetik, Bounds,
Präzisionsumwandlung, Hashing und zufällige Challenges verursachen zusätzliche
Arbeit. Der Prüfrunner enthält außerdem redundante Bounds-Arithmetik; die
Tabelle beschreibt einen optimierten algebraischen Ablauf, keinen Laufzeitbench.

## Architekturentscheidung

Geprüfte Compilerergebnisse bei Installation/Freigabe erzeugen und anschließend
an aktuelle Revisionen binden ist ein sinnvoller nächster Integrationspfad.
Eine aufwendige Rechenkontrolle bei jeder einzelnen kleinen Transportanfrage
ist durch diese Ergebnisse nicht gerechtfertigt. Ob einmalige Prüfung sich
über viele Anfragen amortisiert, muss einschließlich aller Erstellungskosten
und Invalidierungen gemessen werden.

Das löst noch nicht den wichtigeren neuronalen Engpass: ein gemeinsamer
Wissenscompiler für viele neue Fakten und unbekannte Operatoren. Die beiden
letzten Runden dürfen deshalb keine fortwährende Ausweichbewegung in weitere
Lease- oder Zufallsprüfungen werden. Sobald Zugriff und Modell vorhanden sind,
hat echte nicht-farbliche Mehrquellen-/Operatorvalidierung Vorrang.

## Vorarbeiten und offene Neuheit

Deterministisches Nachrechnen und probabilistische Matrixprüfung sind keine
neuen Techniken. Zwei geprüfte Primärquellen zur Einordnung:

- Ji, Mascagni, Li (2017), [Gaussian Variant of Freivalds' Algorithm](https://arxiv.org/abs/1705.10449):
  Matrixproduktprüfung durch Projektionen; exakte und Gleitkommaarithmetik
  müssen unterschieden werden.
- Künnemann (2018), [On Nondeterministic Derandomization of Freivalds' Algorithm](https://arxiv.org/abs/1806.09189):
  die Suche nach deterministischer schneller Matrixproduktprüfung ist eine
  etablierte Forschungsfrage.

Die Datenbeschreibung und Revisionsbindung sind ein konkreter Entwicklungsschritt
für dieses Projekt. Es wird keine Patentneuheit oder Überlegenheit gegenüber
starkem RAG beansprucht.

## Dateien und spätere Integration

`run.py`, `check_product.py`, `field_checks.py` sind tatsächlich ausgeführte
Mathematikexperimente. `audit.py` prüft gespeicherte Daten unabhängig nach.
Alle Protokolle, Seeds, Ergebnisse, Tensoren und die Erstfassung sind enthalten.
Runner benötigen NumPy und einen noch nicht vorhandenen jeweiligen Ergebnisordner.

Dieses Ergänzungspaket muss beim nächsten erfolgreichen Download in die dann
aktuellste Hauptfassung übernommen werden; weder Version 11 noch eine spätere
Version darf blind überschrieben werden. `FROZEN_DOD.md` bleibt unangetastet.
