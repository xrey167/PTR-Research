"""Render DS1 report from completed raw-result audits."""
import json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent.parent;here=Path(__file__).parent
if __name__=='__main__':
 a=json.loads((here/'validation/results.json').read_text());b=json.loads((here/'3b_validation/results.json').read_text())
 assert json.loads((here/'validation/audit.json').read_text())['passed']
 assert json.loads((here/'3b_validation/audit.json').read_text())['passed']
 table='| Pfad | 0,6B korrekt / 24 | 3B korrekt / 24 | 0,6B Änderungspaare / 12 | 3B Änderungspaare / 12 |\n|---|---:|---:|---:|---:|\n'
 for k in ['none','cc8','opposite','text']:table+=f"| {k} | {a['correct'][k]} | {b['correct'][k]} | {a['counterfactual_pairs_correct'][k]} | {b['counterfactual_pairs_correct'][k]} |\n"
 text=f'''# DS1 – echte Decodersemantik und Reparatur der Modellherkunft

## Neue Ausführung

Beide original gepinnten Modelle wurden vollständig wiederhergestellt und alle
Gewichtshashes geprüft. Die Runtime verwendet Torch 2.14.0+cpu, Transformers
5.17.0, NumPy 2.5.3, Safetensors 0.8.0 und huggingface_hub 1.31.0. Der frühere
Blocker „keine Modelle/Dependencies“ ist für diesen lokalen Arbeitsstand behoben.
Gewichte bleiben reproduzierbar über die original gepinnten Downloadprogramme;
sie sind wie bisher nicht Bestandteil des Forschungsarchivs.

Entwicklung: acht 0,6B-Modellsequenzen. Anschließend versiegelte Validierung:
96 Sequenzen auf 0,6B und 96 auf 3B. Insgesamt **200 neue reale Modellsequenzen**.
Das 3B-Protokoll entstand nach der schwachen 0,6B-Textkontrolle; dieselben Fragen
sind daher für 3B eine neue Ausführung, kein neuer unabhängiger Fragensplit.
Alle Antworten einschließlich Tokenfolgen, Terminierung und ersten Yes/No-
Logits sind in den jeweiligen answers.jsonl-Dateien erhalten.

{table}

`none` erhält nur die Frage; `cc8` addiert den unveränderten CC8-Bool-Tensor
zum letzten Prefill-Token; `opposite` verwendet den gegenteiligen Boolcode;
`text` erhält den korrekten Fakttext. Die Frage ist zwischen den beiden
Wertewelten identisch. Der Eingriff geschieht genau einmal an Schicht 23 (0,6B,
FP32) beziehungsweise 17 (3B, BF16), entsprechend den archivierten Basisorten.

Das vorab festgelegte minimale Kandidatengate lautet 24/24 Antworten und
12/12 vollständig richtige Änderungspaare. Status: 0,6B **{a['candidate_gate']}**,
3B **{b['candidate_gate']}**. Die beiden separaten, modellfreien Token-/Zählaudits
bestehen. Dies ist keine externe Replikation. Ein bestandener Audit bestätigt
auch ein negatives Versuchsergebnis; er macht den Kandidaten nicht erfolgreich.

## Was daraus folgt

Die reine Addition des CC8-Codes ist kein nachgewiesener semantischer Reader.
Der Versuch prüft genau diesen Anschluss und keine beliebigen gelernten oder
mehrschichtigen Adapter. Die geringe Qualität der 0,6B-Textkontrolle ist ein
zusätzlicher Confound und bleibt sichtbar. Auch ein perfekter numerischer
Rückdecoder kann keine Aussage über den Sprachdecoder ersetzen.

Der separate Entwurf SEMANTIC_LINKER_DESIGN_DE.md formalisiert die fehlende
Bedeutungsbindung und einen prüfbaren gemeinsamen Linker. Dieser Linker wurde
in DS1 noch nicht trainiert. Es gibt keinen Neuheits-, RAG- oder Pareto-Nachweis.

## Herkunftsfehler in CC5–CC8

Alle vier numerischen Compiler trugen für Qwen3 die falsche Zeichenfolge
`c1899de289a04d12100db3703e84a94a8a3df08b`. Der ursprüngliche archivierte
Modellnachweis, Downloadcode, Guard und der neu geprüfte Anbieterstand tragen
`c1899de289a04d12100db370d81485cdf75e47ca`. Die falsche Revision lieferte HTTP 404.
Die 3B-Revision war korrekt. provenance_audit.json dokumentiert die vier Stellen.

Die alten Quellen und ihre Versiegelungen bleiben unverändert. Der neue
ProvenanceCompiler liest die Herkunft aus den ursprünglichen Nachweisen,
verknüpft Revisionen, Gewichtshashes, Asset- und Quellcodehashes und erzeugt neue
ABI-Roots. Auf den geprüften Symbolen bleiben die numerischen Tensorbytes gleich.
Der Migrationstest verwendet die tatsächliche CC4-QueryBridge; alte Roots müssen
unter dem neuen Compiler abgewiesen und frische Handles akzeptiert werden.
Dies repariert die Deklaration und Bindung, beweist aber nicht rückwirkend,
welcher Prozess die historischen Tensoren berechnet hat.

## Weitere präzisierte Grenzen aus dem Nachaudit

Der in CC8 genannte Faktor 4,6411 verglich rohe Abstände bei unterschiedlichen
Codenormen (CC7: 2; CC8: 1). Bei gleicher Einheitsnorm lautet der geometrische
Faktor 9,2822. Das ist kein gemessener Modell- oder Qualitätsvorteil.

Der CC8-Radius 0,25 garantiert die Identität des nächsten Gitterpunkts, aber
nicht dessen Akzeptanz nach der zusätzlichen Gap-Schwelle 0,01. Ein Radius
kleiner als 0,24 genügt für beides, da Gap >= 0,25 - ||Störung||. Die damaligen
Quantisierungsfehler liegen auch unter diesem kleineren Radius. Die bisherigen
Berichte bleiben als Historie erhalten; geometry_correction.json korrigiert die
Interpretation ausdrücklich.

## Fortsetzung

Die nächste Runde muss einen semantisch kalibrierten, gemeinsam eingefrorenen
Linker gegen exakt übernommene Teacher-Zustände und die vorhandenen KVC1/BT5-
Pfade testen. Inhalt, neue Namen und neue Operatoren sind getrennte Testachsen.
Ein erneut funktionierender Zahlencode reicht nicht. Die ursprünglichen 15
DoD-Punkte gelten gemeinsam; 100.000 Fakten, 4/8/16-Hop, starker RAG-Vergleich,
End-to-End-Overhead und belegte Neuheit bleiben offen.

Quellen der Modellherkunft (erneut über die Anbieter-API geprüft):
- https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca
- https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/tree/aa8e72537993ba99e69dfaafa59ed015b17504d1
'''
 (here/'REPORT_DE.md').write_text(text)
 print('report written from completed result files')
