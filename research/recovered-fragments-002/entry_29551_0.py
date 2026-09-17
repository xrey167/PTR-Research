from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent.parent;HERE=ROOT/'binding_transfer'
def doc(run,n):return json.loads((HERE/run/n).read_text())
runs=('run','run_bt2','run_bt3')
results={run:doc(run,'results.json') for run in runs}
audits={run:doc(run,'audit.json') for run in runs}
assert all(a['passed'] for a in audits.values())
b1,b2,b3=[results[x] for x in runs]
# Same-input exact controls must agree across the two new-value experiments.
rows2=doc('run_bt2','rows.json');rows3=doc('run_bt3','rows.json')
ref2={(tuple(r['world']),r['op']):r['ids'] for r in rows2 if r['mode']=='exact'}
ref3={(tuple(r['world']),r['op']):r['ids'] for r in rows3 if r['mode']=='exact'}
assert ref2==ref3 and len(ref2)==16
n=sum(r['qa_generations']+r['execution_gate_generations'] for r in results.values())
lines=['# Bindung und Wissen: BT1–BT3 Forschungsstand','',
'Das Gesamtziel einer neuen, RAG überlegenen Wissensarchitektur ist weiterhin offen. Diese Runde prüft die Wiederverwendung neuronaler Wissensterme über Entitätsnamen und Wertvokabulare hinweg. Alle Ergebnisse stammen aus tatsächlicher Inferenz des eingefrorenen Qwen2.5-3B-Instruct; keine Modellgewichte wurden trainiert.','',
f'Insgesamt **{n} erzeugte Antwortsequenzen**, einschließlich der Ausführungskontrollen. Das sind keine {n} unabhängigen Fakten. Drei kleine synthetische Versuche ersetzen weder den 100.000-Fakten-Nachweis noch einen starken RAG-Vergleich.','',
'## BT1: Namen übertragen','',
'Drei bisher in IM1 unbenutzte Namenspaare X/Y, C/D und K/M; vier Wissenszustände und vier direkte bzw. referenzielle Fragen. Je Kandidat 48 Fragen. Pro Namenspaar genau ein neuer Ausgangszustand (rot/rot/erstes Objekt). Alle Testfarben stammen auf dem Kandidatenpfad aus der alten, unveränderten IM1-Bank.','',
'| Verfahren | Richtig |','|---|---:|']
labels={'anchor':'Nur Ziel-Ausgangszustand','donor2':'Alte Wissensbilder ohne Namensanpassung','rebase1':'Ziel-Ausgangszustand + alte Einzelterme','rebase2':'Ziel-Ausgangszustand + alte Einzel- und Paarterme','exact':'Vollständig berechneter Zielpräfix'}
for m,t in b1['totals'].items():lines.append(f"| {labels[m]} | {t['correct']}/{t['n']} |")
lines += ['',f"Vorab festgelegte Gates: {b1['gates']}. Der stärkere Kandidat verwechselte bei K/M zweimal das andere mit dem ausgewählten Objekt. Die Fehler wurden nicht mit Goldantworten korrigiert.",'',
'## BT2 und BT3: neue Werte übertragen','',
'Neue Werte schwarz/weiß/lila/orange (englische Prompts), Namenspaar X/Y. Nur acht Ausgangs-/Einzelquellenzustände wurden kalibriert. Alle vier getesteten gemeinsamen Zustände waren davon ausgeschlossen. 16 Fragen pro Verfahren. Die neuen Werte sind neu gegenüber unserer IM1-Kalibration, nicht unbekannt im Sprachmodell.','',
'BT2 übernimmt Paarterme aus dem alten Farbkatalog. BT3 entfernt daraus Komponenten im lokalen Unterraum der sechs alten Farb-Einzelterme. Die Projektion verwendet pro Schicht, K/V-Kanal und Position eine FP64-Pseudoinverse mit festem Schwellwert; Fragen, Testzustände und Gold kommen darin nicht vor. Orthogonalität ist eine geometrische Eigenschaft und kein Nachweis semantischer Trennung.','',
'| Verfahren | Richtig |','|---|---:|']
for run,m,label in [('run_bt2','target1','Nur neue Einzelterme'),('run_bt2','hybrid2','Neue Einzelterme + alte Paarterme'),('run_bt3','projected2','Neue Einzelterme + projizierte alte Paarterme'),('run_bt2','exact','Vollständiger Zielpräfix')]:
 t=results[run]['totals'][m];lines.append(f"| {label} | {t['correct']}/{t['n']} |")
lines += ['',
'Die 16 exakt berechneten Referenzantworten wurden in BT3 erneut erzeugt und stimmen tokenweise mit BT2 überein. Die Ziel-Einzelterme wurden für BT3 erneut berechnet. Kandidatenantworten wurden jeweils vor der Berechnung gemeinsamer Testpräfixe festgeschrieben. BT2/BT3 sind sequenzielle explorative Versuche; sie sind keine unabhängigen Bestätigungsstudien.','',
'## Nachweise und Grenzen','',
'- Separater gespeicherter Token-Audit für alle drei Versuche: Antworten erneut dekodiert, Gold separat hergeleitet, Paarvergleiche und Quellhashes geprüft. Prüfung durch denselben Assistenten, keine externe Replikation.',
'- BT1: 24 Kontrollsequenzen, drei Vergleiche vollständige Eingabe gegen Präfixcache; BT2/BT3 jeweils acht Kontrollsequenzen. Alle Ausführungskontrollen verlangen identische Sequenzen und erste Logit-Abweichung null.',
'- 49 textabgeleitete latente Präfixpositionen. Keine Behauptung, damit bereits die strikte Wissens-Token-Anforderung des Gesamtprojekts erfüllt zu haben.',
'- Keine Gradienten pro Fakt; numerische Zustandsmontage unabhängig von Frage und Gold. Neue Namen/Werte benötigen weiterhin Kalibrationsbilder.',
'- CPU-Läufe teilweise gleichzeitig. Zeitmessungen sind dokumentierte Laufzeiten, kein fairer Geschwindigkeitsvergleich.',
'- Keine 100.000 Fakten, kein 4/8/16-Hop-Nachweis, keine zweite Modell-ABI, keine neue Lifecycle-Integration in dieser Runde. Frühere Lifecycle-Ergebnisse werden hier nicht als integriert ausgegeben.','',
'## Forschungsentscheidung','',
'Die Übertragung der Paarterme hat in diesen kleinen Tests einen messbaren Nutzen gegenüber Einzeltermen. Die Generalisierung ist jedoch nicht exakt. Eine skalierbare Architektur braucht einen gemeinsam verwendbaren Bindungsoperator, der Identität und neue Werte verarbeitet, ohne für jede Entität-Wert-Kombination ein eigenes neuronales Bild zu speichern. Der vorliegende Ansatz bleibt ein Messinstrument für diese Anforderung.','',
'Der nächste Architekturentscheid darf deshalb nicht bloß aus mehr Farbkombinationen bestehen: benötigt wird ein geteilter Compiler für Bindungswechsel mit getrennten Trainings-/Testmengen für Namen, Werte und Operatoren. Seine vollständigen Speicher-, Installations- und Inferenzkosten müssen gegen gewöhnliches Präfix-Caching und starke Retrieval-Verfahren antreten.','',
'## Vorarbeiten','',
'Interne Bindungsrepräsentationen sind bekannt: [Feng & Steinhardt](https://arxiv.org/abs/2310.17191). Ein Reihenfolge-Unterraum mit kausalem Einfluss auf Bindung wurde von [Dai et al.](https://arxiv.org/abs/2409.05448) untersucht. Ankerbasierte Korrektur kontextabhängiger KV-Offsets ist Gegenstand von [KVComm](https://arxiv.org/abs/2510.12872). Abstracts wurden geprüft. Bindungsvektoren, KV-Differenzen oder orthogonale Projektion allein tragen keine Neuheitsbehauptung. Eine Patentprüfung ist offen.','',
'Details: `binding_transfer/PROTOCOL*.md`, `TRANSPORT_ARCHITECTURE.md`, ausführbare Läufe `run_bt1.py` bis `run_bt3.py`, gespeicherte Tokens, Quellseals und Audits. `FROZEN_DOD.md` bleibt unverändert.','']
text='\n'.join(lines)
(HERE/'REPORT_DE.md').write_text(text);(ROOT/'BT_FORSCHUNGSBERICHT_DE.md').write_text(text)
(ROOT.parent.parent/'So_BT_Forschungsbericht.md').write_text(text)
summary=dict(status='experiment_batch_complete',results=results,audits=audits,total_generated_sequences=n,exact_bt2_bt3_sequences_identical=16,full_project_dod_complete=False)
(HERE/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(generations=n,totals={k:v['totals'] for k,v in results.items()})))
