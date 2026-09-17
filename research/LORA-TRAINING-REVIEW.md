# LoRA-Guide auf den aktuellen Prototyp angewendet

Der Nutzer hat den Unsloth-Hyperparameter-Guide als Arbeitsgrundlage geliefert.
Das laufende BF16-Integrationsexperiment bleibt unverändert.

## Tatsächlicher Stand

| Parameter | Trainierter Dialogplaner | Neuer Reader-Kandidat |
|---|---|---|
| Modell | Qwen2.5-0.5B, FP32 im bestehenden PodModel | Qwen2.5-3B, BF16 vorgesehen |
| Rank / Alpha | 8 / 16 | 16 / 32 |
| Module | q_proj, v_proj | q/k/v/o und gate/up/down |
| Lernrate | 3e-4 | 2e-4 |
| Epochen | 3 | 2 |
| Microbatch / Akkumulation | 1 / 1 | 1 / 16 |
| Weight decay | 0 | 0.01 |
| Warmup / Scheduler | keiner / konstante Lernrate | 5% / linear |
| Assistant-only-Loss | implementiert | erforderlich |
| Status | tatsächlich trainiert, 49/50 Adressentscheidungen | nicht trainiert |

Quellprüfung: neural_pods/model.py und research/train_dialogue_planner.py.
Der ältere allgemeine train_adapter-Default von0.003 ist nicht die tatsächlich
verwendete Lernrate des gesonderten Planertrainers (0.0003).

Die konkrete Kandidatenkonfiguration steht in reader-lora-candidate.json.
Sie ist noch nicht an einen neuen Readertrainer angeschlossen. Der vorhandene
3B-Reader ist unverändert; seine Fehler lassen sich nicht durch Hyperparameter
eines ausschließlich auf Adressen trainierten anderen Modells erklären.

Eine Trainingsprimitive ist jetzt implementiert: `reader_training.py` summiert
den Loss über überwachte, kausal verschobene Antworttokens und normalisiert
über das tatsächliche Akkumulationsfenster. Zwei Fensterprüfungen und drei
Autogradtests mit einem kleinen FP64-Modell bestehen. Der Vergleich verwendet
ungleiche Antwortlängen und einen unabhängigen Gesamtbatch-Loss. Das validiert
die Akkumulationsmathematik, noch kein Qwen-LoRA-Training oder DDP/AMP-Verhalten.

## Was aus dem Guide übernommen wird

Breitere Attention-/MLP-Zielmodule, moderates Rank, Assistant-only-Loss,
Warmup, feste Seeds und getrennte Validierung sind sinnvolle Ausgangspunkte.
Das sind Versuchseinstellungen, keine garantierten Optima. Ein größeres Rank
ersetzt keine Trainingsbeispiele für die tatsächlich geforderte Operation.

Für unser Ziel lernen die Gewichte Zugriff und Anwendung: Alias-/Clusterbezug,
Dialogreferenzen, Einheiten, Vergleiche und passende Zurückhaltung. Veränderliche
Lieferantenwerte bleiben in versionsgebundenen Pods. Readertraining soll die
Anwendung wechselnder bereitgestellter Werte lernen, keinen festen Müllerwert.

## Wichtige Präzisierungen

- LoRA ist nicht auf16Bit beschränkt; unser bestehender Planer trainiert FP32.
  QLoRA quantisiert die eingefrorene Basis typischerweise auf4Bit und trainiert
  Adapter mit höherer Präzision. Der gesamte Speicherbedarf sinkt nicht pauschal
  um exakt4x, da Aktivierungen, Gradienten und Optimizerzustand dazukommen.
- Ein Trainingsloss unter0.2 beweist kein Overfitting. Entscheidend ist die
  Leistung auf getrennten Daten; Adressausgaben sind beispielsweise sehr kurz.
- Alpha halbieren bedeutet W + Delta/2. Das entspricht dem Mittel aus W und
  W+Delta. Es ist nicht (W+Delta)/2, was auch die Basisgewichte halbieren würde.
- Effektive Batchgröße umfasst bei mehreren Geräten zusätzlich world_size.
  Bei verschieden langen Assistantantworten muss die Loss-Normalisierung über
  die überwachten Tokens des gesamten Akkumulationsfensters stimmen. Seed und
  gleiche Batchgröße allein garantieren keine bitidentischen Trainingsläufe.
- Unser eigener Torch/PEFT-Loop übernimmt keine Unsloth-Fixes automatisch.
  Unsloth-spezifische Optionen werden nicht blind an PEFT übergeben.
- Ein trainierter Reader hat eine neue Modellidentität. Seine Kapseln müssen
  mit dieser Identität neu kompiliert und die Trainingsherkunft angebunden werden.

Hierfür ist `reader_identity.py` vorbereitet: Gewichte allein reichen nicht,
weil geänderte LoRA-Skalierung oder deaktivierte Adapter dieselben Tensorbytes
behalten können. Ein Test mit echtem kleinen Qwen/PEFT-Modell bestätigt,
dass diese Zustandsänderungen unterschiedliche Identitäten erhalten. Diese
Hilfsfunktion ist noch nicht in die bisherigen Kapselläufe eingebaut und
ersetzt keine umfassende Ausführungsumgebungs-/ABI-Prüfung.

## Quellen

- [Unsloth-Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- [PEFT LoRA-Referenz](https://huggingface.co/docs/peft/en/package_reference/lora)

Nächste Reihenfolge: laufenden integrierten Test und Audit abschließen,
kontrollierte Sprach-/Einheitendiagnose ausführen, anschließend den allgemeinen
Reader-Trainingsdatensatz und den Trainer mit getrennten Entwicklungs-/Testdaten
aufbauen. BF16-Trainingsspeicher muss vor Start gemessen werden; bereits die
Inferenz benötigt auf diesem Rechner starkes Paging.
