# Qwen2.5-3B LoRA Generation-4 — 2026-09-17

## Hypothese

Die Gen-3-Fehleranalyse zeigte: 40 von 132 Guard-Fehlfällen pro Split stammen
aus den Konzept-Familien `model_pod_generation2` (32) und `typed_topic` (8);
alle 40 Ziel-Sätze existieren im Training, die Trainingsfrage-Abdeckung dieser
Familien war aber nur 20 von 404 Zeilen. Gen-4 übersampelt diese Familien 3×
(`research/prepare_generation4.py`), dev/test bleiben byte-identisch
(eingefroren, fairer A/B gegen Gen-3).

## Training

| Item | Result |
|---|---:|
| Train rows | 464 (60 Oversamples) |
| Optimizer updates | 58 (2 epochs, effektiv batch 16) |
| CUDA peak | 8.16 GB |
| Preflight | bestanden; base hash unverändert |
| Adapter reload identity | passed (test + dev) |

## Ergebnis auf den eingefrorenen Splits (identische Cases wie Gen-3)

| Metrik | Gen-3 Adapter | Gen-4 Adapter | Base |
|---|---:|---:|---:|
| Test guarded exact match | 92/132 | **92/132** | 84/132 |
| Test raw exact match | 106/132 | **113/132** | 1/132 |
| Dev guarded exact match | 92/132 | **92/132** | 84/132 |
| Dev raw exact match | 106/132 | **119/132** | 2/132 |

## Ehrliche Interpretation

- **Gewinn:** wörtliche Reproduktion (raw exact match) stieg auf beiden Splits
  deutlich (+7/+13). Das Gate prüft `lora_ab_gen4[_dev]`: raw > Gen-3 UND
  guarded ≥ Gen-3 (kein Regress).
- **Keine Bewegung beim Guard:** die 40 Fehlfälle sind exakt dieselben Cases
  wie in Gen-3. Ursache: die dev/test-*Fragen* sind Paraphrasen ohne
  ableitbare Regel — das Modell lernt Ziel-Sätze für die *Trainings*-Fragen
  (daher raw ↑), generalisiert aber nicht auf neue Frageformulierungen.
  Mehr Wiederholungen derselben Trainingszeilen ändern daran nichts
  (empirisch belegt durch diesen Lauf).
- **Nächster Hebel (B4b):** Frage-Paraphrasen-Synthese im Training — zu jedem
  Ziel-Satz mehrere regelbasierte/LLM-generierte Frage-Varianten, damit die
  Mapping-Vielfalt gelernt wird statt der einzelnen Formulierung.

Artifacts on the server:

- `runs/reader-training-inputs-generation4/` (frozen inputs, hashes in protocol.json)
- `runs/qwen3b-lora-generation4-20260917/adapter/`
- `runs/qwen3b-eval-{test,dev}-gen4-adapter-20260917[-report].json`
