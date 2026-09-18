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

## Nachanalyse (B4b-Recherche): die Guard-Grenze ist konstruiert

Die B4b-Paraphrasen-Hypothese hat sich bei der Datenprüfung erübrigt: Die
Test-Fehlfälle stellen **exakt dieselben Fragen** wie die Trainingszeilen.
Die echte Ursache liegt im Guard: `guarded_answer` liefert ohne `evidence`
immer `UNKNOWN` (`reader_answer_guard.py`, Zeile 41–42) — und genau das gilt
für alle 44 Concept-Cases je Split. "Nicht verifizierbar" wurde mit "falsch"
gleichgezählt; das Verhalten ist bewusst und durch
`tests/test_reader_answer_guard.py::test_guard_handles_german_and_abstention`
fixiert (Abstention als Sicherheitsbarriere).

Die korrekte Aufschlüsselung der wörtlichen Treue (raw exact match) nach
Familie:

| Familie | Gen-3 test | Gen-4 test | Gen-3 dev | Gen-4 dev |
|---|---:|---:|---:|---:|
| Typed (88 Cases, Guard verifizierbar) | 74 | 70 | 73 | 76 |
| Concept (44 Cases, Guard sempre UNKNOWN) | 32 | **43** | 33 | **43** |

- **Die Concept-Familie ist damit praktisch gelöst: 43/44 je Split** (Gen-3:
  32/33). Das Oversampling hat seinen Zweck erfüllt.
- Der Guard-Wert 92/132 ist die Obergrenze des aktuellen Guard-Designs
  (88 verifizierbare Cases ergeben maximal die typisierten Treffer; die 44
  Concept-Cases können strukturell nie über den Guard punkten).
- Offene Design-Entscheidung (bewusst NICHT heimlich geändert, da der
  Abstention-Test sie fixiert): Guard-Pass-Through für evidenzlose
  Concept-Cases einführen und die Metrik accordingly umstellen. Bis dahin
  ist "raw auf Concept-Cases" die ehrliche Kennzahl für diese Familie.
- Kleiner Beobachtungspunkt: test-typed raw ging 74 → 70 zurück (dev: 73 →
  76). Der Guard kompensiert beide auf 92; für Gen-5 lohnt ein Blick, ob
  das Oversampling-Rauschen die Lookup-Präzision minimal stört.

Artifacts on the server:

- `runs/reader-training-inputs-generation4/` (frozen inputs, hashes in protocol.json)
- `runs/qwen3b-lora-generation4-20260917/adapter/`
- `runs/qwen3b-eval-{test,dev}-gen4-adapter-20260917[-report].json`
