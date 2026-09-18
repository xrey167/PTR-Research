# Qwen2.5-3B LoRA Generation-5 — 2026-09-19

## Hypothese

Der Gen-4/DPO-Vergleich zeigte: typed-Präzision und Concept-Wörtlichkeit
konkurrieren um den Gradienten-Anteil. Der Trainer shuffelt pro Epoch, also
ist der Familien-Anteil der Zeilen direkt der Signal-Anteil. Gen-5 mischt
beide Signale in ein Curriculum: Gen-4-Concept-Oversampling (3×) **plus**
eine 1× Anker-Kopie der Lookup-Familie (`research/prepare_generation5.py`).

## Training

| Item | Result |
|---|---:|
| Train rows | 824 (360 Lookup-Anker + 60 Concept-Oversamples) |
| Optimizer updates | 104 (2 epochs, effektiv batch 16) |
| Preflight | bestanden |
| Adapter reload identity | passed (test + dev) |

## Ergebnis auf den eingefrorenen Splits (identische Cases wie Gen-3/Gen-4)

| Metrik | Base | Gen-3 | Gen-4 | **Gen-5 (promotiert)** |
|---|---:|---:|---:|---:|
| Test guarded | 84 | 92 | 92 | **92** |
| Test raw | 1 | 106 | 113 | **119** |
| Test typed raw | — | 74 | 70 | **76** |
| Test concept raw | — | 32 | 43 | **43** |
| Dev guarded | 84 | 92 | 92 | **92** |
| Dev raw | 2 | 106 | 119 | **124** |
| Dev typed raw | — | 73 | 76 | **80** |
| Dev concept raw | — | 33 | 43 | **44** |

## Interpretation

- **Beide Signale gleichzeitig gehalten:** typed (76/80) liegt über Gen-3
  und Gen-4; concept (43/44) auf Gen-4-Niveau, dev sogar perfekt 44/44.
  Der Trade-off aus Gen-4 und DPO ist durch den gemischten Curriculum-
  Ansatz aufgelöst — kein DPO nötig.
- Raw exact match 119/124 ist der beste Stand aller Generationen.
- Guarded bleibt 92/132 (Guard-Design-Grenze: 44 Concept-Cases können
  strukturell nicht punkten, siehe QWEN3B-LORA-GEN4-Doku).
- Gate: `lora_ab_gen5` + `lora_ab_gen5_dev` grün (raw > Gen-4, guarded ≥
  Gen-4). Gen-5 ist der promoted Reader-Adapter.

Artifacts on the server:

- `runs/reader-training-inputs-generation5/` (frozen, hashes in protocol.json)
- `runs/qwen3b-lora-generation5-20260919/adapter/` (promotiert)
- `runs/qwen3b-eval-{test,dev}-gen5-20260919[-report].json`
