# Training-Preflight und Evaluationsbefund

## Ergebnis

Der Datensatz ist strukturell valide, aber der Gesamtordner ist **kein direkt trainierbares Einzel-Causal-LM-Dataset**. Er ist ein kanonischer Multi-Track-Datensatz und muss vor jedem Lauf in Modellspuren partitioniert werden.

Automatisiert erzeugt wurden die Views unter `runs/pod-training-views`:

| Modellspur | Train/Dev/Test | Richtiger Trainer |
|---|---:|---|
| FunctionGemma-270M | 50/50/50 | Tool-call SFT + JSON-Schema-Gate |
| Qwen3-4B-SFT | 50/50/50 | Chat-SFT, grounded/citation |
| Qwen3-SFT | 100/100/100 | Link- und Multi-hop-SFT |
| Qwen3-GRPO | 100/100/100 | Rollout + GRPO/GSPO |
| Zephyr-DPO | 50/50/50 | Chosen/rejected DPO/ORPO/KTO |
| LoftQ-QAT | 50/50/50 | Calibration + Quantisierung |
| PaddleOCR-1B | 50/50/50 | Vision/OCR-SFT |
| MoE-experts | 50/50/50 | Router-/Expertentraining |
| all | 100/100/100 | Symbolischer Lifecycle-Gate + Eval |

## Blockierende Befunde

1. **Spurmischung:** Ein gemeinsamer Causal-LM-Trainer würde Tool-, OCR-, Embedding- und Quantisierungsziele vermischen.
2. **Falsche Modellbindung:** Die geerbte Config referenziert `qwen3b` bzw. Qwen2.5-3B, während die neuen Tracks Qwen3-4B/SFT/GRPO voraussetzen. Jeder View braucht einen exakt gepinnten Base-Checkpoint und Hash.
3. **Falsches Eingabeformat:** `assistant_only_loss=true` setzt Chat-/Message-Daten voraus; die kanonischen Rows enthalten aktuell `question`/`target` und müssen zuerst in Chat-Samples konvertiert werden.
4. **GPU-Nutzung:** Die Config setzt `world_size=1`; dadurch würden nicht automatisch beide Karten verwendet.
5. **Quantisierung/Vision:** LoftQ/QAT und PaddleOCR dürfen nicht durch den Reader-Trainer laufen.

## Remote-Preflight

Auf `xrey@192.168.1.223` wurden geprüft:

- 2 × NVIDIA RTX 3090, jeweils 24,576 MiB VRAM;
- jeweils etwa 24,1 GiB frei;
- PyTorch `2.11.0+cu128`;
- Transformers `5.17.0`;
- 2 CUDA-Geräte sichtbar;
- BF16 wird vom Stack unterstützt.

NVFP4 wird für diese Ampere-Karten nicht als Trainings- oder Serving-Annahme verwendet. Dafür bleiben BF16/QLoRA/LoftQ und gegebenenfalls GGUF die passenden Wege.

## Sichere Ausführungsreihenfolge

1. Kanonische Rows validieren und in Views partitionieren.
2. Pro View Chat-/Tool-/Pair-/Vision-Format erzeugen.
3. Base-Modell, Revision, Tokenizer, Dataset-Hash und Seed einfrieren.
4. Single-GPU Smoke-Run mit wenigen Schritten und Peak-VRAM-Messung.
5. Erst danach Multi-GPU-Training aktivieren.
6. Adapter-Hash, Base-Hash und Optimizer-/Scheduler-Konfiguration speichern.
7. Dev zur Diagnose verwenden, Test erst nach der Modell-/Checkpointwahl.
8. Für RL zusätzlich Rollout-Server, Reward-Trace, Hidden-Holdout und Anti-Hacking-Gates prüfen.
9. Nur ein einzelnes neues Artefakt über den Lifecycle-Gate aktivieren.

Die Preflight-Prüfung kann reproduzierbar ausgeführt werden:

```powershell
python research/validate_pod_model_curriculum.py runs/reader-training-inputs-pod-model-curriculum
python research/prepare_pod_training_views.py runs/reader-training-inputs-pod-model-curriculum runs/pod-training-views
python research/preflight_pod_training.py runs/reader-training-inputs-pod-model-curriculum
```

Der letzte Befehl muss beim gemischten Gesamtordner blockieren. Das ist beabsichtigt und verhindert einen formal erfolgreichen, aber wissenschaftlich unbrauchbaren Lauf.
