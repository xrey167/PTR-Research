# Zusätzliche Research-Integration

## Precision-RL

`Precision-RL` ist für die RL-Spur relevant, nicht für SFT oder Retrieval. Die Arbeit untersucht den Training-/Inference-Mismatch und stellt FP16-Unterstützung für GRPO/GSPO-artige Verfahren in VeRL/OAT bereit. Wir übernehmen daraus eine **Ablation**, keine pauschale Präzisionsumstellung:

1. identischer Seed, Rollout-Prompt, Reward und Hidden-Holdout;
2. BF16 als bestehende Baseline;
3. FP16 als zweiter Lauf mit Loss-Scale-/Overflow-Logging;
4. vergleichen: Reward-Kurve, KL/Importance-Ratio, NaNs, Rollout-Agreement und Endqualität.

Auf unseren RTX 3090 ist FP16 technisch möglich, aber nur dann besser, wenn der konkrete RL-Stack stabiler läuft. Die Entscheidung erfolgt aus Messdaten.

Quelle: [Precision-RL](https://github.com/sail-sg/Precision-RL)

## vLLM Speculators

Speculators trainiert einen kleinen Draft/Speculator, dessen Tokens der großen Verifier-LLM verlustfrei prüft. Das passt zu unserem **Serving- und Speed-Pod**, nicht zum Wissens-Pod:

- Verifier und Speculator müssen dieselbe Tokenizer-/Generation-Identität haben;
- Hidden-State-/MTP-Daten werden offline mit vLLM erzeugt;
- Speculator-Artefakt erhält eigenen `artifact_id` und hängt an derselben Modellgeneration;
- Erfolg wird über Acceptance Rate, tokens/s, p50/p95 und identische Ausgaben geprüft;
- bei schlechter Acceptance fällt der Runtime-Gate auf normales Decoding zurück.

Für unsere Qwen3-Spur sind EAGLE-3, DFlash, P-EAGLE und MTP die relevanten Kandidaten. Das Projekt unterstützt außerdem Multi-Node-Hidden-State-Transfer; das ist erst nach einem lokalen Smoke-Test sinnvoll.

Quelle: [vLLM Speculators](https://github.com/vllm-project/speculators)

## Think Before You Link

Die Arbeit zeigt, dass reine Popularitäts-Splits seltene und schlecht dokumentierte Entitäten verfehlen. Das übernehmen wir als **Link-/Entity-Eval**, nicht als blindes Trainingsziel:

- Rarity-Slices nach Dokumentation, Konnektivität und Sprachabdeckung;
- erster Retrieval-Call verpflichtend, danach adaptive weitere Suche;
- BM25, Embedding und No-Retrieval als kontrollierte Baselines;
- mehrsprachige Alias-/Entity-Paare und harte negative Kandidaten;
- gespeicherte, verlustfreie Tool-/Retrieval-Traces.

Damit messen wir, ob der Link-Pod wirklich seltene oder nur populäre Entitäten kennt.

Quelle: [Think Before You Link](https://github.com/neulab/think-before-you-link)

## DSperse / gezielte Verifikation

DSperse ist kein Trainingsrezept für uns. Das übertragbare Muster ist **selektive Verifikation**: Wir prüfen nicht jede interne Operation kryptografisch, sondern verifizieren gezielt die Grenzen mit dem höchsten Risiko:

- Pod-ID, Generation, Artifact-Hash;
- Provenienz-/ACL-Entscheidung;
- Retrieval-Ergebnis und Lifecycle-Transition;
- optional replizierte oder signierte Zwischenresultate bei Remote-Pods.

Das reduziert Verifikationskosten und stärkt den Pod-Link-Gate.

Quelle: [DSperse](https://huggingface.co/papers/2508.06972)

## Unsloth Packing

Packing wird nur für SFT/CPT-Views aktiviert, nicht für DPO/GRPO, bevor deren Collator- und Maskierungssemantik geprüft ist. Baseline und gepackter Lauf müssen auf demselben kleinen Datensatz exakt gleiche Loss-/Grad-Norm-Kurven liefern. Danach messen wir Tokens/s, Peak-VRAM und Padding-Anteil.

Quelle: [Unsloth: 3x faster training with packing](https://unsloth.ai/docs/blog/3x-faster-training-packing)

## NVIDIA Blueprints

Blueprints dienen als Integrations-/Serving-Referenz für NIM, NeMo Retriever, Agent-Governance, Streaming-RAG und Data-Flywheel. Wir übernehmen daraus Schnittstellen und Betriebsmetriken, aber keine proprietäre Abhängigkeit in den Kern-Pods:

- OpenAI-kompatible Pod-Endpunkte;
- getrennte Retriever-/Ranker-/Generator-Services;
- Traces, Modell-/Datenversionen und Rollback;
- GPU-/Latenz-/Kostenmetriken.

Quelle: [NVIDIA Blueprints](https://build.nvidia.com/blueprints)

## Neue Reihenfolge

1. Baseline ohne Packing/Speculator/FP16.
2. Packing-Ablation für SFT/CPT.
3. FP16-vs-BF16-Ablation für RL.
4. Rarity-/Multilingual-Link-Evaluation.
5. Speculator nur für den stabilen Verifier.
6. Selektive Attestation am Pod-Gate.
7. Blueprint-kompatibles Serving und Lasttests.

## NeMo Data Designer

Data Designer passt in die **Daten-Erzeugungs- und Validierungsschicht**, nicht in den Modell-Lifecycle. Es unterstützt abhängige Spalten, Seed-Daten, strukturierte/multimodale Generierung, MCP-Interaktionen sowie Python-/SQL-/LLM-Validatoren. Das ist nützlich für:

- kontrollierte Alias-/Entity-/Pod-Typ-Verteilungen;
- positive, harte negative und kontrastive Link-Beispiele;
- Tool-Call- und Pod-Link-Traces;
- seltene Entitäten und mehrsprachige Rarity-Slices;
- synthetische OCR-/Vision-Fälle mit Seiten/BBox/Redaction;
- gezielte Fehlerklassen wie stale generation, ACL-Verletzung und Reward-Hacking.

Wir verwenden dafür die lokale/selbstgehostete Variante oder lokale Provider. Private Originaldaten dürfen nicht an NVIDIA Build, OpenAI oder OpenRouter gesendet werden. Die generierten Zeilen müssen danach weiterhin durch unseren eigenen Provenienz-, Leakage-, JSON- und Lifecycle-Validator laufen. Telemetrie wird deaktiviert (`NEMO_TELEMETRY_ENABLED=false`).

Quelle: [NVIDIA NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner)
