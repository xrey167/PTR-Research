# Pod-/Modell-Zweck und Pipeline-Matrix

## Grundentscheidung

Ein Pod ist eine versionierte Fähigkeit mit stabiler Identität. Das Modell ist austauschbar. `pod_id` bleibt gleich, `generation` und `artifact_id` ändern sich. Retrieval-Metadaten bestimmen, welche Pods Kandidaten sind; der Lifecycle-Gate entscheidet, ob die gewählte Generation tatsächlich inferieren darf.

## Matrix

| Pod | Zweck | Daten | Pipeline | Primäre Metriken | Empfehlung |
|---|---|---|---|---|---|
| Router/Function | Intent, Tool- und Pod-Auswahl | Schema-validierte Tool Calls, ungültige Calls, unbekannte Felder | SFT auf FunctionGemma/Qwen; JSON-/Schema-Validator; kleine DPO-Stufe | Tool-/Argument-Exact-Match, unbekanntes Tool abgewiesen, Latenz | klein, deterministisch, keine langen Antworten |
| Reader | Antwort aus freigegebenen Evidenzen | Frage, Evidenz, Zitat, Generation | Qwen SFT, completion-only; danach optional DPO | grounded exact match, Zitat- und Generationstreue | Standardantwort-Pod |
| Research/Multi-hop | Ketten über Knowledge-/Provenance-Objekte | Chain traces, fehlende/revokierte Hops, Gegenbeispiele | SFT für Schema/Strategie, danach GRPO für messbare Ketten | Hop completion, provenance violation, final-answer accuracy | getrennt vom Reader trainieren |
| Parallel Search | Fan-out, Join, RRF, Barriers | unabhängige Branches, Abhängigkeiten, QPS-/Latenzspuren | SFT für Planformat, GRPO/GSPO mit Recall+Latenz+Anti-Hacking | recall@k, p95, parallel efficiency, reward hacking rate | nur mit robustem Reward |
| Link/Transport | Pod-zu-Pod-Aufruf | Attestation, ACL, Generation, Deadline, Hop budget, visited | SFT auf Handshake; negative DPO-Beispiele; Runtime hard gate | stale/revoked rejection, cycle rejection, ACL violations | SSH zuerst, WireGuard/mTLS später |
| Embedding/Retriever | semantische Pod-/Alias-Geometrie | query-positive, hard-negative, alias/entity/type/tag pairs | SentenceTransformer/Embedding-LoRA; eigener Index | Recall@10, MRR/nDCG, link-resolution | eigenes Modell/Namespace, keine Qwen-Text-LoRA |
| Reranker | genaue Top-k-Neubewertung | query/document relevance labels, provenance/ACL negatives | Cross-encoder LoRA oder klassischer Reranker | nDCG@10, MRR, p95 | nach ANN/BM25, vor Reader |
| Preference | aktuelle, belegte, zulässige Antwort bevorzugen | chosen/rejected mit stale, unsupported, ungrounded | DPO/ORPO/KTO | preference accuracy, citation validity | nur klare Präferenzen verwenden |
| OCR/Vision | Dokumente in strukturierte Evidenz umwandeln | page, bbox, text, confidence, origin, redaction | OCR/vision SFT; lokale Anonymisierung | OCR accuracy, field F1, redaction recall | vor externen Modellen anonymisieren |
| Math/Calculator | exakte Berechnung und Validierung | Ausdruck, Tool-Aufruf, Ergebnis, Einheiten | Tool SFT; kein freies Raten; ggf. DPO | exact numeric, unit correctness | deterministisches Tool bevorzugen |
| Model/Quant | Speicher-/Latenzoptimierung | calibration only + unseen holdout | LoftQ/QAT; BF16 source; GGUF/NVFP4 serving branch | task delta, KLD, memory, throughput | NVFP4 nur auf Blackwell; 3090: QLoRA/GGUF |
| MoE/Expert | Routing auf spezialisierte Experten | capability labels, load, expert traces | Router SFT/RL; capacity/load balancing | route accuracy, overflow, utilization | nicht mit Knowledge-Pods vermischen |
| Lifecycle/Evaluation | Aktivierung, Revocation, Vergleich | manifests, hashes, protocol snapshots | kein normales SFT als Wahrheitsquelle; regelbasierter Gate + eval harness | stale/revoked block, reproducibility | Gate bleibt symbolisch erzwingbar |

## Welche Trainingsart wann

1. **Continued pretraining**: nur große, bereinigte Domänentexte oder neue Sprache; keine Labels und keine Testfragen. Es verändert die Basisverteilung und ist deshalb ein neuer Base-Checkpoint.
2. **SFT**: Verhalten, Ausgabeformat, Tool-/Link-Protokolle und strukturierte Ziele.
3. **Embedding-Finetuning**: ausschließlich Query-/Dokument-/Pod-Paare; eigener Encoder und eigener Vektorindex.
4. **DPO/ORPO/KTO**: wenn ein menschlich oder regelbasiert klar bevorzugtes Ergebnis existiert.
5. **GRPO/GSPO**: wenn ein überprüfbarer Reward existiert, etwa korrekte Hops, Recall, Latenz oder Tool-Effizienz. Immer Hidden-Holdout und Reward-Hacking-Checks.
6. **LoftQ/QAT**: nach funktionaler Konvergenz, mit separater Calibration-Split und ungeöffneter Qualitätsprüfung.
7. **Serving-Quant**: erst nach Parität; niemals das Quant-Artefakt als neue Trainingsquelle verwenden.

## Validierte Startkonfiguration für unsere RTX-3090-Karten

- Qwen-LoRA/QLoRA: `r=16`, `alpha=16` oder `32`, `dropout=0`, alle großen Attention-/MLP-Projektionen, effektive Batchgröße 8–16.
- BF16/FP16 testen; keine NVFP4-Annahme auf Ampere. NVFP4 ist für Blackwell dokumentiert.
- Embedding-Pod: kleiner Encoder zuerst (0.3B–0.6B), harte Negative aus benachbarten Pod-Typen; danach optional 4B.
- Router: kurze Sequenzen, Temperatur nahe 0 bei Evaluation, striktes JSON-Schema.
- RL: separate Rollout-Inferenz, begrenzte Tool-/Hop-Zahl, Reward-Trace speichern.

Die Empfehlungen folgen den dokumentierten Unsloth-Pfaden für Continued Pretraining, Tool Calling, Embedding-Finetuning, LoRA und NVFP4: [Continued Pretraining](https://unsloth.ai/docs/basics/continued-pretraining), [Tool Calling](https://unsloth.ai/docs/basics/tool-calling-guide-for-local-llms), [Embedding-Finetuning](https://unsloth.ai/docs/basics/embedding-finetuning), [LoRA-Hyperparameter](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide), [NVFP4](https://unsloth.ai/docs/basics/nvfp4).
