# Neural Pods â€“ VerÃ¶ffentlichungsset

Dieses Set ist der erste vorzeigbare Satz:

1. [LinkedIn-Beitrag](linkedin-post.md) â€“ kurz, sachlich, verÃ¶ffentlichungsfertig.
2. [PrÃ¤sentation](neural-pods-presentation.html) â€“ im Browser Ã¶ffnen; mit Pfeiltasten navigieren.
3. [Mini-Whitepaper](neural-pods-mini-whitepaper.md) â€“ Architektur, Messwerte, Grenzen und Reproduktion.
4. [Landingpage](index.html) â€“ kompakter Einstieg mit allen Downloads.
5. PDF-Versionen â€“ fÃ¼r Beitrag, Whitepaper und PrÃ¤sentation.

Das teilbare Paket ist [neural-pods-public-pack.zip](neural-pods-public-pack.zip).
Es enth?lt ausschlie?lich diese 50 redaktionellen Dateien und keine Modelle,
SSH-SchlÃ¼ssel oder privaten Archive.

Die Zahlen stammen aus den gespeicherten Reports im Projekt. Die wichtigste
Realmodell-Messung ist [Pod vs. RAG](../runs/multifact-internal-lora-004/pod-vs-rag-throughput.json).
Das Gesamt-Gate ist [20/20](../runs/project-gate-001.json).

Die vollstÃ¤ndige Abgrenzung von verifiziert, teilweise verifiziert und offen
steht in [COMPLETION-AUDIT-20260916.md](../research/COMPLETION-AUDIT-20260916.md).

Vor einer Ã¶ffentlichen VerÃ¶ffentlichung sollte die nÃ¤chste Version zusÃ¤tzlich
eine grÃ¶ÃŸere QualitÃ¤tsmessung gegen optimiertes RAG enthalten. Die aktuelle
Fassung macht diese Grenze ausdrÃ¼cklich sichtbar.

Die aktuelle Version enthÃ¤lt auÃŸerdem die leakage-kontrollierte Generalisierungsbenchmark und die Qwen3B-Baseline-, Boundary- und negativen Fortsetzungsreports.

Die Referenzintegration fÃ¼r rekursive Exploration liegt in
[RSI-POD-INTEGRATION.md](RSI-POD-INTEGRATION.md) mit dem reproduzierbaren
[Replay-Demo-Report](recursive-pod-loop-001.json).
Der angeschlossene lokale Search-Agent ist in [recursive-search-agent-001.json](recursive-search-agent-001.json) vermessen.

Der Der paper-kompatible Benchmark-Plan liegt in
[paper-benchmark-manifest-001.json](paper-benchmark-manifest-001.json). Er
enthÃ¤lt OSWorld-V2 (108 Aufgaben) und ALE Near-term (67 Aufgaben), jeweils mit
Baseline-/RSI-Armen und eingefrorenem Test-Memory. Der aktuelle Status ist
`prepared_not_run`, weil die benÃ¶tigten Docker/QEMU-VMs auf Windows und
`xrserver-dev` nicht vorhanden sind.

Die GPU-Messung direkt auf `xrserver-dev` liegt in
[qwen-gpu-xrserver-001.json](qwen-gpu-xrserver-001.json): RTX 3090, 22,48
generierte Token/s bei Batch 4.

Der vollstÃ¤ndige PrÃ¼fablauf ist `research/run_all_evaluations.py`; der
kombinierte Report ist `combined-evaluation-001.json`.

Der aktuelle Remote-Status und die reproduzierbaren Voraussetzungen stehen in [PAPER-BENCHMARK-STATUS-20260916.md](PAPER-BENCHMARK-STATUS-20260916.md).

Die NeoHorse-Referenzintegration und der echte Checkpoint-Probebericht liegen in [NEOHORSE-REFERENCE.md](NEOHORSE-REFERENCE.md) und [neohorse-real-checkpoint-probe-001.json](neohorse-real-checkpoint-probe-001.json).












