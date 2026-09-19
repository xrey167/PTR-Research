# Vergleich mit SID-1, Wilson Search Engine und Inkling-Small

## Zahlen

| System | Qualit?t | Latenz | Gr??enordnung | Einordnung |
|---|---:|---:|---:|---|
| SID-1 | Recall 0,77 (0,84 mit 4x) | 5,5 s/Frage | gro?e Trainingskorpora, 1k+ QPS | ver?ffentlichter Agentic-Search-Vergleich |
| Wilson Search Engine | keine direkte Recall-Zahl im Bericht | ~500 ms | 280 Mio. Dokumente, 3 Mrd. Embeddings | ver?ffentlichte Websuchmaschine |
| Unser adaptiver Pod-Agent | Recall 1,00, MRR 1,00 | 22,3 ms p50 | 2.000 synthetische Dokumente | lokale Referenzmessung |

## Was der Vergleich zeigt

Unsere Messung ist aktuell schneller und erreicht im Test 100 % Recall. Das ist wegen der deutlich kleineren k?nstlichen Aufgabe kein Beleg f?r eine ?berlegenheit. SID misst multi-hop Dokumentabdeckung auf gro?en Korpora; Wilson misst eine vollst?ndige Websuchmaschine; Inkling misst Sprach-/Multimodalf?higkeiten eines 276B-Parameter-MoE.

F?r einen belastbaren Vergleich m?ssen wir dieselbe Query-Suite mit mehreren Gold-Dokumenten, identischem top-k, identischen Suchbudgets und getrennten Messungen f?r Retrieval-Recall, Antwortqualit?t, p50/p95-Latenz, QPS und Kosten ausf?hren.

## Quellen

- [SID-1 Benchmark](https://turbopuffer.com/blog/reinforcement-learning-sid-ai)
- [Wilson Search Engine](https://blog.wilsonl.in/search-engine/)
- [Inkling-Small Evaluation](https://huggingface.co/thinkingmachines/Inkling-Small)
