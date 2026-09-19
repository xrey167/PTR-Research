# Dream-Pod-Design: RSI durch Träumen über die Generations-Historie — 2026-09-20

Angelehnt an Dream-RSI (Zheng et al. 2026, „Recursive Self-Improvement
through Evolving Worlds"): **Die angesammelte Discovery-Historie ist ein
Replay-Simulator.** Abgeschlossene Verbesserungsprozesse zeichnen einen
strukturierten Baum von Entscheidungen und ihren realisierten Ergebnissen
auf; Kandidaten-Strategien können diesen Baum anders durchlaufen und
werden dadurch **off-policy** bewertet — ohne erneuten Trainingslauf.

## Die Übersetzung auf neural-pods

| Dream-RSI | neural-pods |
|---|---|
| Discovery-Prozess | Ein Generation-Lauf (Gen-3→4→5→6) |
| Entscheidungsbaum | Curriculum-Entscheidungen je Generation (Oversampling-Familien, Faktoren, Anker) — dokumentiert in `protocol.json` + `prepare_generation*.py` |
| Realisierte Outcomes | Per-Case-Ergebnisse in jedem Eval-Report (`report.json.rows`: guarded/raw je Case) → Familien-Deltas je Entscheidung |
| History-Pool | `research/runs/` + `runs/*-report.json` aller Generationen |
| Replay-Simulator | Familien-Effekt-Modell aus den (Entscheidung → Delta)-Paaren der Historie |
| Dreaming | Kandidaten-Curriculum-Strategien werden gegen den Replay-Simulator bewertet (off-policy, keine GPU) |
| Redeploy | Beste Strategie → echter Trainingslauf (Gen-7) → neues Outcome erweitert den Pool |

**Der Kreislauf schließt sich:** Online (Gen-N-Training) erweitert den
Pool; Träumen (Replay über den Pool) verbessert die Curriculum-Politik;
die verbesserte Politik wird als Gen-(N+1) eingesetzt.

## Pod-Vertrag

- **Dream-Pod** hält den History-Pool + Replay-Simulator; das Hauptmodell
  erreicht es über den Reflex-Kanal (`invoke("dream", {...})`): „Welche
  Curriculum-Strategie soll Gen-7 verwenden?"
- **Registry:** jeder Zyklus ist ein `dream_cycle`-Artifact mit Eltern
  (Pool-Hash, Politik-Kandidaten, Gewinner) — nachvollziehbar, rollback-
  fähig
- **Executor:** CPU/RAM-Lease (Replay ist Datenauswertung, keine GPU)

## Sicherheitsregeln

1. **Frozen-Split-Tabu:** Träume/Curricula berühren dev/test nie;
   Leakage-Dedup per Hash bleibt hart
2. **Träume sind off-policy Bewertungen, keine Evidenz** — erst der echte
   Trainingslauf (online) erzeugt belastbare Outcomes; das Gate entscheidet
   über Promotion wie immer
3. **Pool-Integrität:** der History-Pool ist append-only (Provenance-Kette);
   Policies werden nie auf zukünftige Outcomes trainiert
4. **Zyklen-Budget** im Event-Log (Autonomie-Budget, Pod-Arm-Design)

## Umsetzungsphasen

| Phase | Inhalt | Gate-Check |
|---|---|---|
| D1 | `neural_pods/dream.py`: HistoryPool (Per-Case-Outcomes aller Generationen einsammeln), Replay-Simulator (Familien-Effekte), Policy-Bewertung | — |
| D2 | `research/run_dream_cycle.py`: Zyklus auf dem Server — Pool bauen, Policies träumen, beste Strategie → Generation-7-Curriculum | `dream_pipeline` |
| D3 | Reflex-Bindung `dream` + Hauptmodell-Demo | `dream_reflex` |
| D4 (später) | Feineres Politur-Raum: Entscheidungen auf Zeilen-Ebene statt Familien-Ebene; MCTS über dem Entscheidungsbaum (vollständiges Dream-RSI) | `dream_deep` |

Akzeptanz D1–D3: kein bestehender Check bricht; der Simulator reproduziert
die beobachteten Familien-Deltas der Historie (Backtest), die beste
geträumte Strategie erzeugt ein vorbereitetes Gen-7-Input-Verzeichnis.


## Status (2026-09-20)

- **D1 umgesetzt:** `neural_pods/dream.py` — HistoryPool (Per-Case-Outcomes
  der Generationen gen3–gen6), ReplaySimulator (linearer Familien-Effekt
  mit learned limits), `backtest()` als Faithfulness-Prüfung. 3 Tests.
- **D2 umgesetzt:** `research/run_dream_cycle.py` auf der echten Historie:
  Pool = gen3–gen6, Backtest mean_abs_error 0,01 / max 0,057, Gewinner-
  Strategie für Gen-7: **concept_oversample 3 + lookup_anchor 2**
  (Vorhersage typed 0,93 / concept 0,98). Messdatei:
  `research/runs/dream-cycle-20260920.json`. Gate-Check `dream_pipeline`.
- Offen: D3 (Reflex-Bindung des Dream-Pods), D4 (generatives Träumen /
  zeilenfeine Politik-Räume), und der eigentliche Gen-7-Online-Lauf, der
  die geträumte Strategie als Evidenz validiert.
