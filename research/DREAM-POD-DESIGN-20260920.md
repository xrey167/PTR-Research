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
| D3 | Reflex-Bindung `dream` + Hauptmodell-Demo | `dream_reflex` **rot**, s. u. |
| D4 (später) | Feineres Politur-Raum: Entscheidungen auf Zeilen-Ebene statt Familien-Ebene; MCTS über dem Entscheidungsbaum (vollständiges Dream-RSI) | `dream_deep` (offen) |

Akzeptanz D1–D3: kein bestehender Check bricht; der Simulator reproduziert
die beobachteten Familien-Deltas der Historie (Backtest), die beste
geträumte Strategie erzeugt ein vorbereitetes Gen-7-Input-Verzeichnis.


## Status (Stand nach dem Pod-Audit, 2026-09-20)

Zählstände stehen ausschließlich in `ARCHITECTURE-MASTER-20260920.md`.

- **D1 umgesetzt:** `neural_pods/dream.py` — HistoryPool, ReplaySimulator,
  `backtest()`. Seit dem Audit: der Pool ist **erzwungen append-only und
  chronologisch** (Sicherheitsregel 3 war vorher nur Prosa), `base_model`
  ist eine modellierte Dimension, mehrdeutige Übergänge werden nicht
  zugeordnet, Extrapolation wird per Vorgabe abgelehnt.
- **D2 umgesetzt:** `research/run_dream_cycle.py`. Seit dem Audit erfüllt der
  Runner die drei übrigen Vertragspunkte, die vorher fehlten: **RAM-Lease**
  über den `ResourceGovernor`, **Autonomie-Quote** aus dem Event-Log
  (`CycleBudget`, vor dem Zyklus geprüft) und ein **`dream_cycle`-Provenance-
  Event** mit Pool-Fingerprint, allen Kandidaten und dem Gewinner. Vorher
  hinterließ ein Zyklus nichts außer einer JSON-Datei.
- **D3 gebaut, Gate-Check rot:** `research/benchmark_dream_reflex.py` +
  Gate-Check `dream_reflex`. Gemessen wird die **Bindung**: deterministischer
  Gewinner über 200 Aufrufe, eine bewusste Fehladressierung zieht den Arm
  korrekt auf den Default-Pod zurück, Alias-Auflösung p95 **0,089 ms** gegen
  das Pod-Arm-Ziel < 5 ms.

  **Zwei Berichtigungen dazu (2026-09-20).** Erstens stand hier „p95 0,098 ms"
  — diese Zahl steht in keiner dream-reflex-Evidenzdatei; die Aufzeichnung
  sagt 0,089 ms. Zweitens war der Haken in der Tabelle oben falsch: der
  Gate-Check ist **rot**, und das zu Recht. Er prüft seit dem Pod-Audit
  `pool_source`, und in einem Klon ohne die Generationsberichte fällt der
  Benchmark auf einen **synthetischen Zwei-Generationen-Pool** zurück. Eine
  Latenz über einem Spielzeugpool ist keine Latenz über dem echten. Die
  Bindung ist fertig; der Beleg dafür, dass sie über der echten Historie
  trägt, fehlt bis zur Aufzeichnung auf dem Server.
- Offen: D4 (zeilenfeine Politik-Räume, MCTS über dem Entscheidungsbaum).

## Gen-7-Online-Lauf: was der Vergleich wirklich zeigt (berichtigt)

Die geträumte Strategie (concept_oversample 3 + lookup_anchor 2, NeoHorse-
Base) wurde real trainiert und auf den unveränderten frozen Splits
evaluiert:

| | Geträumt | Real (Evidenz) | |
|---|---:|---:|---|
| Test typed | 0,9318 → **82**/88 | **81**/88 | −1 Fall |
| Test concept | 0,9773 → **43**/44 | **44**/44 | +1 Fall |
| **Test raw gesamt** | **125** | **125** | 0 |
| Dev raw / guarded | — | 124 / 92 | |

**Die frühere Fassung dieses Abschnitts behauptete, der Simulator habe den
Ausgang „exakt vorhergesagt" und der Kreislauf sei „Ende-zu-Ende
validiert". Beides hält nicht.** Die Summe stimmte, weil sich zwei
gegenläufige Ein-Fall-Fehler aufhoben; **keine** der beiden
Familienvorhersagen war richtig. Dazu kommt:

- Die Zielgröße war bereits gesättigt: Gen-6 stand schon bei 125/132 raw,
  concept bei 44/44. Eine Vorhersage „etwa wie Gen-6" trifft dort fast
  zwangsläufig.
- Die Gewinnerpolitik setzte `lookup_anchor` auf 2 — eine Stufe, die die
  Historie **nie gezeigt** hat. Das war eine Extrapolation, und die
  damalige Limit-Tabelle erlaubte sie, obwohl der Docstring „never dream
  beyond it" versprach.
- Der damals zitierte Backtest-Fehler 0,01 konnte nicht fehlschlagen: bei
  einem Intercept plus einem Koeffizienten je Entscheidung und einer
  Entscheidungsänderung je Generation reproduziert der Fit seine eigenen
  Punkte. Der heutige Leave-one-generation-out-Backtest sagt für gen3–gen6:
  **0 von 4 Generationen out-of-sample vorhersagbar**, je mit benannter
  Ursache.

Was der Lauf zeigt: die *Mechanik* des Kreislaufs trägt — Historie →
Träumen → Vorhersage → Online-Lauf → Evidenz → Pool wächst. Was er **nicht**
zeigt: prädiktive Gültigkeit des Simulators. Dafür braucht es eine
Kennzahl, die Generationen noch unterscheidet (`reader_holdout_data.py`),
und mehr als vier Generationen. Belege:
`research/runs/dream-vs-evidence-20260920.json` (mit Korrekturblock) und
`research/STATE-DEEP-RESEARCH-20260920.md`, Abschnitt 4.
