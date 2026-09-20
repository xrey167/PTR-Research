# Neural Pods — Ist-Stand und Deep Research (2026-09-20)

Untersuchungsgegenstand: `xrey167/PTR-Research`, HEAD `d963123`
("F1-F4 foundation"), Branch `claude/cutten-state-research-t1dxtz`
(inhaltsgleich mit `main`).

Methodik: frischer Klon, keine Server-Zugriffe. Alles unten Behauptete ist
entweder aus dem eingecheckten Code/den eingecheckten Messdateien abgeleitet
oder in diesem Klon nachgemessen. Wo eine Aussage nicht prüfbar war, steht es
ausdrücklich dabei.

Die Abschnitte 1–11 beschreiben den Stand **bei `d963123`** und bleiben als
Befundlage unverändert stehen. Was davon auf diesem Branch inzwischen behoben
ist, steht in **Abschnitt 12**.

---

## 1. Ist-Stand in Zahlen

| Größe | Doku sagt | Hier gemessen | Quelle der Messung |
|---|---|---|---|
| Gate-Checks definiert | 40 (Master), 33 (README) | **40** | AST-Parse von `verify_architecture_gate.py` |
| Gate-Checks grün | 40/40 | **30/40** | `python3 research/verify_architecture_gate.py` |
| Tests | 305 (Master), 285 (README), 268 (HANDOVER) | **307 passed, 6 failed, 3 skipped (316 gesammelt)** | `pytest -q` in frischem venv |
| Commits | 33 | **40** | `git rev-list --count HEAD` |
| Module `neural_pods/` | 47+ | **57 Dateien, 7.401 LOC** | `wc -l` |
| Dateien in `research/` | 86+ | **218** | `ls` |

Die vier Zahlenangaben in README, HANDOVER und Master-Dokument widersprechen
einander und alle drei der Realität. Das ist für sich genommen harmlos, aber
es ist das erste Symptom des Musters, das dieser Bericht beschreibt: **die
Dokumentation beschreibt einen Zustand, den das Repository nicht enthält.**

### 1.1 Warum 30/40 statt 40/40

Die zehn roten Checks sind `gen6_hetero_pod`, `gen6_promoted_dev`,
`gen7_dream_validated`, `native_protocol`, `lora_ab`, `lora_ab_dev`,
`lora_ab_gen4`, `lora_ab_gen4_dev`, `lora_ab_gen5`, `lora_ab_gen5_dev`.

Sie scheitern nicht an einer Regression, sondern an Pfaden. Das Gate liest
diese Evidenz aus `<project_root>/runs/…`, und `runs/` steht in `.gitignore`.
Ein Teil derselben Dateien liegt gleichzeitig unter `research/runs/` und ist
eingecheckt — das Gate schaut aber nicht dorthin:

```python
comm_path    = project_root / "runs" / "native-comm-eval-20260920-report.json"   # ignoriert
dream_ev_path= project_root / "runs" / "qwen3b-eval-test-gen7-20260920-report.json"  # ignoriert
# vorhanden und eingecheckt:  research/runs/native-comm-eval-20260920-report.json
#                             research/runs/qwen3b-eval-test-gen7-20260920-report.json
```

Konsequenz: Das Gate ist auf genau einer Maschine grün — der, auf der die
Benchmarks zuletzt gelaufen sind. Es ist kein Zustand, den das Repository
transportiert.

### 1.2 Die sechs Testfehler

| Test | Ursache | Kategorie |
|---|---|---|
| `test_mesh.py` (4 Tests) | `TimeoutError` — kein MQTT-Broker erreichbar | Umgebung, aber **ohne `skipif`-Schutz** |
| `test_neohorse_reference.py` | `runs/neohorse-reference-manifest-001.json` fehlt | gitignorierte Fixture |
| `test_taxonomy_dataset.py` | `runs/taxonomy-routing-balanced-003.jsonl` fehlt | gitignorierte Fixture |

Zwei Tests können aus einem sauberen Klon **prinzipiell nicht** grün werden.
Die vier Mesh-Tests hart abbrechen zu lassen, ist inkonsistent zum Raft-Binding,
das sauber mit `pytest.importorskip` arbeitet.

---

## 2. Der strukturelle Befund: Das Gate misst Dateien, nicht das System

Das ist der wichtigste Punkt des Berichts, weil er allen anderen zugrunde liegt.

`verify_architecture_gate.py` enthält keinen einzigen Aufruf, der etwas
ausführt. Es lädt 25 JSON/JSONL-Dateien und vergleicht Zahlen darin gegen
Schwellen. Auch der Check, der am meisten nach Verifikation aussieht, ist nur
ein Feldzugriff:

```python
"tests": d["tests"]["passed"] >= 260,
```

`d` ist `research/runs/architecture-20260917.json`, **datiert auf 2026-09-17**,
Inhalt `{"passed": 268, "skipped": 1, "warnings": 15}`. Das Gate führt keine
Tests aus. Es liest eine drei Tage alte Zahl. Sechs weitere Checks
(`retrieval_recall`, `cache`, `authenticated_transport`, `raft_replication`,
`quorum`, `resource_release`) speisen sich aus derselben eingefrorenen Datei.

Damit ist die Sicherheitsregel 1 des Master-Dokuments — *„Promotion nie ohne
Gate"* — in ihrer Wirkung schwächer als sie klingt. Das Gate ist ein
**Regressions-Journal über aufgezeichnete Evidenz**, kein fail-closed
Verifikationslauf. Es kann eine Regression, die nach der Aufzeichnung
entstanden ist, konstruktionsbedingt nicht sehen. Wer die Messdateien nicht neu
erzeugt, bekommt beliebig lange grün.

Das ist kein Betrug und in Forschungscode auch nicht unüblich — aber die
Dokumentation verkauft es als etwas anderes, und das Projekt stützt seine
Promotionsentscheidungen darauf.

**Zweiter Befund derselben Art:** Der Master-Plan kündigt für F1–F4 die neuen
Gate-Checks `storage_facade` und `storage_l2_lance` an und setzt den Endstand
auf 42/42. Der Code für F3/F4 ist gelandet (`neural_pods/storage.py`), die
beiden Gate-Checks existieren nicht. Ebenso fehlen alle für H4/H5/H7 und S2–S5
geplanten Checks. Das Gate wächst nicht mit dem Code mit.

---

## 3. Prüfung der vier Leistungs-Kernaussagen

### 3.1 „TaskGraph: Speedup 2,59×" — tatsächlich ≈ 0,93×

Die Zahl kommt aus `neural_pods/taskgraph.py`:

```python
sequential_s += finished - started          # je Knoten, unter Konkurrenz gemessen
"speedup": round(sequential_s / max(wall, 1e-9), 2)
```

`sequential_s` ist die **Summe der Wanduhrzeiten der parallel laufenden
Knoten**. Sobald Knoten aufeinander warten, wächst jeder Summand um genau die
Wartezeit — die Metrik steigt also *mit* der Contention. Sie kann strukturell
kaum unter 1 fallen und ist keine Aussage über Parallelisierungsgewinn.

Die eingecheckte Messung `research/runs/taskgraph-20260920.json` belegt es:

| Knoten | Dauer (s) | Ende (relativ) |
|---|---:|---:|
| ingest_a | 0,141 | 0,141 |
| ingest_b | 0,241 | 0,241 |
| ingest_c | 0,341 | 0,341 |
| ingest_d | 0,441 | 0,441 |
| plan | 0,301 | 0,542 |
| verify | 0,201 | 0,642 |

Die Endzeitpunkte liegen exakt 100 ms auseinander — genau der
`REMOTE_DELAY_S = 0.1` des Responders. Der Responder verarbeitet also strikt
seriell: `MeshEndpoint` startet paho mit `loop_start()` (ein Netzwerk-Thread)
und ruft die Handler in `_on_message` **inline** auf; das `time.sleep(0.1)` im
Handler blockiert diesen einen Thread.

Die vier „parallelen" Ingest-Knoten sind also nicht parallel. Die echte
sequentielle Referenz ist 6 × 100 ms = 0,600 s; die gemessene Wanduhr ist
0,643 s. **Realer Speedup ≈ 0,93×.** Die 2,59 entstehen ausschließlich daraus,
dass die Warteschlangenzeit im Zähler mitgezählt wird
(0,141+0,241+0,341+0,441+0,301+0,201 = 1,666 ≙ `sequential_equivalent_s` 1,665).

Das Gate prüft `speedup > 1.5` — auf eine Metrik, die bei genügend Contention
jeden Wert annimmt.

### 3.2 „gemessene 2,81× Optimierung" im traced pipeline — Wanduhr echt, Stufenmetrik erfunden

Der Wanduhrvergleich (2,845 s → 1,012 s) ist legitim: Lauf 1 macht 132
sequentielle Round-Trips à 20 ms, Lauf 2 feuert alle Anfragen und sammelt ein.
Der Gewinn kommt aus Pipelining, nicht aus Parallelität — das ist ein
brauchbares Ergebnis.

Die *Stufenlatenzen* desselben Reports sind es nicht. In
`research/benchmark_traced_pipeline.py`:

```python
lookup_ms = (time.perf_counter() - start_collect_ms) * 1000 if False else None
tracer.record(trace_id, case, "lookup", -1, parallel=True, ...)
stage_stats["lookup"].append(0.05)   # amortized batch latency
```

Die im Report ausgewiesenen `stage_p50_ms.lookup = 0.05` und
`stage_p95_ms.lookup = 0.05` sind eine **hartkodierte Konstante**. Die echte
Zeitmessung ist per `if False` stillgelegt, und in die JSONL-Trace-Datei —
laut Docstring „a trace record per hop" — werden für den optimierten Lauf 132
Zeilen mit `latency_ms: -1` geschrieben. Der Vergleich „lookup 20,5 ms → 0,05 ms"
ist damit kein Messergebnis.

Zusätzlich: Der Docstring verspricht „Run 2 = warm (cache hits)". Einen Warm-Lauf
gibt es nicht; `main()` invalidiert den Cache vor beiden Läufen, und beide
Läufe im Report haben `cache_hits: 0`.

### 3.3 „Native Protokolle: 100 % Frame-Validität" — Syntax 100 %, Semantik 55 %

`research/runs/native-comm-eval-20260920-report.json`:

```json
{"frames_valid_rate": 1.0, "exact_rate": 0.55, "total": 80,
 "acl_refused_forbidden": true, "acl_violations": 1}
```

`frames_valid_rate` misst, ob das Modell syntaktisch parsebare Frames erzeugt.
`exact_rate` misst, ob es das **richtige** Frame erzeugt — 0,55. Der Gate-Check
`native_protocol` prüft `frames_valid_rate >= 0.98` und die ACL-Refusal, aber
`exact_rate` mit **keiner Schwelle**. Master-Dokument und Commit-Message führen
die 1,0 als Kernergebnis („dialect LoRA at 100% frame validity"), die 0,55 taucht
in keinem Dokument auf.

Auch `acl_violations: 1` steht unkommentiert im Report, während der Gate-Check nur
das Gegenstück (`acl_refused_forbidden`) abfragt.

### 3.4 „Mesh RTT p50 0,30 ms" — nicht knotenübergreifend

Die Messdatei sagt es selbst:

> „Presence discovery host↔container peer (np-node2); RTT **between two host
> endpoints** over the remote broker in np-node1"

Beide RTT-Endpunkte liegen auf demselben Host; nur die *Discovery* war
host↔container. Im Master-Dokument steht die Zahl in der Zeile
„Mesh (mesh.py, RTT 0,3 ms)" unter „NERVENSYSTEM" ohne diese Einschränkung.

Nebenbefund: Dieselbe Datei meldet `endpoint_stats.received: 4` bei
`rounds_sent/rounds_ok: 100/100`. Der Host-Endpunkt hat also 4 von 100
Antworten über seinen Envelope-Pfad gesehen, während 100 Runden als erfolgreich
gezählt werden. Die beiden Zähler messen offensichtlich Verschiedenes; das
„100/100" trägt weniger, als es suggeriert.

---

## 4. Der Dream-Pod: die Kernaussage des Projekts, genau geprüft

Die prominenteste wissenschaftliche Behauptung lautet (HANDOVER, README,
Master-Dokument, `dream-vs-evidence-20260920.json`):

> „geträumte Gen-7-Strategie … wurde online trainiert und die Vorhersage
> **exakt** bestätigt (125 predicted, 125 real). … Dream-RSI loop validated
> end to end."

Drei Prüfungen, alle mit negativem Ergebnis für die Stärke der Behauptung.

### 4.1 Der Backtest ist eine algebraische Identität, kein Test

`ReplaySimulator.simulate()` ist ein lineares Modell mit zwei Koeffizienten:

```
rate(familie) = baseline(familie) + Σ_d  response[d][familie] · level(d)
```

`baseline` ist die Rate von gen3. `response[d]` wird in `family_deltas()` aus
**genau einem** Generationenpaar geschätzt, in dem sich `d` geändert hat
(`step = delta / (level - prev_level)`), und bei mehreren Änderungen überschreibt
die spätere die frühere. Damit gilt konstruktionsbedingt:

- `simulate(gen3)` = baseline = gen3 → Fehler 0
- `simulate(gen4)` = gen3 + 3·(gen4−gen3)/3 = gen4 → Fehler 0
- `simulate(gen5)` = gen4 + 1·(gen5−gen4)/1 = gen5 → Fehler 0

Nur gen6 kann einen Fehler haben — und zwar genau deshalb, weil sich zwischen
gen5 und gen6 das **Basismodell** änderte (`{"base_model": "neohorse"}`), was
`family_deltas()` gar nicht als Entscheidung kennt und deshalb komplett ins
Residuum schiebt.

Nachgestellt in diesem Klon mit frei gewählten synthetischen Raten, gleiche
Entscheidungsstruktur (`scratchpad/dream_algebra.py`):

```
gen3 typed    0.7045  ->  0.7045   |Fehler| 0.0000
gen3 concept  0.2045  ->  0.2045   |Fehler| 0.0000
gen4 typed    0.7955  ->  0.7955   |Fehler| 0.0000
gen4 concept  0.5455  ->  0.5455   |Fehler| 0.0000
gen5 typed    0.8636  ->  0.8636   |Fehler| 0.0000
gen5 concept  0.9773  ->  0.9773   |Fehler| 0.0000
gen6 typed    0.9091  ->  0.8636   |Fehler| 0.0455
gen6 concept  0.9318  ->  0.9773   |Fehler| 0.0455
backtest(): {'generations': 4, 'mean_abs_error': 0.0114, 'max_abs_error': 0.0455}
```

Sechs von acht Residuen sind **exakt** null, unabhängig von den Daten. Der
publizierte `mean_abs_error: 0.01` ist deshalb keine Aussage über
Vorhersagegüte; er ist der durch acht geteilte gen6-Fehler. Der Gate-Check
verlangt `max_abs_error < 0.15` — eine Schwelle auf einer Größe ohne
Informationsgehalt.

Nebenbei: README/HANDOVER/Master zitieren „Backtest-Fehler 0,01" — das ist der
**Mittelwert**; der Maximalfehler in der Datei ist 0,0569.

### 4.2 Die „exakte" Gen-7-Vorhersage sind zwei Fehler, die sich aufheben

`dream-vs-evidence-20260920.json`, vollständig gelesen:

| | vorhergesagt | tatsächlich | Differenz |
|---|---:|---:|---:|
| typed | 0,9318 → 82/88 | 0,920 → **81**/88 | −1 Fall |
| concept | 0,9773 → 43/44 | 1,000 → **44**/44 | +1 Fall |
| **Summe raw** | **125** | **125** | **0** |

Die Vorhersage war in **beiden** Familien falsch, je um einen Fall, in
entgegengesetzte Richtungen. Der Gesamttreffer ist die Differenz zweier Fehler.
Der Report notiert die Einzelfehler korrekt (`typed_abs_error: 0.0113`,
`concept_abs_error: 0.0227`) und schließt trotzdem: *„The replay simulator
predicted the Gen-7 test raw total exactly (125). The dream became evidence."*
Die Schlussfolgerung benutzt die eine Zahl, die durch Fehlerkompensation
zustande kam, und ignoriert die beiden, die den Modellfehler zeigen.

### 4.3 Die Vorhersage war zudem extrapolativ und auf ein gesättigtes Ziel gerichtet

- `ReplaySimulator.limits` erlaubt `lookup_anchor` bis 2, beobachtet wurden nur
  die Stufen 0 und 1. Der Docstring sagt „Per-decision limits learned from
  history (never dream beyond it)" — genau das tut die Gewinnerpolitik
  (`lookup_anchor: 2`).
- Zielmetrik gesättigt: gen6 lag bereits bei 125/132 raw (94,7 %), concept bei
  44/44 (100 %). `simulate()` klemmt nach oben (`min(rate, 1.0)`), nach unten
  nicht. Eine Vorhersage „etwa so gut wie gen6" trifft hier fast zwangsläufig.
- Der Gate-Check `gen7_dream_validated` verlangt `>= 125` und `>= 92` — also
  **Nicht-Regression gegenüber gen6**, nicht Verbesserung. Gen-7 erreicht exakt
  die gen6-Werte (test raw 125, guarded 92, `first_logits_sha256` sogar identisch).

### 4.4 Weitere Mängel in `dream.py`

| Stelle | Befund |
|---|---|
| `_family(case_id, evidence)` | Parameter `case_id` wird nie benutzt |
| `simulate()` | klemmt nur nach oben; negative Raten möglich |
| `family_deltas()` | `max(level - prev_level, 1)` verfälscht Vorzeichen/Betrag bei sinkenden Stufen |
| `from_project()` | überspringt fehlende Generationen still (`continue`), verlangt nur `>= 2` — die Poolzusammensetzung hängt davon ab, welche `runs/`-Verzeichnisse auf der Maschine liegen |
| gesamt | Alle vier Quellpfade liegen unter `runs/` (gitignored) → der Dream-Zyklus ist aus dem Repo nicht reproduzierbar |

**Fazit zu Abschnitt 4:** Der Dream-Pod ist als *Idee* — Curriculum-Entscheidungen
aus Historie off-policy vorbewerten, statt jeden Kandidaten zu trainieren —
richtig und die Umsetzung ist sauber gekapselt. Als *validierter RSI-Kreislauf*
ist er nicht belegt. Die Evidenzlage trägt die Aussage „ein Zwei-Parameter-
Regressionsmodell auf vier Punkten hat eine gesättigte Kennzahl in der Summe
getroffen"; sie trägt nicht „Dream-RSI loop validated end to end".

---

## 5. Zustand der zuletzt gelandeten Phasen F1–F4

### 5.1 F1 `registry.events()` — erledigt, halb

Die öffentliche Lese-API existiert (`registry.py:228`) und der vorher
fehlschlagende `test_household.py` läuft. Die **Schreibseite** geht weiterhin
über die private `registry._event(...)`, u. a. aus `household.py`. Das ist die
API, an der die Auditierbarkeit hängt.

### 5.2 F3/F4 `PodStorage` + LanceDB-L2 — sechs belegte Defekte

Alle folgenden Punkte wurden in diesem Klon mit installiertem `lancedb`
reproduziert (`scratchpad/storage_probe.py`, `scratchpad/docs_probe.py`):

| # | Defekt | Beleg |
|---|---|---|
| B1 | `put()` schreibt die erste Zeile doppelt: `_ensure_table(name, [row])` legt die Tabelle **mit** der Zeile an, danach folgt `table.add([row])` | „Zeilen für k1 nach EINEM put(): **2**" |
| B2 | `get()` auf einen unbekannten Key **legt eine Müllzeile an** (`value: "null"`), weil derselbe `_ensure_table`-Pfad im Lesepfad benutzt wird | „Zeilen in kv-Tabelle nach reinem Lesezugriff: **1**" |
| B3 | SQL-Prädikate werden per f-String gebaut (`f"key = '{key}'"`) — kein Quoting | `ValueError: … Unterminated string literal` bei Key `O'Brien` |
| B4 | `search_documents()` auf leerem Namespace legt eine Dummy-Zeile mit `vector: [0.0]` an und ist danach kaputt | `ValueError: There is no vector column in the data` |
| B5 | `write_traces()` dupliziert ebenfalls den ersten Batch | „Trefferzahl für t1: **2**" |
| B6 | Annotation `lance_dir: str \| Path` — `Path` ist im Modul nie importiert | `typing.get_type_hints(...)` → `NameError: name 'Path' is not defined` |

Die Folge von B1/B5 ist nicht kosmetisch. Mit fünf echten Dokumenten:

```
nach add_documents(5 Zeilen):  Zeilen in docs_ns: 10
search_documents(top_k=2) -> Treffer: ['d1', 'd1']
```

**Die Vektorsuche liefert dasselbe Dokument mehrfach in den Top-k.** Jedes
Retrieval, das auf L2 aufsetzt, bekommt verfälschte Ergebnisse.

Die vorhandenen vier Tests in `tests/test_storage.py` decken den KV-Roundtrip,
den L1-Ausfall, Traces und die Session-Affinity ab — und **keinen** der sechs
Punkte. `add_documents`/`search_documents`, also der eigentliche Zweck des
L2-Tiers („Vektoren/Dokumente/Traces"), haben null Testabdeckung.

Weitere Beobachtungen: `get()` füllt L1 nach einem L2-Treffer nicht zurück; für
L2 gibt es kein `delete`, keine TTL, keine Kompaktierung; `lancedb` ist in
`requirements-lock.txt` **ungepinnt** eingetragen (in einer Datei, deren
einziger Zweck das Pinnen ist) — dasselbe gilt für `paho-mqtt`, `xgboost`,
`redis`. Die Deprecation-Warnung `table_names() is deprecated` zeigt, dass die
API-Kompatibilität bereits driftet.

### 5.3 `household.py` — BUSY wird nie freigegeben

| # | Defekt | Wirkung |
|---|---|---|
| H1 | Es gibt **keine** Methode, die `offer.busy` zurücksetzt (`grep busy` findet nur Setzen und Lesen) | Ein Donor ist nach der ersten Allokation dauerhaft BUSY; der Haushalt kann pro Donor genau eine Allokation ausführen |
| H2 | `start()` schreibt **kein** Provenance-Event | Genau die Handlung, die Ressourcen bindet, ist nicht auditiert; Request und Approval sind es |
| H3 | Der Klassen-Docstring behauptet „household state is reconstructible from the registry events" | BUSY-Status und Reservierungen stehen in keinem Event → nicht rekonstruierbar |
| H4 | `claimed[pod_id]` ist **ein** Zähler für VRAM *und* RAM | Ein in VRAM platziertes Segment belastet das RAM-Budget desselben Donors. Die Tier-Semantik von ADR-7 wird auf einen gemeinsamen Pool reduziert |
| H5 | `approve()` prüft den Household-Key nicht und läuft **außerhalb** `self._lock` | Jeder Aufrufer mit `request_id` + `pod_id` kann im Namen eines Donors zustimmen; nebenläufige Approvals können sich über `(*request.approved, pod_id)` gegenseitig verlieren |
| H6 | BUSY-Ablehnung liefert ein `AllocationRequest(request_id="BUSY", status="refused")` zurück und schreibt **kein** Event | ADR-6 („BUSY ersetzt stillen Override", Approvals auditierbar) ist auf der Ablehnungsseite nicht belegt |

H4 wird durch `test_allocation_requires_full_approval_and_respects_busy`
sogar festgeschrieben — der Test wählt Budgets, die großzügig genug sind, dass
die Konflation nicht auffällt.

H5/H7 (Training auf Donoren, DatasetStore) sind wie im Master-Dokument
angegeben reiner Entwurf; es existiert kein `request_training`, kein
`DatasetStore`.

---

## 6. Sicherheitsbild

Die acht Sicherheitsregeln des Master-Dokuments sind als Absicht gut gewählt.
Zwei davon halten der Implementierung nicht stand.

**Regel 6 „Egress-ACL je Pod".** Die Topic-ACL in `mesh.py` wird
**ausschließlich clientseitig** erzwungen: `publish()` und `subscribe()` rufen
`_topic_allowed()` auf, `publish_raw()` umgeht sie ausdrücklich per Docstring
(„the caller … is responsible for access control"). Der Broker erzwingt nichts.
Ein Pod, der die Regel nicht einhalten will, ruft `publish_raw` — oder redet
direkt MQTT.

**Regel 7 „Fail-closed Parser".** Die als „fail-closed Envelope-Validierung"
dokumentierte Prüfung in `_on_message` besteht aus JSON-Parse plus **einem
Integer-Vergleich**:

```python
if envelope.get("protocol_version") != self.protocol_version:
    raise ValueError("protocol version mismatch")
# Manifest hashes differ legitimately between pods … integrity is enforced
# per channel and by PodTransport, not by rejecting foreign hashes here.
```

`manifest_hash` und `principal` werden entgegengenommen, aber nicht geprüft.
Wer Broker-Zugriff hat, kann Envelopes mit beliebigem `principal` einspeisen.
Der „Lineage-Envelope" ist damit Metadaten-Transport, keine Authentifizierung.
Für ein LAN-Forschungssetup ist das vertretbar — als Sicherheitsregel
dokumentiert ist es irreführend.

Der Kommentar verweist auf „PodTransport" als eigentliche Durchsetzungsstelle;
diese Verlagerung ist nirgends als Gate-Check oder Test verankert.

---

## 7. Die inhaltlich wichtigste Stagnation: `guarded` steht seit Gen-3

| Generation | test raw | test guarded | dev raw | dev guarded |
|---|---:|---:|---:|---:|
| Base | 1 | 84 | 2 | 84 |
| Gen-3 | 106 | 92 | 106 | 92 |
| Gen-4 | 113 | 92 | 119 | 92 |
| Gen-5 | 119 | 92 | 124 | 92 |
| Gen-6 | 125 | 92 | 124 | 92 |
| **Gen-7** | **125** | **92** | **124** | **92** |

`guarded` steht seit Gen-3 auf 92/132 — über fünf Generationen, drei
Trainingsstrategien, einen Basismodellwechsel und einen DPO-Versuch hinweg.
Das HANDOVER erklärt das korrekt und ehrlich: 44 concept-Fälle können
strukturell nicht punkten, weil `guarded_answer` ohne evidence immer UNKNOWN
liefert.

Die Konsequenz wird aber nicht gezogen: **die Kennzahl, auf die das Gate
ratcht, ist für 44 von 132 Fällen konstruktionsbedingt blind, und die
Ersatzkennzahl `raw` ist bei 94,7 % gesättigt** (concept sogar bei 100 %).
Zwischen Gen-6 und Gen-7 ist auf keiner der beiden Kennzahlen eine Änderung
messbar; die Testreports beider Generationen teilen sogar denselben
`first_logits_sha256`.

Ein Verbesserungskreislauf, der auf einer gesättigten und einer teilblinden
Kennzahl optimiert, kann nicht mehr unterscheiden, ob er besser wird. Das ist
kein Implementierungsfehler, sondern das drängendste inhaltliche Problem des
Projekts.

---

## 8. Deep Research: Einordnung gegen den Stand der Technik

### 8.1 Multi-LoRA-Serving — der Ansatz ist richtig, die Skalenaussage fehlt

Der Pod-Gedanke (viele kleine, austauschbare LoRA-Wissenseinheiten auf
gemeinsamer Basis) deckt sich mit der etablierten Forschungslinie: **S-LoRA**
serviert 2.000 gleichzeitige Adapter über Unified Paging aus einem gemeinsamen
Speicherpool und erreicht gegenüber HuggingFace PEFT und naivem vLLM-LoRA bis
zu 4× Durchsatz; **Punica** batcht Anfragen über verschiedene Adapter per
SGMV-CUDA-Kernel auf einer GPU.

Das Projekt serviert derzeit zwei Basen auf zwei GPUs mit vLLM-Multi-LoRA. Die
README benennt die Grenze selbst („prüft weder 100–1000 Pods noch die
Überlegenheit gegen Produktions-RAG") — das ist vorbildlich. Wichtig für die
Planung: Der Sprung von „2 Adapter" auf „viele Adapter" ist in der Literatur
**kein Skalierungsproblem der Registry, sondern eines des GPU-Speicher-Pools
und des Batching-Kernels**. Die Vier-Tier-Storage-Architektur adressiert L1–L3,
aber nicht den Adapter-Pool auf der GPU — genau die Stelle, an der S-LoRA
ansetzt. Das ist die relevanteste offene Architekturfrage, und sie steht in
keinem der fünf Design-Dokumente.

### 8.2 ADR-3 (Mooncake / KV-Affinity) — richtig gelesen, nicht gemessen

**Mooncake** trennt Prefill- und Decode-Cluster und baut einen verteilten
KVCache über ungenutzte CPU-, DRAM- und SSD-Ressourcen; in Produktionstraces
59–498 % mehr effektive Requestkapazität unter SLO, im Betrieb über
100 Mrd. Token/Tag. Die Entscheidung von ADR-3, statt KV-Transfer nur
**Session-Affinität** zu implementieren („vLLM exponiert KV-Caches nicht stabil
transferierbar"), ist gegenüber dem 2026er Stand korrekt und konservativ.

Die Umsetzung (`SessionAffinity` in `storage.py`) ist 20 Zeilen, korrekt und
getestet — zählt aber nur `failovers`/`reuses` in einem Dictionary. Der
behauptete Nutzen („Affinity über Redis jetzt messbar") ist nicht gemessen: es
gibt keinen Benchmark, der Time-to-first-token mit und ohne Affinität
vergleicht. Der geplante Gate-Check `kvcache_affinity` existiert nicht. Der
Mooncake-Vergleich liefert dafür die naheliegende Messgröße: TTFT bzw.
Prefill-Anteil bei Session-Wiederkehr.

### 8.3 Dream-RSI gegen die RSI-Literatur

Die aktuelle Übersichtsliteratur trennt **bounded self-refinement** —
konvergent, evaluierbar, industriell im Einsatz — von offener **rekursiver
Selbstverbesserung**, die durch Grounding-Bedarf, Kollapsdynamik und Compute
begrenzt bleibt. Zwei Befunde dieser Literatur treffen den Dream-Pod direkt:

1. **Ohne externes Feedback können Modelle ihr eigenes Reasoning weitgehend
   nicht korrigieren**; naive Selbstkorrektur verschlechtert Antworten teils.
   Der Dream-Pod umgeht das im Prinzip richtig — er trainiert und evaluiert
   *wirklich* und nennt Träume ausdrücklich „off-policy estimates, never
   evidence". Das ist konzeptionell die richtige Absicherung. Nur wird der
   Simulator, der die Absicherung liefern soll, gegen sich selbst gemessen
   (Abschnitt 4.1).
2. **Benchmark-Sättigung:** sobald Modelle eine Benchmark konstant über 90 %
   lösen, unterscheidet sie nicht mehr (MMLU als Referenzfall). Genau dort
   steht das Reader-Eval (94,7 % raw, 100 % concept). Der Ausweg der Literatur
   ist der von LiveBench: laufend neue, nicht kontaminierte Fälle — hier also
   ein **frisch generierter, eingefrorener Split**, nicht mehr Curriculum auf
   den bestehenden 132 Fällen.

Die Diagnose ist damit: Der Dream-Pod ist kein RSI-Problem, sondern ein
Messproblem. Solange die Zielmetrik gesättigt ist, kann keine Curriculum-
Strategie einen Unterschied zeigen — und ein Simulator, der auf dieser Metrik
trainiert wurde, kann keinen vorhersagen.

### 8.4 Der „native Dialekt" gegen den Protokollstandard 2026

Das Nervensystem-Design trainiert eine 0,5B-Dialekt-LoRA, die MQTT/TCP-Frames
ohne Tool-Use erzeugt. Parallel hat sich die Standardlandschaft 2026
konsolidiert: **A2A** ist seit Januar 2026 auf v1.0.0 (Produktionsreife,
**signierte Agent Cards** zur kryptografischen Verifikation), **ACP** ist in
A2A aufgegangen, und MCP wie A2A liegen seit Dezember 2025 unter der
**Agentic AI Foundation** der Linux Foundation. Die Arbeitsteilung: MCP
vertikal (Agent → Werkzeug), A2A horizontal (Agent ↔ Agent) mit Agent Cards
als Discovery und Tasks mit definiertem Lebenszyklus.

Das Projekt hat beides unabhängig nachgebaut: Presence-retained-Topics
entsprechen Agent Cards, der TaskGraph entspricht dem A2A-Task-Lebenszyklus.
Zwei Schlüsse:

- **Der Dialekt-Ansatz bleibt begründbar**, weil er ein anderes Ziel hat als
  A2A: nicht Interoperabilität, sondern Wegfall des Tool-Use-Overheads. Das
  sollte aber explizit als ADR stehen, sonst wirkt es wie Unkenntnis des
  Standards.
- **Signierte Agent Cards sind die fertige Antwort auf den Befund aus
  Abschnitt 6.** A2A v1.0 löst genau das Problem, das `mesh.py` offen lässt
  (ungeprüfter `manifest_hash`/`principal`). Das Signaturmodell lässt sich
  übernehmen, ohne das Protokoll zu übernehmen.

Zur Einordnung gehört auch: Eine Untersuchung der Governance-Lücken von MCP,
A2A und ACP zeigt, dass auch diese Standards Zusicherungen wie Delegation und
Haftung nicht ausdrücken können — die Provenance-Registry des Projekts ist
hier kein Rückstand, sondern potenziell ein Vorsprung.

### 8.5 Provenance und Reproduzierbarkeit

Die Praxisliteratur zu ML-Lineage formuliert die Anforderung so: Lineage muss
**automatisch und unveränderlich** sein, und jeder Trainingslauf muss seine
Eingaben (Datensatzversion, Konfigurationshash, Basismodellreferenz), seine
Umgebung (Container-Digest, Bibliotheksversionen, Hardware) und seine Ausgaben
(Artefakt-Hash, Metriken) programmatisch protokollieren.

Der Kern des Projekts — SHA256-OriginKeys, `knowledge:v2:`-Identitäten,
Generationen, transitiver Widerruf, Commit-Prüfung in einer
SQLite-Schreibtransaktion — erfüllt diese Anforderung auf der *Wissensebene*
sehr gut und ist der stärkste Teil des Systems.

Auf der *Evidenzebene* gilt das Gegenteil, und es ist derselbe Punkt wie in
Abschnitt 1.1: Die Messdateien, auf denen jede Promotionsentscheidung beruht,
liegen in einem gitignorierten Verzeichnis auf einer Maschine. Ein
Repository-Klon kann keine einzige Kernmessung nachvollziehen. Das ist genau
das Anti-Pattern, das die Lineage-Literatur benennt.

Bemerkenswert: `research/runs/` ist bereits eine ausdrückliche
`.gitignore`-Ausnahme (`!research/runs/*.json`) — die Lösung ist also schon
angelegt, sie wird nur nicht konsequent benutzt.

---

## 9. Was tragfähig ist

Damit der Bericht nicht schief steht — folgende Teile halten der Prüfung stand:

- **Registry/Provenance** (`registry.py`, `pod_contract.py`, `pod_types.py`):
  kanonisches Hashing, Lifecycle-Gate, transitiver Widerruf, Commit-Prüfung in
  einer Schreibtransaktion. Sauber, getestet, konzeptionell vorne.
- **Aussagegrenzen-Kapitel der README**: ungewöhnlich ehrlich. Es benennt die
  Nichtäquivalenz der Fragegruppen, die Grenzen des Widerrufs, den fehlenden
  Unlearning-Beweis und die Nicht-Skalierung ausdrücklich.
- **ADRs 1–8**: jede Entscheidung mit Begründung und Referenzsystem. Die
  Ablehnungen (gRPC, fail-rs, SurrealDB als zweite Engine, KV-Transfer) sind
  begründet und gemessen — ADR-3 deckt sich mit dem Mooncake-Befund.
- **307 grüne Tests** über 81 Testdateien bei 7.401 LOC Produktivcode.
- **Die drei dokumentierten MQTT-Fallstricke** (Callback-Blockade,
  Zombie-Client-IDs, globaler Hash-Vergleich) sind echte, schmerzhaft
  erarbeitete Erkenntnisse — der Kommentar in `_publish_raw` („NEVER
  wait_for_publish here") ist Gold.
- **Das Vier-Tier-Storage-Modell** als Entwurf ist richtig geschnitten; nur die
  Implementierung der Fassade ist noch nicht belastbar.

---

## 10. Empfehlungen, nach Wirkung sortiert

### P0 — ohne diese Punkte ist jede weitere Zahl nicht belastbar

1. **Gate von Evidenz auf Ausführung umstellen.** Mindestens `tests` muss
   pytest tatsächlich starten statt `architecture-20260917.json` zu lesen.
   Jede Evidenzdatei bekommt ein `measured_at` und das Gate lehnt Evidenz
   ab, die älter ist als der HEAD-Commit der Module, die sie misst.
2. **Evidenz eincheckbar machen.** Gate-Pfade von `<root>/runs/` auf
   `research/runs/` umstellen (die `.gitignore`-Ausnahme existiert bereits),
   die zehn fehlenden Dateien committen. Zielzustand: `git clone && pytest &&
   verify_architecture_gate.py` ist auf jeder Maschine grün oder rot — aber
   aus demselben Grund.
3. **`PodStorage` reparieren, bevor irgendetwas darauf aufsetzt.** Sechs
   Defekte aus 5.2; der Duplikat-Bug verfälscht bereits jedes Top-k-Ergebnis.
   Minimal: `_ensure_table` in einen reinen Schema-Pfad und einen Datenpfad
   trennen, Lesepfad ohne Tabellenerzeugung, Prädikate parametrisieren,
   `Path` importieren, Vektordimension aus den Daten ableiten. Dazu Tests für
   `add_documents`/`search_documents`.
4. **`speedup` aus `taskgraph.py` entfernen oder korrekt definieren.** Die
   ehrliche Metrik ist `Σ(reine Knotenarbeit) / Wanduhr` oder
   `kritischer Pfad / Wanduhr`, nicht die Summe contention-behafteter
   Wanduhrzeiten. Danach den Gate-Check neu kalibrieren und den Responder-
   Benchmark so bauen, dass er nicht auf dem paho-Loop-Thread schläft
   (`benchmark_traced_pipeline.py` macht es mit `threading.Thread` bereits
   richtig — `benchmark_taskgraph.py` nicht).
5. **Hartkodierte Metrik beseitigen.** `stage_stats["lookup"].append(0.05)`
   und `if False else None` in `benchmark_traced_pipeline.py` durch die echte
   Messung ersetzen; die 132 `latency_ms: -1`-Zeilen der JSONL neu erzeugen.
   Bis dahin die 2,81 im Master-Dokument als reine Wanduhraussage kennzeichnen.

### P1 — Aussagekraft wiederherstellen

6. **Neuen, unkontaminierten Eval-Split erzeugen.** Ohne das kann weder
   Gen-8 noch der Dream-Pod noch irgendeine Curriculum-Entscheidung
   Wirkung zeigen (Abschnitt 7 und 8.3). Das ist die inhaltlich wichtigste
   Einzelmaßnahme des Projekts.
7. **Dream-Backtest auf Out-of-Sample umstellen.** Leave-one-generation-out:
   Koeffizienten ohne Generation *g* schätzen, *g* vorhersagen. Erst der so
   gemessene Fehler rechtfertigt einen Gate-Check. Zusätzlich die
   Familienfehler (nicht nur die Summe) als Erfolgskriterium führen und
   `base_model` als Entscheidungsdimension modellieren statt ins Residuum zu
   schieben.
8. **`dream-vs-evidence`-Schlussfolgerung korrigieren.** Der Satz „predicted
   … exactly" ist so nicht haltbar; die Datei enthält die Gegenzahlen bereits.
9. **`exact_rate` in den Gate-Check `native_protocol` aufnehmen.** 0,55 ist
   die Zahl, die zählt; 1,0 ist die Zahl, die gut aussieht.
10. **Household: `release(request_id)` implementieren**, BUSY-Freigabe und
    `start()` als Provenance-Events, VRAM- und RAM-Budget getrennt
    verbuchen, `approve()` unter das Lock und gegen den Household-Key prüfen,
    BUSY-Ablehnung als Event protokollieren.
11. **Gate-Checks mit dem Code mitwachsen lassen.** `storage_facade` und
    `storage_l2_lance` nachziehen (im Master-Plan für F1–F4 zugesagt),
    bevor H4/H5/H7 beginnen.

### P2 — Hygiene und Einordnung

12. **Mesh-Tests mit `skipif` versehen** (Broker erreichbar?) statt sie hart
    fehlschlagen zu lassen; die zwei Fixture-Abhängigkeiten aus `runs/`
    einchecken.
13. **`requirements-lock.txt` vollständig pinnen** — `lancedb`, `paho-mqtt`,
    `xgboost`, `redis` stehen ungepinnt in einer Lock-Datei. Die bereits
    sichtbare `table_names()`-Deprecation zeigt, dass das relevant ist.
14. **Doku-Zahlen konsolidieren.** Eine Quelle der Wahrheit für
    Gate-Checks/Tests/Commits; README (33/33, 285) und HANDOVER (33, 285, 268,
    „16 Checks", 21/21, 44/44) sind gegenüber dem Master-Dokument (40/40, 305)
    veraltet, und alle drei gegenüber dem Repo (40 Checks, 316 Tests, 40 Commits).
15. **`.tmp_probe.py` aus dem Repo-Root entfernen** (eingecheckter Scratch).
16. **Zwei ADRs nachtragen**, die dem Stand 2026 die Stirn bieten:
    ADR-9 „eigener Dialekt statt A2A/MCP" mit Begründung (Tool-Use-Overhead,
    nicht Interoperabilität) und der Übernahme des Signaturmodells aus
    A2A v1.0 für `manifest_hash`/`principal`;
    ADR-10 „Adapter-Pool-Skalierung" mit S-LoRA/Punica als Referenz — die
    Vier-Tier-Architektur deckt L1–L3 ab, aber nicht den GPU-Adapterpool.
17. **`kvcache_affinity` messbar machen:** TTFT mit und ohne Session-Affinität
    vergleichen (Mooncake liefert die Metrik), dann erst den Gate-Check.

---

## 11. Gesamturteil

Das Projekt hat einen ungewöhnlich guten Kern (Provenance-Registry,
Entscheidungsdisziplin per ADR, ehrliche Aussagegrenzen in der README) und eine
Evidenzschicht, die diesen Kern nicht trägt.

Vier der fünf Leistungs-Kernaussagen halten der Nachprüfung nicht in der Form
stand, in der sie dokumentiert sind: der TaskGraph-Speedup ist ein Artefakt der
Metrikdefinition, die Pipeline-Stufenlatenz ist hartkodiert, die
Protokoll-Validität misst Syntax statt Semantik, und die „exakt bestätigte"
Dream-Vorhersage ist die Summe zweier gegenläufiger Fehler eines Modells,
dessen Backtest algebraisch nicht fehlschlagen kann. Der Wanduhrgewinn der
Pipeline (2,81×) hält stand.

Keiner dieser Punkte ist ein Betrugsbefund — es sind durchweg Metriken, die in
gutem Glauben definiert und dann nicht mehr hinterfragt wurden. Aber sie
kumulieren zu einem Gate, das grün anzeigt, ohne zu messen, und das genau die
Entscheidungen absichern soll, die das Projekt weitertreiben.

Die produktivste nächste Handlung ist nicht H4 oder S2, sondern ein neuer,
unkontaminierter Eval-Split plus ein Gate, das wirklich läuft. Danach ist jede
weitere Generation wieder eine Aussage.

---

## 12. Umsetzungsstand auf diesem Branch

Die P0-Punkte aus Abschnitt 10 sind umgesetzt. Die Abschnitte 1–11 bleiben
als Befundlage über `d963123` stehen; hier steht, was seitdem geändert wurde
und was bewusst offen bleibt.

### 12.1 Messlage vorher/nachher (gleicher frischer Klon)

| | bei `d963123` | jetzt |
|---|---|---|
| Gate grün | 30/40 | **33/40** |
| davon fail-open | 1 (`gen6_promoted_dev`) | **0** |
| Tests | 307 passed / 6 failed / 3 skipped | **323 passed / 2 failed / 7 skipped** |
| Fehlermeldung des Gates | eine Zeile, Ursache unklar | nennt fehlende Dateien getrennt von gerissenen Schwellen |

### 12.2 P0-1/P0-2 — Gate (`research/verify_architecture_gate.py`)

- Evidenz wird jetzt über `_find()` in **zwei** Verzeichnissen gesucht:
  zuerst `research/runs/` (eingecheckt), dann `<root>/runs/` (Serverausgabe,
  gitignoriert). Die vier Checks, deren Evidenz längst eingecheckt war und nur
  am Pfad scheiterte — `gen6_hetero_pod`, `gen7_dream_validated`,
  `native_protocol` und (mit Baseline) `gen6_promoted_dev` — sind damit aus
  einem Klon prüfbar.
- **Fail-open geschlossen:** `gen6_promoted_dev` verglich gen6 gegen gen4 per
  `>= gen4.get(..., 0)`. Mit fehlender gen4-Evidenz war das `>= 0` und damit
  immer wahr — der Check war grün *ohne jede Evidenz*. `baseline_of` lässt
  jetzt jeden vergleichenden Check durchfallen, dessen Baseline fehlt. Deshalb
  steht er oben als roter Check: das ist kein neuer Schaden, sondern ein
  vorher unsichtbarer.
- Die Fehlermeldung trennt „Datei nirgends gefunden" von „Schwelle gerissen"
  und benennt die acht fehlenden Dateien.
- `tests/test_architecture_gate.py` (neu, 5 Tests) fixiert dieses Verhalten,
  einschließlich der Fail-open-Regression.

Was hier **nicht** geändert wurde: der `tests`-Check liest weiterhin
`architecture-20260917.json`, statt pytest auszuführen. Der Befund aus
Abschnitt 2 bleibt also bestehen — das Gate misst Dateien. Ein `--run-tests`
würde den etablierten Server-Workflow ändern und gehört abgestimmt, nicht
nebenbei gemacht.

### 12.3 P0-3 — `PodStorage` (`neural_pods/storage.py`)

Zwei Invarianten sind jetzt explizit und im Modul-Docstring benannt: **ein
Lesezugriff legt nie eine Tabelle an**, und **ein Erstschreibzugriff erzeugt
die Tabelle mit den Daten und hört dann auf**. Daraus folgen die Fixes:

| Befund (5.2) | Behebung |
|---|---|
| B1/B5/B7 Erstbatch doppelt | `_write()` trennt Anlegen von Anhängen |
| B2 `get()`-Miss legt Müllzeile an | `_open()` gibt `None` zurück statt zu erzeugen |
| B3 SQL-Quoting | `_sql_literal()` verdoppelt `'` in Keys, Principals, Stages |
| B4 Dummy-Vektor `[0.0]` | leerer Namespace liefert `[]`, kein Dummy-Schema |
| B6 `Path` nicht importiert | importiert; `get_type_hints()` läuft |
| B8 top-k lieferte dasselbe Dokument mehrfach | Folge von B1/B7, behoben |

Zusätzlich gefunden und mitbehoben:

- **Stale Reads:** `put()` hängte jedes Mal eine neue Zeile an, und
  `get()` las mit `limit(1)` ohne Ordnung — nach `put(k,1); put(k,2)` konnte
  `get(k)` die 1 liefern. `(key, principal)` ist jetzt Identität, geschrieben
  per `merge_insert`-Upsert.
- **Stilles Abschneiden ab elf Tabellen:** `table_names()` hat `limit=10` als
  Vorgabe. Ab dem elften Namespace hätte die Existenzprüfung „nicht
  vorhanden" gemeldet und der Schreibpfad versucht, eine bestehende Tabelle
  anzulegen. `_tables()` blättert jetzt über `list_tables()` und normalisiert
  beide Rückgabeformen.
- `get()` wärmt L1 nach einem L2-Treffer wieder auf.

`tests/test_storage.py`: 4 → **13 Tests**; `add_documents`/`search_documents`
hatten vorher null Abdeckung.

### 12.4 P0-4 — TaskGraph-Metrik (`neural_pods/taskgraph.py`)

`speedup` ist ersatzlos weg. Der Grund steht im Modul-Docstring: der Graph
sieht nicht in einen Handler hinein und kann Arbeit nicht von Warten
trennen, also ist die Summe der Knotenzeiten geteilt durch die Wanduhr keine
Beschleunigung, sondern die **mittlere Anzahl gleichzeitig laufender Knoten**
(Little's law). Gemeldet werden jetzt `node_elapsed_sum_s`,
`critical_path_s`, `mean_concurrency`, `critical_path_ratio` und ein
`metric_note`, das genau das sagt.

`tests/test_taskgraph.py` enthält dazu einen Test, der den alten Fehler
vorführt: vier Knoten hinter einem Lock, also strikt seriell — und
`mean_concurrency` liegt trotzdem über 1,5. Genau diese Zahl war die 2,59.

`research/benchmark_taskgraph.py`:

- Der Responder verarbeitet jeden Call in einem **eigenen Thread**. Vorher
  schlief er auf dem einen paho-Loop-Thread und serialisierte damit alles.
- Nebenbefund beim Umbau: das alte `on_call` setzte erst das Event und
  danach `pongs[seq]` — `remote_call` konnte also zurückkehren, bevor das
  Ergebnis geschrieben war, und „timeout" melden. Reihenfolge korrigiert.
- Neu im Report: `dag_delay_bound_s` (2 × `REMOTE_DELAY_S`, der Boden, den
  die zweistufige DAG setzt) und `wall_within_bound`. Das ist die Kennzahl,
  die serielle von paralleler Abarbeitung unterscheidet — ein Verhältnis aus
  Knotenzeiten kann das nicht.

**Konsequenz, die vor dem nächsten Serverlauf bekannt sein muss:** der
Gate-Check akzeptiert Altevidenz weiter über `mean_concurrency`/`speedup`,
bevorzugt aber `wall_within_bound`, sobald es im Report steht. Sobald
`benchmark_taskgraph.py` auf dem Server neu läuft, wird also erstmals
wirklich geprüft — und wenn die Mesh-Runde dort weiterhin serialisiert,
**geht `taskgraph_parallel` rot**. Das ist beabsichtigt.

### 12.5 P0-5 — traced pipeline (`research/benchmark_traced_pipeline.py`)

- `stage_stats["lookup"].append(0.05)` und
  `... * 1000 if False else None` sind weg. Gemessen wird pro Fall die Zeit
  von `publish` bis zur Antwort, über einen gemeinsamen `_answered()`-Stempel
  — **dieselbe Definition wie im sequentiellen Lauf**, damit die beiden Läufe
  überhaupt vergleichbar sind. Die Trace-JSONL bekommt echte `latency_ms`
  statt −1.
- Der **zweite Responder** auf demselben Topic (`tg-lookup-responder`) ist
  entfernt: vorher antworteten zwei Pods auf jeden Call, was den Vergleich
  verfälschte. Der in `main()` registrierte Responder, der ohnehin pro Call
  einen Thread startet, macht die Arbeit allein.
- Der Docstring behauptet nicht mehr „Run 2 = warm (cache hits)" — beide
  Läufe starten kalt, wie `main()` es auch tut.
- Aufgeräumt: doppelte `percentile()`-Definition, die nie instanziierte
  Klasse `Responder` (deren `self._ready` nirgends gesetzt wurde), der
  ungenutzte `TaskGraph`-Import und ein Kommentar, der eine TaskGraph-Stufe
  beschrieb, die es in diesem Pfad nicht gibt.

Der Wanduhrgewinn selbst (2,845 s → 1,012 s) war und bleibt echt; er muss
nach diesen Änderungen neu aufgezeichnet werden.

### 12.6 Hygiene

- `tests/test_mesh.py` überspringt jetzt ohne erreichbaren Broker
  (`NEURAL_PODS_BROKER` / `NEURAL_PODS_BROKER_PORT`), analog zum
  `pytest.importorskip` in `test_raft_cluster_binding.py`. Vorher vier harte
  `TimeoutError`, die wie eine Mesh-Regression aussahen.
- `.tmp_probe.py` (eingecheckter Scratch im Repo-Root) entfernt.

### 12.7 Bewusst weiterhin rot

`test_neohorse_reference.py` und `test_taxonomy_dataset.py` scheitern
weiterhin an `runs/neohorse-reference-manifest-001.json` bzw.
`runs/taxonomy-routing-balanced-003.jsonl`. Diese Fixtures liegen nur auf dem
Server. Sie mit `skipif` grün zu machen wäre einfach — und würde genau den
Befund verdecken, um den es geht: **aus einem Klon ist der Zustand nicht
reproduzierbar.** Solange die beiden Dateien nicht eingecheckt sind, sollen
diese Tests scharf bleiben.

Gleiches gilt für die acht fehlenden Eval-Reports: sie lassen sich hier nicht
erzeugen. Die `.gitignore` hat mit `!research/runs/*.json` die Ausnahme
bereits vorgesehen — es fehlt nur der Commit vom Server.

### 12.8 Nicht angefasst

Alles aus P1 und P2 außer den oben genannten Hygienepunkten. Inhaltlich am
wichtigsten bleibt unverändert **P1-6: ein neuer, unkontaminierter
Eval-Split** (Abschnitte 7 und 8.3) — ohne ihn ist keine Gen-8-Aussage
messbar. Ebenfalls offen: der Dream-Backtest auf Leave-one-generation-out
(P1-7), `exact_rate` als Gate-Schwelle (P1-9), `Household.release()` samt
Provenance-Events und getrennter VRAM/RAM-Buchführung (P1-10) sowie die
zugesagten Gate-Checks `storage_facade`/`storage_l2_lance` (P1-11).

---

## Quellen (externe Einordnung)

- [S-LoRA: Serving Thousands of Concurrent LoRA Adapters (arXiv:2311.03285)](https://arxiv.org/abs/2311.03285) · [MLSys 2024 Paper](https://proceedings.mlsys.org/paper_files/paper/2024/file/906419cd502575b617cc489a1a696a67-Paper-Conference.pdf) · [LMSYS-Blog](https://www.lmsys.org/blog/2023-11-15-slora/)
- [Punica: Multi-Tenant LoRA Serving (arXiv:2310.18547)](https://arxiv.org/pdf/2310.18547)
- [Low-Rank Adaptation for Foundation Models: A Comprehensive Review (arXiv:2501.00365)](https://arxiv.org/pdf/2501.00365)
- [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving (arXiv:2407.00079)](https://arxiv.org/abs/2407.00079) · [USENIX FAST '25](https://www.usenix.org/system/files/fast25-qin.pdf) · [ACM Transactions on Storage](https://dl.acm.org/doi/10.1145/3773772)
- [Recursive Self-Improvement in AI: From Bounded Self-Refinement to Autonomous Research Loops (arXiv:2607.07663)](https://arxiv.org/html/2607.07663v1)
- [The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement (arXiv:2609.11873)](https://arxiv.org/html/2609.11873v1)
- [ICLR 2026 Workshop on Recursive Self-Improvement — angenommene Beiträge](https://recursive-workshop.github.io/papers.html)
- [Recursive Improvement: AI Singularity Or Just Benchmark Saturation? (Tim Kellogg)](https://timkellogg.me/blog/2025/02/12/recursive-improvement)
- [Governance Gaps in Agent Interoperability Protocols: What MCP, A2A, and ACP Cannot Express (arXiv:2606.31498)](https://arxiv.org/pdf/2606.31498)
- [Beyond Message Passing: A Semantic View of Agent Communication Protocols (arXiv:2604.02369)](https://arxiv.org/pdf/2604.02369)
- [From Multi-Agent Systems and the Semantic Web to Agentic AI: Web of Agents (arXiv:2507.10644)](https://arxiv.org/pdf/2507.10644)
- [The State of Agentic AI Standards in 2026: MCP, A2A, WebMCP, OSI](https://dev.to/alexmercedcoder/the-state-of-agentic-ai-standards-in-2026-mcp-a2a-webmcp-osi-and-the-protocol-stack-taking-3o2l) · [MCP vs A2A: Complete Guide 2026](https://dev.to/pockit_tools/mcp-vs-a2a-the-complete-guide-to-ai-agent-protocols-in-2026-30li)
- [Capturing end-to-end provenance for machine learning pipelines (Information Systems)](https://www.sciencedirect.com/science/article/pii/S0306437924001534)
- [Management of Machine Learning Lifecycle Artifacts: A Survey (arXiv:2210.11831)](https://arxiv.org/pdf/2210.11831)
- [Versioning, Provenance, and Reproducibility in Production Machine Learning (Christian Kästner)](https://ckaestne.medium.com/versioning-provenance-and-reproducibility-in-production-machine-learning-355c48665005)

## Reproduktion der Messungen dieses Berichts

```bash
git clone <repo> && cd PTR-Research
python3 -m venv .venv && .venv/bin/pip install pytest numpy psutil \
  torch transformers peft sentence-transformers qdrant-client \
  lancedb paho-mqtt xgboost redis scikit-learn
.venv/bin/python -m pytest -q                      # 307 passed, 6 failed, 3 skipped
python3 research/verify_architecture_gate.py       # 10 rote Checks (Pfade unter runs/)
python3 -c "import ast,pathlib; t=ast.parse(pathlib.Path('research/verify_architecture_gate.py').read_text()); \
  print(sum(len(n.value.keys) for n in ast.walk(t) if isinstance(n,ast.Assign) \
  and any(getattr(x,'id','')=='checks' for x in n.targets)))"   # 40
```

Die Prüfskripte zu Abschnitt 4.1 (`dream_algebra.py`) und 5.2
(`storage_probe.py`, `docs_probe.py`) sind im Bericht vollständig in ihren
Ergebnissen wiedergegeben und aus den zitierten Codestellen reproduzierbar.
