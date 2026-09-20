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

Die Punkte aus Abschnitt 10 sind umgesetzt — erst P0, dann P1 und P2. Die
Abschnitte 1–11 bleiben als Befundlage über `d963123` stehen; hier steht, was
seitdem geändert wurde und was offen bleibt.

### 12.1 Messlage vorher/nachher (gleicher frischer Klon)

| | bei `d963123` | nach P0 | jetzt |
|---|---|---|---|
| Gate-Checks definiert | 40 | 40 | **43** |
| Gate-Checks grün | 30 | 33 | **36** |
| davon fail-open | 1 | 0 | **0** |
| `tests`-Check | liest eine Datei vom 2026-09-17 | unverändert | **führt aus oder prüft gegen einen Quellcode-Hash** |
| Tests | 307 / 6 failed / 3 skipped | 323 / 2 failed / 7 skipped | **356 passed / 0 failed / 8 skipped** |

Die sieben roten Checks sind dieselben sieben: sie brauchen Eval-Reports, die
nur auf dem Server liegen. Keiner davon ist eine Regression, und das Gate sagt
das jetzt auch.

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

### 12.7 P0-1 (Rest) — das Gate führt die Tests jetzt wirklich aus

Der offen gebliebene Kern von Abschnitt 2 ist geschlossen. `research/record_test_run.py`
startet pytest und schreibt `research/runs/tests-20260920.json` mit
`passed`, `failed`, `skipped` **und `sources_sha256`** — einem Digest über
jede `.py`-Datei unter `neural_pods/`, `research/` und `tests/`. Der
`tests`-Check rechnet den Digest neu aus und lässt Evidenz durchfallen, die
von anderem Code stammt. Nachgewiesen: eine einzige zusätzliche Zeile in
`neural_pods/storage.py` reicht, und der Check springt auf rot.

`python research/verify_architecture_gate.py --run-tests` überspringt die
Aufzeichnung und führt die Suite im Gate selbst aus.

Was weiterhin gilt: die übrigen 42 Checks lesen aufgezeichnete Messdateien.
Das Gate ist dort ein Regressions-Journal. Der Unterschied ist, dass es das
jetzt selbst sagt — `verify()` gibt `tests_evidence` zurück, und ohne
Testaufzeichnung steht dort ausdrücklich „stale: no digest, predates the code
it covers".

### 12.8 P1-6 — ein frischer, unkontaminierter Eval-Split

Die inhaltlich wichtigste Maßnahme, und beim Umsetzen kam ein Befund dazu,
der in den Abschnitten 1–11 noch fehlt:

> **Die frozen Splits sind aus dem Repository nicht regenerierbar.**
> `research/reader_training_data.build_data()` erzeugt heute **96** Test- und
> 96 Dev-Fälle. Jeder Eval-Report nennt **132**. Kein `prepare_*`-Skript
> erweitert dev/test — `prepare_generation7.py` bricht sogar ab, wenn sich
> dort ein Byte ändert. Die 36 zusätzlichen Fälle (u. a. die `test:ngu:*`-IDs
> aus den Reports) stammen aus Code, der nicht im Repo liegt. Die
> Reproduzierbarkeitslücke aus Abschnitt 1.1 reicht also bis in die
> Evaluationsdaten selbst.

Neu, und bewusst ohne die frozen Splits anzufassen (Sicherheitsregel 5):

- `research/reader_holdout_data.py` baut einen `holdout`-Split mit **denselben
  13 Fallfamilien** wie der frozen Test-Split (damit die Zahl vergleichbar
  bleibt), aber mit Lieferanten, Komponentencodes, Laufzeiten und
  Formulierungen, die in **keinem** frozen Split vorkommen. 96 Zeilen,
  48 en / 48 de.
- `disjointness_report()` vergleicht IDs, vollständige Fragetexte,
  Lieferanten, Komponenten und Antwortwerte gegen alle frozen Splits.
  Überschneidung: leer, in jeder Facette.
- `research/generate_holdout_split.py` schreibt Split und Manifest nach
  `research/runs/` — **eingecheckt**. Das Manifest führt `cases_sha256` **und
  `generator_sha256`**, damit genau der Drift, der die frozen Splits
  unregenerierbar gemacht hat, hier auffällt. `--check` vergleicht Artefakt
  gegen Generator.
- Gate-Check `holdout_split_disjoint`; `tests/test_reader_holdout_data.py`
  (6 Tests) fixiert Disjunktheit, Familienparität und Artefakt-Treue.

Was hier **nicht** geht: den Split zu bewerten. Dafür braucht es den
Reader-Checkpoint auf dem Server. Das Manifest sagt das in seinem
`scope`-Feld.

### 12.9 P1-7 — Dream-Backtest: leave-one-generation-out

`neural_pods/dream.py` misst jetzt, was es vorher behauptet hat.

- **Der In-Sample-Wert ist als solcher markiert.** `backtest()` liefert
  `in_sample.degenerate`, wahr genau dann, wenn das Modell mindestens so
  viele freie Parameter hat wie die Historie Generationen — der Fall, in dem
  die Residuen arithmetisch null sind. Ein Test führt das vor.
- **`out_of_sample`** hält jede Generation gegen die übrigen. Fehlt in den
  restlichen Übergängen die Identifikation, wird die Generation **mit Grund**
  übersprungen statt stillschweigend mitgezählt.
- Auf der echten Entscheidungsstruktur gen3–gen6 lautet das Ergebnis:
  **0 von 4 Generationen out-of-sample vorhersagbar**, je mit benannter
  Ursache (gen3: `intercept`, gen4: `concept_oversample`, gen5:
  `lookup_anchor`, gen6: `base_model=neohorse`). Das ist die ehrliche
  Antwort auf „Backtest-Fehler 0,01".
- **`base_model` ist eine modellierte Dimension.** Nichtnumerische
  Entscheidungswerte werden zu 0/1-Indikatoren; der Gen-5→Gen-6-Wechsel fiel
  vorher komplett ins Residuum.
- **Mehrdeutige Übergänge** (zwei Knöpfe gleichzeitig) landen in `ambiguous`
  statt dem zuletzt iterierten Knopf angelastet zu werden. Koeffizienten
  werden über **alle** sauberen Übergänge gemittelt, nicht überschrieben.
- **Extrapolation wird per Vorgabe abgelehnt.** Die Limits kommen aus
  `observed_limits()` statt aus einer Tabelle, die `lookup_anchor=2` erlaubte,
  obwohl nur 0 und 1 je vorkamen. `allow_extrapolation=True` ist weiterhin
  möglich, markiert dann aber jeden Eintrag mit `extrapolates`.
- `simulate()` klemmt nach unten wie nach oben; eine nicht identifizierte
  Vorhersage liefert `estimable: False` statt einer Zahl, die die unbekannte
  Entscheidung stillschweigend als wirkungslos behandelt.
- `HistoryPool.from_project()` sucht in `research/runs/` **und** `runs/`, und
  die Fehlermeldung nennt die fehlenden Generationen.

Der Gate-Check akzeptiert beide Report-Formen; die alte Evidenz bleibt gültig,
neue Läufe werden auf dem Out-of-Sample-Teil geprüft.

### 12.10 P1-8/P1-9 — Deutung und Schwelle

`research/runs/dream-vs-evidence-20260920.json`: **kein Messwert verändert**
(im Skript per Assertion abgesichert). Berichtigt wurde die `conclusion`, und
ergänzt wurden die Zahlen, die ihr widersprachen —
`per_family_counts` (typed 82 vorhergesagt / 81 real, concept 43 / 44),
`predicted_families_correct: 0`, die Sättigungslage und die Notiz, dass die
Politik extrapoliert war. Ein `correction`-Block hält Datum, geänderte Felder
und Grund fest.

`native_protocol` prüft jetzt zusätzlich `exact_rate >= 0.50`. Die Schwelle
liegt bewusst knapp **unter** dem gemessenen 0,55: sie sichert die Kennzahl
gegen Rückschritt, ohne dass ich den heutigen Stand einseitig zum Fehlschlag
erkläre. Sie gehört angehoben, sobald der Dialekt besser wird.

### 12.11 P1-10 — Household

| Befund (5.3) | Behebung |
|---|---|
| H1 BUSY wird nie freigegeben | `release(request_id, key)` gibt Donoren frei; ein Donor kann wieder mehr als eine Allokation bedienen |
| H2 `start()` ohne Event | `allocation_started` mit vollem Plan; dazu `allocation_released` und `allocation_refused` |
| H3 „reconstructible from the registry events" war unbelegt | `restore_from_events()` spielt den Log ab und stellt Reservierungen und BUSY-Flags wieder her — Test inklusive Neustart-Szenario |
| H4 VRAM und RAM teilten einen Zähler | getrennte Budgets je Tier |
| H5 `approve()` ohne Lock und ohne Key | Household-Key erforderlich, Lauf unter `self._lock` |
| H6 BUSY-Ablehnung ohne Event | `allocation_refused` |

Zusätzlich gefunden: die Tier-Präferenz wurde **nicht** eingehalten — ein
Segment mit `tier_preference=("vram",)` fiel trotzdem in RAM durch. Jetzt
strikt; passt kein Tier, schlägt `start()` fehl. `tests/test_household.py`:
6 → 13 Tests.

### 12.12 P1-11 — Gate-Checks für F3/F4

`research/benchmark_storage_facade.py` läuft **ohne Server** (LanceDB im
Temp-Verzeichnis, L1 gegen einen Stub, wenn kein Redis erreichbar ist —
im Report als `l1_backend` vermerkt). Er misst genau die sechs Defekte aus
5.2 als Eigenschaften: Duplikate, vom Lesepfad erzeugte Tabellen, Stale
Reads, L1-Rückbefüllung, Quoting, verschiedene top-k, Tabellenlisting über
zehn hinaus. Die Evidenz liegt eingecheckt in `research/runs/`.

Gemessen nebenbei: L1 p50 **0,057 ms** gegen L2 p50 **23,4 ms** — die
Tier-Reihenfolge ist kein Detail.

Zwei neue Gate-Checks: `storage_facade`, `storage_l2_lance`.

### 12.13 Sicherheit — signierte Envelopes

Abschnitt 6 beschrieb zwei Regeln, die weniger leisten als ihr Wortlaut.
Beide sind jetzt entweder belegt oder ehrlich beschriftet:

- **`mesh.py` signiert Envelopes**, wenn ein `secret` gesetzt ist: HMAC-SHA256
  über die kanonische Form ohne das Signaturfeld. Ein Endpunkt mit Schlüssel
  weist unsignierte und falsch signierte Envelopes ab und zählt sie in
  `unsigned_rejected` — kein stilles Zurückfallen auf ungesichert, wenn
  signierte und unsignierte Pods auf demselben Topic sprechen. Das ist das
  Signaturmodell aus A2A v1.0, ohne das Protokoll zu übernehmen (ADR-9).
- Auch ohne Schlüssel müssen `manifest_hash` und `principal` **vorhanden**
  sein statt nur mitzureisen.
- `stats()` meldet `signed`, `unsigned_rejected` und
  `topic_acl_enforced_by: "client"`. Der Docstring von `publish_raw()` sagt
  jetzt ausdrücklich, dass er die ACL umgeht, und der Modul-Docstring sagt,
  dass die ACL eine Leitplanke für kooperierende Pods ist, keine Grenze.
- `tests/test_mesh_envelope.py` (7 Tests, ohne Broker) prüft Round-Trip,
  fehlende Signatur, manipulierten `principal`, manipulierten Body und
  fremden Schlüssel.

`secret` ist optional, damit der laufende Betrieb nicht bricht — die
Entscheidung, ihn zu setzen, gehört auf den Server.

### 12.14 P2 — Fixtures, Lock, Doku, ADRs

- **`taxonomy-routing-balanced-003.jsonl` ist jetzt eingecheckt.** Der
  Datensatz ist reines, deterministisches Code-Ergebnis;
  `generate_taxonomy_dataset.py` bekam `render()` und ein `--output`, das
  standardmäßig nach `research/runs/` schreibt. Der Test prüft zusätzlich,
  dass das Artefakt zum Generator passt und dass Held-out-Formulierungen nie
  im Trainingsanteil auftauchen. Aus Rot wurde **Grün**, nicht „Skip".
- **`test_neohorse_reference`** überspringt mit benanntem Grund: Manifest und
  GPU-Probe entstehen auf dem Server und lassen sich hier nicht erzeugen. Der
  Test sucht in `research/runs/` und `runs/`; ein Commit vom Server genügt.
- **`requirements-lock.txt`** pinnt `lancedb==0.39.0`, `paho-mqtt==2.1.0`,
  `xgboost==3.2.0`, `redis==8.1.0` — mit Kommentar, gegen welche Umgebung
  verifiziert wurde und wie man mit dem Server abgleicht. Gerade lancedb hat
  `table_names()` bereits durch `list_tables()` ersetzt und den Rückgabetyp
  geändert; ungepinnt ist hier keine neutrale Wahl.
- **Doku-Zahlen:** `ARCHITECTURE-MASTER-20260920.md` ist die einzige Quelle
  der Wahrheit und trägt eine nachgemessene Standtabelle. README und HANDOVER
  führen keine eigenen Zählstände mehr und verweisen dorthin. Die
  Messwertspalten im Master-Dokument sind korrigiert (Frames 1.0 **und**
  exact 0.55; `mean_concurrency` statt „Speedup"; RTT auf **einem** Host;
  Dream out-of-sample 0). Der HANDOVER-Absatz zum Dream-Pod trägt einen
  Überholt-Hinweis.
- **ADR-9** (eigener Dialekt statt A2A/MCP, Signaturmodell übernommen) und
  **ADR-10** (Adapter-Pool-Skalierung, S-LoRA/Punica als Referenz) sind
  nachgetragen. Die Sicherheitsregeln 1, 6 und 7 haben jetzt einen Absatz, der
  sagt, was sie tatsächlich leisten.
- **P2-17:** `research/benchmark_kvcache_affinity.py` misst TTFT mit und ohne
  Session-Affinität gegen zwei vLLM-Replicas — die Kennzahl, die Mooncake
  nahelegt und die ADR-3 bisher schuldig blieb. Routing und Statistik sind
  rein und getestet (`tests/test_kvcache_affinity.py`, 6 Tests); der
  HTTP-Durchlauf braucht die Replicas und ist hier **nicht** gelaufen. Deshalb
  auch kein Gate-Check `kvcache_affinity`: dafür fehlt die Evidenz.

### 12.15 Was offen bleibt

1. **Acht Eval-Reports vom Server committen.** Danach sind 43/43 aus einem
   Klon prüfbar. Die `.gitignore`-Ausnahme existiert; es fehlt der Commit.
2. **Den Holdout-Split bewerten.** Braucht den Reader-Checkpoint. Erst danach
   ist wieder messbar, ob eine Generation besser ist als die vorige.
3. **`benchmark_taskgraph.py` und `benchmark_traced_pipeline.py` neu
   aufzeichnen.** Beide sind geändert; die eingecheckte Evidenz stammt noch
   vom alten Code. Beim TaskGraph kann `wall_within_bound` dabei rot werden —
   das ist der Zweck der Kennzahl.
4. **`benchmark_kvcache_affinity.py` laufen lassen**, dann den Gate-Check
   ergänzen.
5. **`exact_rate`-Schwelle anheben**, sobald der Dialekt über 0,55 kommt.
6. **`secret=` im Mesh setzen** — die Signaturmechanik steht, die
   Schlüsselverteilung ist eine Betriebsentscheidung.
7. **`perception.py` und `mesh_cache.py`** haben weiterhin nur Benchmarks,
   keine Unit-Tests.
8. **Die frozen Splits** bleiben unregenerierbar (96 vs. 132). Entweder taucht
   der erzeugende Code wieder auf, oder der Holdout-Split löst sie als
   Messgrundlage ab.

---

## 13. Review vor dem Merge und Architekturarbeit

### 13.1 Messlage

| | `d963123` | nach P0 | nach P1/P2 | jetzt |
|---|---|---|---|---|
| Gate-Checks definiert | 40 | 40 | 43 | **44** |
| Gate-Checks grün | 30 | 33 | 36 | **37** |
| Tests | 307 / 6 failed | 323 / 2 failed | 356 / 0 failed | **381 / 0 failed / 8 skipped** |
| Schichtverstöße | nicht geprüft | nicht geprüft | nicht geprüft | **0, geprüft** |

### 13.2 Der Review fand acht Fehler — alle in Code aus diesem PR

Ein vollständiger Review des Diffs gegen `main` vor dem Merge. Jeder Befund
wurde nachgestellt, behoben und mit einem Test festgenagelt.

| # | Befund | Warum er zählt |
|---|---|---|
| 1 | `restore_from_events()` baute die Reservierungen wieder auf, **aber nicht `self.requests`** — `release()` warf `KeyError`, der Donor blieb für immer BUSY | Genau der Defekt, den `release()` beheben sollte, auf dem Wiederherstellungspfad wieder eingebaut |
| 2 | Der Round-Robin-Arm des KV-Affinitäts-Benchmarks **routete identisch zum Affinitätsarm** (globaler Zähler, Sessions in der inneren Schleife, gerade Sessionzahl über zwei Replicas) | Die Kontrollgruppe war keine; `affinity_helps` hätte Rauschen gemessen |
| 3 | Der dokumentierte `--redis HOST`-Pfad des Storage-Benchmarks erzeugte **garantiert gate-rote Evidenz**: L1 wurde nie geleert, und die Rückbefüllung wurde an einem Stub-Attribut gezählt | Ein Benchmark, der mit echtem Backend falsch misst, ist schlimmer als keiner |
| 4 | `_answered_at` wurde **zwischen beiden Läufen geteilt und nie geleert** — ein Timeout in Lauf 2 wurde gegen den Stempel aus Lauf 1 gemessen und schrieb **negative** Latenzen | Dieselbe Klasse wie die hartkodierte 0,05, nur subtiler |
| 5 | Die neue `wall_within_bound`-Kennzahl wird von der eingecheckten Evidenz **umgangen** | War in Prosa angekündigt, aber nicht im Gate-Output sichtbar |
| 6 | `start()` prüfte den Status **außerhalb** von `self._lock` | Zwei gleichzeitige Starts überschrieben die Reservierung; ein Donor blieb dauerhaft BUSY |
| 7 | Ein vollständig unidentifiziertes Ranking lieferte trotzdem einen **`winner` mit `score: None`** | Der Gate-Check prüfte nur, dass der Gewinner Entscheidungen trägt |
| 8 | Der neue L1-Rückbefüllungspfad in `get()` **hebelte `put(..., l1=False)` aus** | Der erste Lesezugriff legte den Wert in genau den Tier, den der Schreiber ausgeschlossen hatte |

Behebungen im Einzelnen:

- **1+6:** `allocation_started` trägt jetzt `model_ref` und `donors`, damit
  `restore_from_events()` den Request selbst rekonstruiert; Statusprüfung und
  Platzierung liegen unter einem Lock. Zwei Tests, einer davon nebenläufig.
- **2:** Der Kontrollarm rotiert **pro Session** statt global — eine Session
  landet in aufeinanderfolgenden Zügen auf verschiedenen Replicas, also genau
  das Gegenteil dessen, was Affinität bewahren soll. Ein Test vergleicht beide
  Arme über 8 Sessions × 4 Züge.
- **3:** L1 wird über die Schlüssel der Fassade selbst geleert und gezählt,
  mit `delete`/`get` — funktioniert gegen Stub und echten Redis gleich.
- **4:** Beide Bookkeeping-Dicts werden zu Beginn jedes Laufs geleert; ein
  unbeantworteter Fall wird gegen das Ende der Sammelschleife gemessen, nie
  gegen einen Stempel aus einem früheren Lauf.
- **5:** Das Gate meldet akzeptierte Altformen jetzt in `legacy_evidence` und
  in der Fehlermeldung, statt sie nur im Bericht zu erwähnen.
- **7:** `run_dream_cycle.py` bricht ab, wenn keine Kandidatenpolitik
  identifiziert ist; der Gate-Check verlangt `winner.estimable is not False`.
- **8:** `l1=False` ist eine Eigenschaft des **Werts**, nicht des Aufrufs, und
  wird als `l1_eligible` in der Zeile gespeichert. Die Rückbefüllung achtet
  darauf.

Nicht als Befund gewertet und geprüft: die HMAC-Envelope-Signatur (kanonische
Form schließt `sig` aus, beide Seiten sortieren Schlüssel), die Blätterlogik
von `_tables()` gegen lancedb 0.39, und `MAX_RENDERABLE_DAYS = 69` als exakte
Grenze des Wochen-Wortschatzes von `duration()`.

### 13.3 Ein Fehler, den erst der Review-Test sichtbar machte

Der nebenläufige Start-Test schlug zunächst nicht an der Sperre fehl, sondern
an der Datenbank:

```
sqlite3.ProgrammingError: SQLite objects created in a thread can only be
used in that same thread.
```

**Die Provenance-Registry — die Komponente, in die jede Schicht schreibt —
war nicht aus den Threads benutzbar, in denen das System läuft.**
`Household` hat ein Lock, `TaskGraph` einen Thread-Pool, `AdaptiveBatcher`
einen Hintergrund-Thread, die Mesh-Handler laufen auf dem paho-Loop — und
jeder Event-Schreibvorgang aus einem dieser Threads wäre abgestürzt. Das war
kein Befund des Reviews, sondern ein Nebenprodukt davon, dass der Review
überhaupt einen nebenläufigen Test verlangt hat.

Behoben: `check_same_thread=False` plus ein `RLock`, das die mehrstufige
Transaktion serialisiert (Pythons sqlite3 läuft im serialized mode, die
Verbindung selbst verträgt also nebenläufige Statements; was Serialisierung
braucht, ist `BEGIN`/`COMMIT`). Transaktionen sind jetzt wiedereintrittsfähig.
Drei Tests, darunter acht Threads, die gleichzeitig Events schreiben.

### 13.4 Architektur: die Schichtenregel ist jetzt eine Eigenschaft

Abschnitt 1 des Master-Dokuments zeichnet fünf Schichten und formuliert die
Regel in einem Satz. Der Satz war Prosa — also genau die Sorte Zusicherung,
die dieser Bericht sonst überall als ungeprüft ausweist.

`neural_pods/architecture.py` führt das Modell jetzt als Daten und prüft den
Baum statisch dagegen. Verstoß ist: ein Import nach oben, ein Backend-Zugriff
an der Fassade vorbei, ein Laufzeit-Zyklus, ein Modul ohne Schichtzuordnung,
ein Manifest-Eintrag ohne Modul. Stand: **56 Module, 28 Laufzeitkanten,
0 Verstöße.**

Zwei Details, die den Unterschied zwischen einem echten und einem
kosmetischen Check ausmachen:

- **`if TYPE_CHECKING:` zählt nicht.** `ranking` und `local_search` verweisen
  genau so aufeinander; ein naiver Parser meldet dort einen Zyklus, den es
  zur Laufzeit nicht gibt. Der Check meldet solche Kanten getrennt.
- **Der Prüfer wird gegen ein synthetisches Paket geprüft.** Eine Regel, die
  nie hat fehlschlagen sehen, ist keine Regel: `tests/test_architecture_layers.py`
  baut Pakete mit einem Import nach oben, einem Backend an der Fassade vorbei,
  einem echten Zyklus, einem nur annotierten Zyklus und einem gelöschten
  Manifest-Modul, und verlangt jeweils genau den passenden Verstoß.

Der Gate-Check `layering` führt das bei jeder Promotion aus. Er liest keine
Evidenzdatei — er analysiert den Baum, wie er gerade ist.

### 13.5 Architektur: das Gate prüft seine eigene Abhängigkeitstabelle

Die handgepflegte `baseline_of`-Tabelle war selbst wieder eine Zusicherung,
an die sich jemand erinnern muss — dieselbe Fehlerklasse wie der Fail-open,
den sie behebt. Statt die 44 Prädikate kurz vor dem Merge umzuschreiben
(Risiko ohne Gegenwert), prüft ein Test die Tabelle jetzt:

Er leitet per AST aus `verify()` ab, welche Evidenzdatei jedes Prädikat
liest, entfernt dann **jede** Datei einzeln und verlangt, dass genau die
Checks rot werden, die sie lesen. Ein Check, der ohne seine Evidenz grün
bleibt, fällt sofort auf — das ist die Prüfung, die `gen6_promoted_dev`
gefunden hätte, bevor jemand danach suchen musste.

Gegenprobe gelaufen: mit entferntem `baseline_of`-Eintrag schlägt sie fehl.

**Bewusst nicht gemacht:** der vollständige Umbau des Gates auf eine
deklarative Check-Registry. Er hätte alle 44 Prädikate angefasst, deren
Korrektheit gerade erst durch Tests festgestellt wurde, und der konkrete
Schmerz, den er lindern sollte, ist durch die obige Prüfung bereits weg.
Bleibt als Folgearbeit notiert, nicht als stiller Verzicht.

### 13.6 Architektur: öffentliche Event-API, Paketoberfläche, Testimporte

- **`registry.record_event()`** ist die öffentliche Schreibseite. F1 hatte nur
  `events()` öffentlich gemacht, also ausgerechnet die API, die den Audit-Trail
  trägt, blieb privat; `household` und `adopt_lineage` schrieben über `_event`.
  Der alte Name bleibt als Alias bestehen.
- **`neural_pods/__init__.py`** war über die Zeit gewachsen: 82 Zeilen, in
  denen sich Importe und ein Dutzend `__all__ +=`-Anhängsel abwechselten — die
  öffentliche Oberfläche ließ sich nur durch Ausführen lesen. Jetzt nach
  Schichten gruppiert, ein einziges `__all__`. **Die Oberfläche ist dabei
  bitgleich geblieben** (131 Namen, 69 in `__all__`, vorher/nachher
  verglichen), und drei Tests halten sie fest — einer davon prüft, dass ein
  unter „layer 2" eingeordneter Import auch wirklich in einem Storage-Modul
  liegt. Dieser Test hat sofort eine falsche Einordnung von mir gefunden
  (`adaptive_batcher` unter Schicht 1 statt 4).
- **`tests/conftest.py`** setzt die Importpfade einmal; vier Testdateien
  wiederholten das jeweils selbst und hingen damit still an ihrer Tiefe im
  Baum.

### 13.7 Was offen bleibt

Unverändert die Liste aus 12.15 — acht Eval-Reports vom Server, Bewertung des
Holdout-Splits, Neuaufzeichnung von `benchmark_taskgraph` und
`benchmark_traced_pipeline`, Lauf des KV-Affinitäts-Benchmarks,
`exact_rate`-Schwelle, `secret=` im Mesh — dazu neu:

9. **Deklarative Gate-Registry** (13.5), wenn der nächste Schwung Checks
   ansteht.
10. **Träges Laden im Paket-Init:** `import neural_pods` zieht weiterhin
    Retrieval, Raft-Transport und vLLM-Router mit. PEP 562 (`__getattr__`)
    würde das entkoppeln, ändert aber, wann ein fehlendes optionales Paket
    auffällt — eine bewusste Entscheidung, keine Aufräumarbeit.
11. **Acht Quelldateien tragen ein UTF-8-BOM** (`alias_resolver.py`,
    `block_postings.py`, `contextual_retrieval.py`, `pod_profiles.py`,
    `postgres_store.py`, `registry_lookup_kb.py` u. a.). Harmlos, bis ein
    Werkzeug ohne `utf-8-sig` liest — meine erste Importanalyse ist genau
    daran gescheitert.


## 14. Pod-Audit: die Verträge gegen den Code

Auftrag: die Architekturen und das Pod-Design prüfen und reparieren — zuerst
der Dream-Pod, dann die übrigen Pods. Danach, auf Nachfrage, drei
unabhängige Review-Agenten über das Ganze: einer gegen die
Design-Dokumente, einer adversarial gegen den eigenen Diff, einer
querschnittlich über die Architektur.

### 14.1 Der gemeinsame Nenner

Alle drei Agenten kamen unabhängig auf dieselbe Wurzel, und sie ist die
beste Formulierung, die dieses Projekt bisher für sein Kernproblem hat:

> **Aussage und Beleg entstehen im selben Arbeitsschritt, und der Beleg ist
> nicht an das gebunden, worüber er etwas aussagt.**

Ein Docstring behauptet einen Vertrag, weil er neben dem Code steht, der ihn
erfüllen soll. Ein Benchmark liefert eine Zahl, weil derselbe Commit die
Schwelle festlegt. Eine Evidenzdatei bleibt gültig, weil sie nur sich selbst
kennt. Ein Gate-Check ist grün, weil sein Prädikat und seine Eingabe
zusammen entstanden sind.

Die Gegenprobe dazu ist ebenso klar: jedes Mal, wenn das Projekt diese
Kopplung durchbrochen hat — `source_fingerprint` über den Baum, der
Meta-Test, der Evidenzdateien entfernt, das synthetische Paket für den
Schichtenprüfer, der nebenläufige Test, der das SQLite-Thread-Problem
freilegte — **hat es sofort einen echten Fehler gefunden. Vier von vier.**
Das ist keine Glückssträhne, das ist die Methode.

### 14.2 Der teuerste Einzelbefund: das Gate war vom System entkoppelt

Agent 3 hat es gemessen, nicht argumentiert: in einer Kopie des Baums je ein
Kernmodul durch eines ersetzt, das beim Import wirft, und das Gate laufen
lassen.

```
neural_pods/mesh.py         neu rot: KEINER
neural_pods/taskgraph.py    neu rot: KEINER
neural_pods/storage.py      neu rot: KEINER
neural_pods/dream.py        neu rot: KEINER
neural_pods/native_comm.py  neu rot: KEINER
neural_pods/mesh_cache.py   neu rot: KEINER
neural_pods/perception.py   neu rot: KEINER
```

44 von 45 Checks blieben grün, während das, worüber sie etwas aussagen,
nicht mehr existierte. Die einzige Kante zwischen Gate und Quellbaum war
`tests` über `sources_sha256` — und der war zu diesem Zeitpunkt selbst rot.

Der Producer-Stempel, den ich im Durchgang davor eingebaut hatte, schloss
die schmalere der beiden Lücken: *Benchmark geändert, Evidenz alt*. Die
breitere — *System geändert, Benchmark unverändert, Evidenz alt* — blieb
offen.

**Behoben:** `research/evidence.py` nimmt jetzt `subject=` — die Module,
über die die Zahlen etwas aussagen — und schreibt `subject_sha256`, gebaut
wie `record_test_run.source_fingerprint()`, also über Pfad **und** Inhalt.
Das Gate rechnet beide Hashes nach und verweigert fail-closed. Gegenprobe
gelaufen:

```
$ echo "# drift" >> neural_pods/storage.py && python research/verify_architecture_gate.py
stale evidence: storage-facade-20260920.json: the code it measured has
changed since this was recorded (neural_pods/storage.py) - re-run
benchmark_storage_facade.py
```

**Drei der 35 gelesenen Evidenzdateien haben diese Bindung** — die 14
Benchmarks, die `subject=` übergeben, müssen am Server neu laufen, damit die
Zahl steigt. Welche sie nicht haben, nennt das Gate in
`evidence_without_a_subject`; was gebunden ist, in `evidence_with_a_subject`.

*Berichtigung (Review-Runde 5).* Hier stand „zehn der 37", und im
Master-Dokument „10 von 38". Beide Zahlen waren falsch, und der Grund ist
der Befund: `evidence_without_a_subject` verlangte einen `producer`-Stempel,
und weil jede ungebundene Datei hier grandfathered — also ungestempelt — ist,
las das Feld **0**, während 32 von 35 Dateien keine Bindung hatten. Ein Feld,
das eine Lücke sichtbar halten soll, darf nicht null melden können, während
die Lücke der Normalfall ist. Es meldet jetzt beide Seiten, und ein Test
verlangt, dass die Listen alles Gelesene partitionieren.

Dazu zwei Nebenbefunde derselben Art, beide behoben:

- Die Veraltungsprüfung lief über ein **handgepflegtes 8-Tupel**, während
  das Gate 37 Dateien liest. 30 waren per Auslassung ausgenommen — der Fix
  für eine handgepflegte Tabelle war eine zweite handgepflegte Tabelle.
  `_read()` registriert jetzt jede gelesene Datei selbst; ein neuer Check
  kann keine Evidenz mehr beschaffen, die der Prüfung entgeht.
- **Ungestempelte Evidenz ließ das Gate nicht scheitern.** Die Meldung
  wurde nur ausgegeben, wenn ohnehin schon etwas anderes rot war, also nie
  dann, wenn sie gebraucht wurde. Jetzt scheitert sie; die 32 Altdateien
  stehen in `UNSTAMPED_GRANDFATHERED`, einer Sperrklinke, die nur schrumpfen
  kann und von einem Meta-Test bewacht wird.

### 14.3 Checks, die grün waren, ohne etwas zu prüfen

| Check | was er las | warum das nichts belegte |
|---|---|---|
| `reflex_dispatch` | `metrics.errors == 0` | Schlüssel wird auf 0 gesetzt und nie hochgezählt |
| `reflex_dispatch` | Qualität + `failovers == misses` | **`reflex_hits 0` von 132** — alle Antworten kamen vom Failover |
| `mesh_cache` | `principal_isolated` | Principal gelesen, unter dem nie etwas geschrieben wurde |
| `dream_reflex` | Latenz p95 | an einem Zwei-Generationen-Spielzeugpool gemessen |
| `traced_pipeline` | `reflex_frames_valid == 132` | der Benchmark parst seinen eigenen f-String |
| `mesh_presence` | `(rtt or 999) < 10` | macht aus einer legitimen 0.0 einen Fehlschlag |
| `tests` | `failed == 0` | `300 passed, 4 errors` parst zu `failed: 0` |

**Der Reflex-Befund ist der schwerste.** `research/runs/reflex-dispatch-20260919.json`
sagt wörtlich `reflex_hits: 0`, `reflex_misses: 132`, `failovers: 132`.
Kein einziges Adress-Signal löste auf; alle 132 Antworten kamen vom
Default-Pod. Das POD-ARM-Design benennt das in seinem P1-Text ehrlich
(„Hit-Rate 0.0 ist das erwartete Ergebnis"), aber die Gate-Spalte daneben
sagte `reflex_dispatch grün` — und ein grüner Check mit diesem Namen wird
gelesen, als funktioniere die Adressierung.

Der Check ist jetzt geteilt: **`reflex_failover`** belegt, was der Lauf
zeigte (jeder Miss aufgefangen, Qualität gehalten, kein Dispatch-Fehler) und
ist grün. **`reflex_dispatch`** verlangt mindestens einen aufgelösten Alias
und bleibt rot bis P5. Der Benchmark zeichnet ab sofort die rohen
Selektor-Ausgaben und die Miss-Gründe auf, damit beim nächsten Lauf
diagnostizierbar ist, welcher Schritt versagt.

### 14.4 Pod-Verträge, die nur in Docstrings existierten

Agent 1 hat alle Design-Dokumente Zeile für Zeile gegen den Code gelesen.
Befund: **von 24 in den Dokumenten zugesagten Gate-Checks existierten 9.**
Die vier blockierenden Sicherheitsbefunde, alle behoben:

1. **`EgressACL` las nie einen `link_contract`.** Der Docstring sagte „from
   its link contract", während `PodLink` gar keine Egress-Felder hatte und
   jede ACL an der Aufrufstelle handgebaut wurde. `PodLink` trägt jetzt
   `egress_topics/hosts/ports`, `EgressACL.from_link()` ist die Ableitung,
   und `next_hop()` reicht sie weiter — ein Hop, der sie fallen ließe, gäbe
   dem nächsten Pod eine leere Allowlist.
2. **ACL-Verstöße landeten nirgends.** Ein abgewiesener Frame ist ein Pod,
   der etwas adressiert, was er nicht darf — die einzige Sache in diesem
   Modul, die man später im Log finden will. Mit `registry=` wird jeder
   Verstoß zum `egress_refused`-Provenance-Event; ohne sagt `stats()`, dass
   nichts geloggt wird.
3. **„Fail-closed" galt pro Zeile, nicht pro Ausgabe.** `sudo rm -rf /`
   gefolgt von einem wohlgeformten `PUB` publizierte. Eine Ausgabe, von der
   der Parser eine Zeile nicht lesen konnte, ist keine verstandene Ausgabe.
   `strict=True` (Vorgabe) verweigert sie ganz; `strict=False` bleibt für
   die Rate-Messung, die die partiellen Fälle sehen muss.
4. **`hop_budget` und `deadline_ms` standen im `PodLink` und kamen nie auf
   den Draht.** Jeder Mesh-Hop baute einen frischen Link mit frischem
   Budget — ein Ring von Pods konnte Arbeit unbegrenzt im Kreis schicken,
   während jeder einzelne Hop validierte. Beide reisen jetzt im Envelope,
   mit der **ursprünglichen** `ts_ms` (ein Neustempeln pro Hop gäbe jedem
   Hop die volle Frist zurück, derselbe Fehler eine Ebene tiefer). Die
   Entscheidung liegt in `envelope_refusal()`, modulweit und rein, damit
   die Regel ohne Broker testbar ist — eine Regel, die nur gegen lebendes
   MQTT läuft, läuft nie.

Dazu die Isolationsbehauptung: `MeshCache`s „principal isolation" war
Schlüsselableitung, sonst nichts. Der Parameter `principal=` kam vom
Aufrufer, also las jeder Pod jeden Principal. Jetzt gibt es eine
Aufrufer-seitige Prüfung (`allowed_principals`, `PrincipalRefused`) **und**
den Satz, dass das keine Sicherheitsgrenze ist: eine Redis-Datenbank, ein
Credential, jeder Prozess mit diesem Credential liest alles. Der Test, den
der Benchmark hätte fahren müssen, steht jetzt in
`tests/test_mesh_cache.py` und **bestätigt die Lesbarkeit**, statt sie
wegzuassertieren.

### 14.5 Defekte im eigenen Diff des vorigen Durchgangs

Agent 2 hat den uncommitteten Diff adversarial geprüft und elf Befunde
belegt, die meisten mit ausgeführtem Reproduktionsskript. Alle behoben:

| # | Befund | Warum es zählt |
|---|---|---|
| 1 | Die Autonomie-Quote zählte nur **erfolgreiche** Zyklen | Jeder Abbruch hatte Pool, Backtest und alle Kandidaten bezahlt; 10 Läufe, Quote unverändert 0/2 |
| 2 | `drain()` meldete Erfolg, während das letzte Event im Consumer war | 200 von 200 reproduziert; der „lossless"-Nachweis baute darauf |
| 3 | Das RAM-Lease war eine **selbst ausgestellte Quittung** | Governor in derselben Funktion gebaut, Budget vom Aufrufer genannt — konnte nicht ablehnen |
| 4 | `dream_reflex` grün am Spielzeugpool | siehe 14.3 |
| 5 | Abgelehntes `add_generation` schrieb `_max_order` fort | Die legitime nächste Generation wurde mit Verweis auf eine nie belegte Position abgewiesen |
| 6 | Perception-Evidenz fiel durch alle Netze | Kein Check las sie, kein Stempel, erzeugt vom alten Drain-Loop — **zurückgezogen** |
| 7 | Der Producer-Stempel konnte die Drift nicht sehen, die er behauptete | siehe 14.2 |
| 8 | `resolve_p95_ms` schloss die **fehlgeschlagenen** Auflösungen aus | Ein Kanal, der öfter danebengreift, sah schneller aus |
| 9 | `CycleBudget` behauptete die Registry-Uhr, nahm die des Aufrufers | Mit eingespeister Uhr band die Quote nie |
| 10 | Ein werfender Consumer tötete den Drain-Thread lautlos | `has_consumer: True`, `consumed` eingefroren, für immer |
| 11 | Ein von der Queue verworfenes Event verbrauchte Rate-Budget | Der nächste legitime Emit wurde für ein nie angenommenes Event abgewiesen |

Dazu Kleineres: zwei unbegrenzt wachsende Latenz-Listen, die `stats()` bei
jedem Aufruf sortierte; `factory.create()` vor der Duplikatprüfung (ein
abgelehntes Re-Activate lud erst das komplette Modell von Platte);
`stats()["budget_bytes"]` im Governor-Modus `None`; die `ts`-Migration
außerhalb der Sperre; ein als Hit gezählter Cache-Eintrag, bevor
`json.loads` scheitern konnte; ungesperrte Zähler über den Netzwerk-Thread.

**Das ist der zweite Durchgang in Folge, in dem ein adversarialer Review
meines eigenen Diffs zweistellig viele echte Fehler findet.** Beim ersten
Mal waren es acht, jetzt elf. Der Unterschied zum Durchgang davor ist
nicht, dass weniger Fehler entstanden — es ist, dass sie gefunden wurden,
bevor sie in einem Commit landeten.

### 14.6 Was Agent 3 als tragfähig bezeichnet und nicht angefasst werden soll

Der Vollständigkeit halber, weil ein Prüfbericht, der nur Mängel auflistet,
ein unvollständiger Prüfbericht ist:

1. `tests/test_architecture_gate.py::test_removing_an_evidence_file_reddens_exactly_die_checks_that_read_it`
   — leitet die Abhängigkeiten per AST aus `verify()` ab, entfernt jede
   Evidenzdatei einzeln und verlangt exakt die passenden roten Checks. Er
   prüft die Prüfung. Er ist die Vorlage für alles, was in 14.2 dazukam.
2. `neural_pods/architecture.py` als Konstruktion — Architekturregel als
   Daten, statische Prüfung, Gegenprobe an einem synthetischen Paket. Die
   Behandlung von `if TYPE_CHECKING:` als Nicht-Laufzeitkante ist korrekt
   und subtil.
3. `record_test_run.source_fingerprint()` — das einzige Konstrukt, das
   Evidenz an den geprüften Baum band, bevor 14.2 es verallgemeinerte.
4. Die Registry als Nahtstelle — 91 konsumierende Dateien, thread-sicher,
   öffentliche Event-API. Die eine Stelle, an der die Architektur hält,
   was sie verspricht.
5. Die Ehrlichkeitskultur der Docstrings und ADRs. Wörtlich: *„Das ist
   selten und wertvoll — und es ist der Grund, warum die Befunde oben
   überhaupt auffindbar waren."*

### 14.7 Was offen bleibt, benannt statt verschwiegen

- **`perception_stream`, `improve_cycle`, `latent_addressing`** — die
  Perceptions-Messung braucht den MQTT-Broker, P4 und P5 sind nicht gebaut.
  Die Phasentabelle in `POD-ARM-DESIGN-20260919.md` sagt das jetzt, statt
  Check-Namen zu führen, die es nie gab.
- **Die Vier-Tier-Fassade implementiert zwei Tiers.** L0 und L3 kommen in
  `storage.py` nur im Docstring vor. Drei unabhängige L1-Implementierungen
  (`PodCache`, `MeshCache`, `PodStorage`), keine benutzt eine andere. Im
  Storage-Dokument berichtigt; die Entscheidung, welche bleibt, steht aus.
- **`neural_pods/` hat 56 Module, 31 davon ohne jede paketinterne Kante.**
  Der `layering`-Check ist korrekt implementiert und fängt derzeit fast
  nichts, weil zwischen den Schichten kaum Kanten verlaufen. Das liegt am
  Baum, nicht am Prüfer.
- **85 Benchmark-Skripte, 6 von einem Test erreichbar.** 6.575 Zeilen
  Messcode, an denen die gesamte Gate-Evidenz hängt, sind die am
  schlechtesten geprüfte Schicht im Repo — und fünf der acht Review-Befunde
  aus 13.2 lagen genau dort. `research/benchmark_xgboost_pod.py` ist der
  erste, dessen Kern als `measure()` importierbar ist und einen eigenen
  Test hat (`tests/test_benchmark_xgboost_pod.py`). Das ist ein Anfang von
  85, und es ist die wirksamste offene Maßnahme.
- **26 Testdateien importieren torch hart**, `pytest` bricht im frischen
  Klon mit 26 Collection-Errors ab. Kein `gpu`/`broker`/`server`-Marker.
  Solange das so ist, kostet Neuaufzeichnen Server-Zugang, wird
  aufgeschoben, und genau daraus entsteht eingefrorene Evidenz.

### 14.8 Stand nach diesem Durchgang

```
Tests        454 grün · 0 rot · 0 errors · 8 übersprungen
Gate         47 Checks · 37 grün · 10 rot
             davon 7 mangels Server-Evidenz
             davon 3 zu Recht rot (14.3), vorher grün ohne Beleg
Schichten    56 Module · 0 Verstöße
Evidenz      3 von 35 Dateien an den gemessenen Code gebunden
```

## 15. Messcode testbar machen

Die wirksamste offene Maßnahme aus 14.7, umgesetzt. Ausgangslage: **85
Benchmark-Skripte, 6.575 Zeilen, erzeugen die gesamte Gate-Evidenz; sechs
davon sind von einem Test erreichbar.** Fünf der acht Review-Befunde aus
13.2 lagen in dieser Schicht.

### 15.1 Warum ausgerechnet dort

Das Projekt kennt zwei Codequalitäten, und die Grenze verläuft nicht entlang
der Wichtigkeit, sondern entlang der **Importierbarkeit**:

| | Test/Code-Verhältnis |
|---|---|
| `neural_pods/` — importierbarer Bibliothekscode | 0,43 |
| `research/` — Skripte ohne Einstiegspunkt | ≈ 0,05 |

Ein Benchmark, der als `if __name__ == "__main__"`-Monolith geschrieben ist,
bietet nichts zum Testen an. Und weil die einzige Instanz, die seine Ausgabe
liest — das Gate — nur Zahlen gegen Schwellen vergleicht, fällt eine
**falsch messende Messung** nie auf. Das ist die Wurzel der Fehlerklasse
„Kennzahl misst etwas anderes als ihr Name sagt".

### 15.2 Das Muster

```
collect()    braucht Broker, Redis, LXD, GPU.
             Misst und gibt Beobachtungen zurück. Zieht keine Schlüsse.
summarise()  rein. Macht aus den Beobachtungen das Urteil,
             das das Gate liest. Testbar ohne alles.
main()       verdrahtet, stempelt, schreibt.
```

Entscheidend ist die Richtung der Tests. Ein Test, der nur den grünen Fall
zeigt, belegt, dass die Rechnung *läuft* — nicht, dass sie *unterscheidet*.
Jede der unten stehenden Suiten füttert deshalb auch einen Beobachtungssatz,
bei dem das Urteil **kippen muss**.

### 15.3 Stand: alle 16 gate-relevanten Skripte

69 Skripte schreiben Evidenz, aber nur 16 schreiben Evidenz, die das Gate
liest — und das Gate entscheidet über Promotion. Das ist die Menge, die
zählt, und sie ist vollständig:

| Skript | reine Funktionen | Tests |
|---|---|---|
| `benchmark_taskgraph` | `summarise`, `collect` | 9 |
| `benchmark_traced_pipeline` | `summarise`, `percentile` | 10 |
| `benchmark_storage_facade` | `summarise`, `collect` | 12 |
| `benchmark_mesh` | `summarise`, `percentile` | 4 |
| `benchmark_mesh_cache` | `summarise` | 4 |
| `benchmark_mesh_e2e` | `summarise` | 5 |
| `benchmark_native_tcp` | `summarise`, `percentile` | 4 |
| `benchmark_redis_cache_tier` | `summarise_tier`, `percentile` | 2 |
| `benchmark_reflex_dispatch` | `summarise`, `percentile` | 4 |
| `benchmark_ensemble_router` | `summarise` | 4 |
| `benchmark_hetero_ensemble` | `summarise` | 2 |
| `benchmark_xgboost_pod` | `measure` | 6 |
| `benchmark_dream_reflex` | `measure` | 3 |
| `benchmark_kvcache_affinity` | `summarize`, `compare`, `measure` | vorhanden |
| `generate_holdout_split` | `render` | vorhanden |
| `record_test_run` | `source_fingerprint`, `run_pytest` | vorhanden |

### 15.4 Was der Umbau freigelegt hat

Fünf Defekte, die nicht gesucht, sondern beim Zerlegen sichtbar wurden:

1. **`benchmark_storage_facade` prüfte mit `assert` innerhalb der Messung.**
   Ein Assert, der feuert, tötet den Lauf — der eine Fall, der
   aufzeichnenswert ist (*die Fassade hat es falsch gemacht*), erzeugte einen
   Traceback und **keine Evidenz**. Das Gate meldet dann „keine Evidenz"
   statt „Evidenz sagt nein". Das sind verschiedene Befunde mit verschiedenen
   Konsequenzen, und die Unterscheidung war genau der Punkt, den P0 am Gate
   repariert hatte — im Benchmark war sie nie angekommen.
2. **`quoted_key_roundtrip` stand als Literal `True` im Bericht.** Der
   Gate-Check `is True` darauf konnte nicht fehlschlagen — dieselbe Klasse
   wie `reflex_frames_valid == 132`. Er wird jetzt aus einer Zählung
   abgeleitet, und mein erster Lauf nach dem Umbau zeigte sofort, dass ich
   400 Round-Trips gegen 200 Schlüssel verglichen hatte.
3. **`benchmark_mesh_e2e` meldete `pod_b_validated_all: true` für einen
   Lauf, in dem gar kein Ack ankam.** `all()` über ein leeres Dict ist
   `True`. Die Anzahl wird jetzt mitgeprüft.
4. **`benchmark_hetero_ensemble` konnte `fallback_used > n` erzeugen.** Ein
   fehlgeschlagener Fallback-Aufruf übersprang den Fall — nachdem
   `fallback_used` bereits hochgezählt war. Die beiden Zähler beschrieben
   verschiedene Fallmengen.
5. **`benchmark_ensemble_router` enthielt eine Tautologie in einer
   Promotionsmetrik:** `g3_raw and row['target'] == row['target']`. Zur
   Genauigkeit: der Ausdruck reduziert sich auf `g3_raw`, die aufgezeichneten
   **Zahlen waren also nie falsch**. Falsch war, dass ein Leser nicht
   erkennen konnte, welche Bedingung gemeint war — und eine Bedingung, die
   nicht falsch werden kann, ist von einer Absicherung, die still aufgehört
   hat abzusichern, nicht zu unterscheiden.

Dazu eine Berichtigung im Bericht selbst: `mesh-presence` schreibt jetzt
`rtt_is_cross_node: false` in die Evidenz. Die 0,30 ms wurden zwischen zwei
Endpunkten **auf demselben Host** gemessen; nur der Broker ist entfernt. Die
Zahl war im Master-Dokument und in drei Design-Dokumenten als
knotenübergreifende Mesh-Latenz zitiert (siehe 3.4).

### 15.5 Was das an der Fehlerklasse ändert

Die Kernaussage aus 14.1 lautet: *Aussage und Beleg entstehen im selben
Arbeitsschritt, und der Beleg ist nicht an das gebunden, worüber er etwas
aussagt.* Abschnitt 14.2 hat die zweite Hälfte adressiert (`subject_sha256`
bindet die Evidenz an den gemessenen Code). Dieser Abschnitt adressiert die
erste: **die Rechnung, die aus Beobachtungen ein Urteil macht, ist jetzt von
der Messung getrennt und kann gegen bekannte Antworten geprüft werden.**

Was das nicht leistet: `collect()` bleibt ungetestet, und damit die Frage,
ob die Beobachtungen selbst stimmen. Ein `collect()`, das die falschen
Zahlen sammelt, produziert weiterhin ein korrekt gerechnetes falsches
Urteil. Diese Hälfte braucht den Server.

### 15.6 Stand nach diesem Durchgang

```
Tests        517 grün · 0 rot · 0 errors · 8 übersprungen
             davon 63 neu über Messcode, der vorher nicht erreichbar war
Gate         47 Checks · 37 grün · 10 rot (unverändert, siehe 14.8)
Schichten    56 Module · 0 Verstöße
Messcode     16 von 16 gate-relevanten Skripten importierbar
             69 Skripte schreiben Evidenz insgesamt; die übrigen 53
             schreiben nichts, was das Gate liest
```

## 16. Der erste externe Prüfer

PR #1 hat keine CI (siehe 15.6 und den offenen Punkt unten), also war
CodeRabbit der erste maschinelle Prüfer, der über diesen Diff gelaufen ist.
Er meldete **11 Inline- und 3 Outside-diff-Befunde**. Ich habe jeden einzeln
gegen den Code geprüft und drei ausführbar reproduziert.

**Alle 14 waren echt. Kein einziger Fehlalarm.**

### 16.1 Wo sie saßen, und warum das kein Zufall ist

Fast ausnahmslos in Pfaden, die in dieser Umgebung nie ausgeführt werden:
der Fehlerzweig eines Benchmarks, der zwei GPUs braucht; eine Migration für
eine gewachsene Tabelle, die es hier nicht gibt; ein Envelope mit kaputtem
Feld, den nur ein echter Sender schickt.

Das ist wörtlich die Lücke, die Abschnitt 15.5 benannt hatte: *„`collect()`
bleibt ungetestet … diese Hälfte braucht den Server."* Ein externer Prüfer,
der den Code liest statt ihn auszuführen, greift genau dort hin. Elf der
vierzehn Befunde lagen in Code aus meinen eigenen Commits dieser Session.

### 16.2 Die vier Major-Befunde

1. **`architecture.py` warf `KeyError`, statt die unbekannte Schicht zu
   melden.** `unknown_layers` wird berechnet und als Verstoß ausgegeben — die
   Absicht ist eindeutig. Die Schleife darunter wachte aber über
   `module not in layer_of` statt über die Zugehörigkeit zu `rank`. Ein
   Tippfehler in `LAYER_OF` ließ das Gate mit einem **Traceback abbrechen**,
   statt `layering` rot zu melden: ein Absturz, wo ein fail-closed-Urteil
   hingehört. Reproduziert mit `LAYER_OF["dragonfly"] = "kernn"`.
   Präzisierung, die der Bot nicht hatte: nur Module mit *ausgehenden*
   Kanten lösen aus — 17 von 56. Mein erster Versuch mit einem Fan-out-0-
   Modul lief durch und bewies nichts.

2. **Die Mesh-Bounds wurden außerhalb des Schutzblocks ausgewertet.**
   `envelope_refusal` rechnet `int()`/`float()` auf **Draht-Daten**. Ein
   Sender mit `hop_budget: "x"` erzeugte einen `ValueError` im
   paho-Callback; paho 2.1.0 reicht Callback-Ausnahmen in seinen
   Netzwerk-Loop weiter (`suppress_exceptions` ist standardmäßig `False`),
   was den Empfangs-Thread beenden kann. Die kaputte Nachricht umging
   zusätzlich `bad_envelopes`. **Ein Feld, das ich zur Begrenzung von
   Anfragen eingeführt habe, war ein Weg, den empfangenden Pod
   stillzulegen.** Behoben über das, was der Bot vorschlug, **und** darüber
   hinaus: „malformed" ist jetzt ein benanntes Ergebnis statt einer
   Ausnahme, sodass die Sendeseite (`forward()`, die denselben Fehler hatte)
   mitprofitiert und die Regel ohne Broker prüfbar ist.

3. **`storage.py`: Altbestand und der NULL-Fall.** Nachgemessen gegen
   lancedb 0.39.0 — und schlimmer als gemeldet. Der Bot schrieb „does not
   evolve the schema", was nach stillem Verlust klingt. Tatsächlich:

   ```
   ValueError: Field 'l1_eligible' not found in target schema
   ```

   `merge_insert` castet hart gegen das Tabellenschema, `allow_subschema`
   erlaubt nur *weniger* Spalten. Auf einer `kv`-Tabelle von vor diesem PR
   hätte **jedes `put()` geworfen**. Dazu die zweite Hälfte: nach der
   Migration existiert die Spalte und Altzeilen tragen `NULL` — und
   `dict.get(key, default)` greift nur bei **fehlendem Schlüssel**. `None`
   ist falsy, der Backfill entfiel, und der Kommentar direkt darüber sagte
   das Gegenteil. Beide Hälften in einem Commit, denn nur (a) ausgeliefert
   erzeugt genau die NULL-Zeilen, die (b) braucht.

4. **Der Fallback-Aufruf im Ensemble-Router war ungeschützt**, während der
   Primär-Aufruf es war. Ein Replica-Ausfall verwarf **alle bis dahin
   gesammelten Beobachtungen**, keine Evidenzdatei entstand. Ich hatte genau
   das in `benchmark_hetero_ensemble.py` im selben Commit repariert.

### 16.3 Der Befund, der am meisten über die Methode sagt

Zwei der vierzehn waren **Assertions in meinen eigenen, neu geschriebenen
Tests, die nicht fehlschlagen können**:

```python
assert str(subject_file.relative_to(REPO)) in data["subject"] or data["subject"]
```

Python liest das als `(x in liste) or (liste)`. Eine nichtleere Liste ist
wahr — die Assertion galt, ob der Pfad enthalten war oder nicht.
Nachgestellt:

```
"neural_pods/voellig_falsch.py" in ['neural_pods/storage.py'] or [...]
  -> ['neural_pods/storage.py']   bool: True
```

Der zweite Fall war subtiler: ein Test rechnete im `except SystemExit`-Zweig
die Gate-Logik **selbst nach**, statt das Feld zu lesen, das er prüfen
sollte. Da das Gate auf diesem Baum rot ist, lief immer dieser Zweig — und
die Schlussassertion verglich zwei Mengen, die per Konstruktion disjunkt
sind. Auch nicht fehlschlagbar. **Diesen fand CodeRabbit nicht; ich fand ihn
beim Nachziehen der Klasse.**

Das ist dieselbe Fehlerklasse wie `quoted_key_roundtrip: True` und
`reflex_frames_valid == 132` — die ich in diesem PR aus den Benchmarks
entfernt und gleichzeitig in die Tests eingebaut habe. Ein Test, der nicht
fehlschlagen kann, ist schlimmer als ein fehlender: er erzeugt die
Zuversicht, ohne die Prüfung zu leisten.

### 16.4 Was daraus an Struktur entstanden ist

Drei Meta-Tests, jeder mit ausgeführter Gegenprobe:

| Test | prüft | fängt |
|---|---|---|
| `test_assertions_can_fail.py` | jede Assertion der Suite per AST | `assert x or <literal>` und `assert x in c or c` |
| `test_benchmark_subject_scope.py` | jedes Benchmark mit `subject=` | `SUBJECT` in einem String statt auf Modulebene |
| `test_design_docs_match_the_gate.py` | jedes Design-Dokument gegen das echte Gate | ein `✔` an einem roten Check |

Der letzte adressiert den Fehler, den dieses Projekt am häufigsten gemacht
hat. Der erste fand beim ersten Lauf nebenbei, dass
`tests/test_routing_harness.py` ein **UTF-8-BOM** trägt und sich gar nicht
parsen ließ — meine Regel hätte sonst nur die Dateien geprüft, die zufällig
parsen, was dieselbe Fehlerklasse gewesen wäre.

Dazu strukturell: **der Gate-Bericht reist jetzt mit der Ablehnung.** Ein
rotes Gate hatte nur eine Textmeldung, also musste alles, was ein Feld
daraus brauchte, die Gate-Logik nachrechnen — und eine Nachimplementierung
ist keine Prüfung dessen, was sie nachimplementiert. Genau daraus entstand
der zweite vakuume Test.

Und: `benchmark_storage_facade.py` legt jetzt selbst eine Alt-Tabelle an und
misst die Migration, sodass `storage_facade` den Fall sieht. Er war
unsichtbar, weil der Benchmark jedes Mal in einem frischen Temp-Verzeichnis
startet und deshalb nie auf einen gewachsenen Bestand trifft.

### 16.5 Was ich nicht gemacht habe

**Docstring-Coverage 43,28 % gegen eine Schwelle von 80 %** (Pre-Merge-
Warnung). Das ist eine Bot-Richtlinie, kein Defekt. Die Zahl zählt 372
Funktionen aus dem Diff, darunter jede Testfunktion und jeden Stub-Helfer.
Dieses Repo hat die umgekehrte Konvention: lange, begründende Docstrings
dort, wo eine Entscheidung erklärt werden muss, und gar keine an einem
dreizeiligen Fake-Redis. 200 Funktionen mit Füllsätzen zu versehen würde die
Signalqualität der vorhandenen senken — und die ist der Grund, warum die
Befunde in diesem Projekt auffindbar sind.

### 16.6 Stand nach diesem Durchgang

```
Tests        675 grün · 0 rot · 0 errors · 8 übersprungen
             davon 158 neu in diesem Durchgang
Gate         47 Checks · 37 grün · 10 rot (unverändert, siehe 14.8)
Schichten    56 Module · 0 Verstöße
Meta-Tests   3 neu, jeder mit ausgeführter Gegenprobe
```

**Die Lehre, als offener Punkt formuliert:** dass ein externer Prüfer 14 von
14 richtig lag und zwei davon in frisch geschriebenen Tests saßen, heißt,
dass der teuerste blinde Fleck dieses Projekts nicht mehr der ungetestete
Messcode ist, sondern **die Prüfungen selbst**. Die drei Meta-Tests sind ein
Anfang. Eine systematische Mutationsprobe — eine Behauptung verfälschen und
nachsehen, ob irgendein Test rot wird — wäre der nächste Schritt.

## 17. Die zweite Runde: die Prüfungen prüfen

Derselbe externe Prüfer hat `772e821` — meinen Fix-Commit für Runde 1 —
erneut gelesen und **sechs** Befunde gemeldet. Ich habe jeden gegen den Code
geprüft und zwei ausführbar nachgestellt. **Alle sechs stimmen. Über beide
Runden: 20 von 20 Befunden echt, kein einziger Fehlalarm.**

Das Muster dieser Runde ist das eigentliche Ergebnis. Vier der sechs liegen
in Code, den ich in Runde 1 geschrieben habe — und **drei davon in den drei
Meta-Tests, die ich gebaut hatte, um genau diese Fehlerklassen zu
schließen.** Abschnitt 16.6 endete mit der Vermutung, der teuerste blinde
Fleck seien inzwischen die Prüfungen selbst. Diese Runde belegt das.

### 17.1 Zwei fail-open-Stellen im Gate

**Ein Backtest ohne Vorhersage galt als bestanden.** `_dream_backtest_ok`
hatte zwei Auswege: ein Lauf ohne Out-of-sample-Generationen gab `True`
zurück, und der Legacy-Zweig akzeptierte die In-sample-Zahl, die **der
Docstring derselben Funktion** als nicht fehlschlagbar beschreibt.

Behoben durch dieselbe Aufspaltung wie bei `reflex_failover`/
`reflex_dispatch` in Runde 1 — ein Check, eine Behauptung:

| Check | belegt |
|---|---|
| `dream_pipeline` | der Zyklus **lief**: abgeschlossen, Gewinner identifizierbar, Backtest in der Form, die fehlschlagen kann |
| `dream_predictive` (neu) | der Simulator hat **vorhergesagt**: mindestens eine zurückgehaltene Generation, Fehler unter 0,15 |

Der Legacy-Zweig gibt jetzt `False` zurück. Damit ist **beides rot**, denn
die eingecheckte `dream-cycle-20260920.json` trägt genau diese alte Form.
Das ist das ehrliche Ergebnis, nicht das bequeme: der Zyklus muss auf der
Maschine mit den Generationsberichten neu laufen, und bis dahin wäre ein
grünes `dream_pipeline` ein Urteil über eine Zahl, die nicht fehlschlagen
kann. Abschnitt 4 hatte genau das gemessen: 0 von 4 Generationen
out-of-sample vorhersagbar.

**Fehlende Testevidenz fiel auf die Zahl vom 2026-09-17 zurück.** Ohne
`tests-20260920.json` bestand `tests` anhand eines Zählstands aus einem
Dokument, das Tage vor dem Code geschrieben wurde — ohne Quellcode-Hash.
Das war mein eigener Befund aus einer früheren Runde, halb behoben: der
Kommentar im Gate sagte wörtlich „*which is exactly the blind spot*", und
der Fallback stand weiter da. Die Datei wird jetzt wie jede andere Evidenz
als fehlend gemeldet.

### 17.2 Die drei Meta-Tests

**Der schärfste Befund.** `test_design_docs_match_the_gate.py` sollte
verhindern, dass ein Dokument einen Namen als Gate-Check ausgibt, den das
Gate nicht kennt. Die Regel lautete:

```python
if re.fullmatch(r"[a-z]+(?:_[a-z0-9]+)+", name) and name in KNOWN_INVENTED:
```

`KNOWN_INVENTED` war eine **handgepflegte Allowlist** mit vier Namen. Nur
wer schon draufstand, wurde bemängelt; jeder neu erfundene Name passierte
ungeprüft. Das ist wörtlich der systemische Befund S3 aus der
Querschnittsanalyse — *„der Fix für eine handgepflegte Tabelle war eine
zweite handgepflegte Tabelle"* — von mir reproduziert **in dem Test, der
diese Klasse schließen sollte**.

Die Liste ist ersatzlos weg. Stattdessen wird die Gate-Check-**Spalte** der
Phasentabellen geparst (die Kopfzeile wird an der Trennzeile darunter
erkannt, nicht am Text) und jeder dort genannte Name zurückgewiesen, den das
Gate nicht definiert. Ein Name mit ehrlichem Marker daneben — `(offen)`,
`existiert nicht`, `(geplant)` — ist keine Behauptung und bleibt erlaubt.

Die umgedrehte Regel hat sofort **elf** Zeilen gefunden, die die alte nicht
sah: sieben Phasen im Haushalts-Design, vier im Speicher-Design, alle mit
einem Check-Namen in der Spalte und keinem Check dahinter. Sie tragen jetzt
`(geplant)`. Die Gegenprobe steht als Test: ein frei erfundener Name in
einer Phasentabelle wird rot, ohne dass irgendeine Liste ihn nennt.

**Die Assertion-Regel war zu breit.** `_redundant_operand` fragte, ob ein
späterer Operand **irgendwo** im früheren vorkommt. Nachgestellt:

```
assert transform(value) or value     geflaggt=True
```

Diese Assertion kann fehlschlagen — wenn beide Seiten falsy sind. Mein
eigener Docstring in derselben Datei sagt, warum das schlimm ist: *„eine
breite Heuristik … würde innerhalb einer Woche abgeschaltet."* Die Regel
prüft jetzt nur noch die belegte Form `x in c or c` (`ast.Compare` mit
`ast.In`/`ast.NotIn`, dessen Container einem späteren Operanden entspricht).
Gegenproben in beide Richtungen: die Originalzeile aus Runde 1 wird weiter
gefangen, `transform(value) or value` nicht mehr.

**Der Struktur-Test prüfte zu wenig.** Er wies nur ein *Literal* als
`status` zurück — ein `status`, der von der falschen Variablen abhängt,
bestand ihn. Ersetzt durch direkte Fälle gegen `summarise()`.

### 17.3 Der Benchmark dahinter

`benchmark_perception.py` war das einzige der Messskripte, das die
`collect()`/`summarise()`-Behandlung aus Abschnitt 15 nicht bekommen hatte —
und es enthielt prompt zwei Defekte: `throughput_events_s` rechnete
`COUNT / elapsed` statt aus dem Zugestellten (ein Lauf, der 900 von 2000
Ereignissen lieferte, meldete den vollen Durchsatz), und `len(mesh_received)`
wurde zweimal zu verschiedenen Zeitpunkten gelesen. Jetzt: ein Schnappschuss
nach dem `wait()`, jede Kennzahl aus dem Zugestellten, und `no_deadlock`
vergleicht Versuche gegen Kapazität statt gegen die Konstante 45.

### 17.4 Stand nach diesem Durchgang

```
Tests        685 grün · 0 rot · 0 errors · 8 übersprungen
Gate         48 Checks · 36 grün · 12 rot (2 neu rot, beide zu Recht)
Schichten    56 Module · 0 Verstöße
Doku         11 geplante Checks als geplant markiert
```

**Bewusst nicht gebaut: ein vierter Meta-Test**, der Meta-Tests auf
Allowlists prüft. Das wäre derselbe Reflex, der `KNOWN_INVENTED`
hervorgebracht hat. Was eine Prüfung wirklich prüft, ist eine
**Mutationsprobe**: eine Behauptung verfälschen und nachsehen, ob irgendein
Test rot wird. Genau das ist bei jedem Punkt dieser Runde einzeln gemacht
worden — systematisch über das ganze Repo ist es der nächste Schritt und
bleibt offen.

Und der teuerste offene Punkt ist unverändert: **26 Testdateien importieren
torch hart, es gibt keine CI, und damit bleibt ein externer Bot der einzige
maschinelle Prüfer dieses Repos.** Nach zwei Runden mit 20 von 20 echten
Befunden ist das keine Randnotiz.

### 17.5 Die dritte Runde, in derselben Datei

`e78cde4` ist noch nicht kalt gewesen, da meldete derselbe Prüfer **fünf**
weitere Befunde — vier inline, einen außerhalb des Diffs. Alle fünf geprüft,
zwei ausführbar nachgestellt, **alle fünf echt**. Über drei Runden: **25 von
25, kein Fehlalarm.**

Drei davon liegen in Code, den ich in Runde 2 geschrieben habe, und zwei
wieder in den Meta-Tests. Das Muster hat sich verschärft: **die Korrektur
einer fail-open-Stelle hat eine neue erzeugt, und die Gegenprobe, die im
selben Commit dafür geschrieben wurde, konnte sie nicht fangen.**

**Die gemessene Null.** `(out.get("max_abs_error") or 1.0) < 0.15` — `0.0`
ist falsy. Ein Lauf, der eine zurückgehaltene Generation *exakt* vorhersagt,
machte `dream_predictive` rot: das einzige Ergebnis, das die Vorhersagekraft
belegen würde, war das einzige, das der Check ablehnte. Die Zeile stammt aus
dem alten Code und ist bei der Aufspaltung mitgewandert, ohne geprüft zu
werden. Fehlende Messung und gemessener Wert sind jetzt getrennt.

**Der Marker, der die ganze Zeile entlastete.** `_gate_check_claims` gab
jedem Namen die komplette Tabellenzeile als Kontext. Nachgestellt:

```
| H9 | Zwei Checks | `household_join` (geplant) · `cortex_map` |
gefunden:  ['household_join', 'cortex_map']
gemeldet:  []                                (erwartet: cortex_map)
```

Ein ehrlich markierter Name deckte jeden unmarkierten neben sich. Der Kontext
reicht jetzt vom Namen bis zum nächsten; Text **vor** dem ersten Namen
(„geplant: `a`, `b`, `c`") gilt für alle, weil er zu keinem einzelnen gehören
kann. In Prosa darf der Marker auch vor dem Anker stehen — „D3 gebaut,
Gate-Check rot: … `dream_reflex`" —, dort bekommt nur der *erste* Name den
Vorlauf.

**Die Gegenprobe, die den Filter nicht aufrief.** Sie prüfte, dass
`_gate_check_claims` den erfundenen Namen *extrahiert*, und hörte da auf. Die
Entscheidung — nicht definiert **und** kein Marker — stand nur im
Produktivtest. Deshalb ist der Defekt darüber durch sie hindurchgegangen.
Beide gehen jetzt durch `_invented_claims`, und der Gegenprobe-Fall enthält
genau die Falle: ein markierter neben einem unmarkierten Namen, in **einer**
Zeile.

**Die Fixture, die `tests` per Konstruktion rot machte.** `evidence_dir`
kopierte nur `research/runs` nach `tmp_path`; das Gate berechnet seinen
Quellcode-Fingerabdruck aber über `parents[2]`, also `/tmp`. Gemessen, mit
vorhandener Evidenzdatei: `tests check in fixture: False`. Damit bestanden
**drei** Tests aus dem falschen Grund — die drei parametrisierten Fälle aus
Runde 1, die belegen sollten, dass ein Lauf mit `exit_code 1` abgelehnt wird,
konnten nichts belegen, weil der Check ohnehin rot war. Die Fixture spiegelt
den Baum jetzt per Symlink. (Die drei Verzeichnisse als Ganzes zu verlinken
reicht nicht: `rglob` steigt nicht in verlinkte *Unter*verzeichnisse ab, 55
Dateien fehlten und der Fingerabdruck blieb falsch. Verzeichnisse echt,
Dateien verlinkt.)

**Und die Landkarte selbst.** `ARCHITECTURE-MASTER` führte unter der
Überschrift „Implementiert + **validiert** (Gate-Checks)" drei rote Checks —
`dream_pipeline`, `mesh_cache`, `reflex_dispatch` —, bei `dream_pipeline` mit
der Wahrheit in der Nachbarspalte (`out-of-sample: 0 Generationen`). Kein
Test sah das: die Namen standen **ohne Backticks**, und `TICKED` erkannte nur
`✔`, nicht `✓`. Drei Ergänzungen, keine davon eine Liste:

* in der Gate-Check-Spalte zählen auch Namen ohne Backticks;
* `TICKED` erkennt beide Haken;
* eine dritte Regel: **ein Check, den das Gate rot führt, braucht einen
  ehrlichen Marker neben seinem Namen.** Sie hat sofort vier weitere Stellen
  gefunden — die D2-Zeile im Dream-Design und, schärfer, den N4-Status im
  Nervensystem-Design, der immer noch „Gate-Check `mesh_cache` grün → Gate
  37/37" behauptete, obwohl der Pod-Audit den Check vor zwei Commits rot
  gemacht hatte.

Ausgenommen ist genau ein Check, `tests`, und der Grund ist nachgeprüft statt
behauptet: er vergleicht `sources_sha256` gegen den **Arbeitsbaum**, ist
also in jedem Checkout mit einer nicht aufgezeichneten Änderung rot. Ein Test
liest per AST aus dem Gate, welche Checks `source_fingerprint` aufrufen, und
besteht nur, wenn diese Menge genau der Ausnahmeliste entspricht.

### 17.6 Was diese Runde gelehrt hat

Eine Regel, und sie ist Befund 3 verallgemeinert:

> **Eine Gegenprobe muss denselben Codepfad aufrufen wie die Prüfung, die sie
> absichert.** Sonst belegt sie, dass ein Zwischenschritt funktioniert — und
> genau im übersprungenen Stück saß dann der Defekt.

Kein vierter Meta-Test, keine neue Ebene. Die drei Regeln in
`test_design_docs_match_the_gate.py` stehen nebeneinander (Haken, Existenz,
Rot-Stand) und teilen sich eine Zerlegung und einen Filter; jede hat eine
Gegenprobe, die durch diesen Filter geht.

```
Tests        697 grün · 0 rot · 0 errors · 8 übersprungen
Gate         48 Checks · 36 grün · 12 rot (unverändert)
Schichten    56 Module · 0 Verstöße
Doku         15 Zeilen berichtigt (11 geplante Checks, 4 rote Stände)
```

### 17.7 Vierte Runde: das Fenster war keine Grenze

Zwei Befunde, einer echt. Über vier Runden: **26 von 27**.

Der echte ist wieder der Marker-Scope — zum dritten Mal dieselbe Klasse, und
diesmal in Code, der zwölf Stunden alt war. In 17.5 hatte ich den Kontext
eines Namens von „ganze Zeile" auf „60 Zeichen vor dem Anker" verkürzt, damit
ein Marker vor dem Anker stehen darf („D3 gebaut, Gate-Check **rot**: …").
Der Vorlauf hielt aber an nichts an. Nachgestellt:

```
"Der vorige Check ist rot. Gate-Check `cortex_map` deckt die Karte ab."
gemeldet:  []                                  (erwartet: cortex_map)

"| A | Alter Stand: rot | Gate-Check `cortex_map` |"
gemeldet:  []
```

Der zweite Fall stand nicht im Befund und ist der schlimmere: ein „rot" in
der **Nachbarzelle** entlastete den erfundenen Namen in der Check-Zelle — also
genau in Tabellen, für die der Tabellenpfad die Regel korrekt zieht. Die
Prosa-Behandlung hat das fail-open eine Runde nach seiner Schließung wieder
eingeschleust.

Der Vorlauf endet jetzt am nächsten Satz- oder Zellende; die Zeichenzahl ist
nur noch die äußere Schranke. Die Lehre steht im Docstring der Funktion:

> **Ein Kontextfenster ist keine Regel, solange es nicht an einer Grenze
> endet.** Zeichenzahlen sind Schranken, Satz- und Zellgrenzen sind die
> Semantik. Eine kleinere Zahl ist keine Korrektur — sie verschiebt die Lücke
> nur dorthin, wo man nicht hinsieht.

Eine Vorhersage im Plan traf dabei **nicht** zu: ich hatte erwartet, dass die
Satzgrenze die beiden `dream_reflex`-Treffer im Dream-Design fallen lässt und
dort ein „(rot)" nachgetragen werden muss. Nachgemessen tragen beide ihren
Marker im *eigenen* Satz (`- **D3 gebaut, Gate-Check rot:** …`), also war
keine Doku-Änderung nötig. Die Prüfung hat die Annahme korrigiert, nicht
umgekehrt.

**Der eine Befund, dem ich nicht folge — der erste über vier Runden.**
Gemeldet war, dass vier Zitate mit `„` öffnen und mit ASCII `"` schließen, mit
der Anweisung, diese vier auf `“` zu ändern. Die Beobachtung stimmt, das
Mittel nicht. Gezählt über dieses Dokument:

```
U+201E  „  :  77        U+201C  “  :  0        ASCII  "  : 163
```

`„…"` ist die durchgehende Konvention über alle 17 Abschnitte; `“` kommt kein
einziges Mal vor. Vier Stellen zu ändern erzeugt die einzigen vier `“` in 77
Zitaten. Alle 77 umzustellen ist nicht mechanisierbar — von den 163
ASCII-Anführungszeichen schließen nur 77 ein `„`, der Rest steht in
Codeblöcken und zitierten Programmausgaben, die ein `sed` zerschießen würde.

```
Tests        698 grün · 0 rot · 0 errors · 8 übersprungen
Gate         48 Checks · 36 grün · 12 rot (unverändert)
Schichten    56 Module · 0 Verstöße
```

### 17.8 Fünfte Runde: eine Kennzahl, die sich selbst nicht messen konnte

Zwei Befunde, beide echt. Über fünf Runden: **28 von 29**.

**Der gemeldete Major** war die zweite Hälfte der Satzgrenze aus 17.7: der
Vorlauf hatte `[.!?]`, das Fenster **hinter** dem Anker nur `.`. Ein „rot"
hinter einem Ausrufe- oder Fragezeichen reichte also weiter zurück, als es
durfte. Beide Richtungen benutzen jetzt dasselbe Muster, denn sie
beantworten dieselbe Frage — wo hört der Text auf, der diese Behauptung
qualifiziert? Drei Satzzeichen × zwei Richtungen stehen als Gegenprobe.

**Der Minor führte auf etwas Größeres.** Gemeldet war, `ARCHITECTURE-MASTER`
nenne „10 von 38", das Gate aber 37. Nachgemessen stimmte **keine der beiden
Zahlen**:

```
Evidenzdateien, die das Gate liest:  35
davon mit subject-Bindung:            3
davon ohne:                          32
evidence_without_a_subject meldete:   0
```

Das Feld verlangte einen `producer`-Stempel — und weil jede ungebundene Datei
hier grandfathered, also ungestempelt ist, meldete es **null**, während die
Lücke der Normalfall war. Der Meta-Test dazu bestand, weil er nur die Dateien
prüfte, die das Feld *nannte*: bei einer leeren Liste ist beide Richtungen zu
prüfen trivial erfüllt.

> **Ein Feld, das eine Lücke sichtbar halten soll, darf nicht null melden
> können, während die Lücke der Normalfall ist.** Und ein Test, der nur das
> Gemeldete prüft, prüft die Meldung nicht.

Der Report nennt jetzt beide Seiten (`evidence_with_a_subject`,
`evidence_without_a_subject`, dazu `stamped_without_a_subject` für die
Sperrklinke), fehlende Dateien zählen nicht als „ungebunden" — sie sind
abwesend und stehen unter `missing_evidence` —, und ein Test verlangt, dass
die Listen alles Gelesene **partitionieren**. Die Zahl in beiden Dokumenten
ist berichtigt: **3 von 35**, nicht 10 von 38.

Das ist die unangenehmste Berichtigung dieses PR: die Kennzahl, mit der er
seinen eigenen Fortschritt bei der Evidenzbindung ausweist, war um mehr als
das Dreifache überzeichnet, und zwar seit dem Pod-Audit. Die 14 Benchmarks,
die `subject=` übergeben, sind gebaut — aufgezeichnet sind erst drei, der
Rest braucht den Server.

```
Tests        702 grün · 0 rot · 0 errors · 8 übersprungen
Gate         48 Checks · 36 grün · 12 rot (unverändert)
Schichten    56 Module · 0 Verstöße
```

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
.venv/bin/python -m pytest -q                      # 702 passed, 0 failed, 8 skipped
python3 research/verify_architecture_gate.py       # 12 rote Checks (siehe 14.8, 17.1)
python3 -c "import ast,pathlib; t=ast.parse(pathlib.Path('research/verify_architecture_gate.py').read_text()); \
  print(sum(len(n.value.keys) for n in ast.walk(t) if isinstance(n,ast.Assign) \
  and any(getattr(x,'id','')=='checks' for x in n.targets)))"   # 48
```

Die Prüfskripte zu Abschnitt 4.1 (`dream_algebra.py`) und 5.2
(`storage_probe.py`, `docs_probe.py`) sind im Bericht vollständig in ihren
Ergebnissen wiedergegeben und aus den zitierten Codestellen reproduzierbar.
