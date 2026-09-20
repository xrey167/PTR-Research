# Pod-Arm-Design: Das Hauptmodell und seine Pod-Gliedmaßen — 2026-09-19

## Vision

Das Hauptmodell nutzt Pods nicht wie externe Werkzeuge, sondern wie
Gliedmaßen: ein kleines LLM, ein Spezialisierter (Objekterkennung), ein
Bereich, der sich selbst weitertrainiert. Technisch heißt das drei Dinge:
**Reflex-Zugriff ohne Protokoll-Overhead**, **Sinnesorgane als Streams** und
**ein geregelter Selbstverbesserungs-Kreislauf** — alles auf dem bestehenden
Fundament (Registry-Provenance, LifecycleGate, fail-closed Gate).

## 1. Die drei Bindungstiefen

### 1.1 Reflex (gelernt, nativ)

Das Hauptmodell emittiert ein latentes Adress-Signal statt eines geparsten
Tool-Calls. Basis: die bereits trainierten „neural symlinks" (`symlink.py`,
Pod-Links im Qwen-LoRA) und die gelernten Addressvektoren (`dragonfly.py`,
`DragonflyRouter`).

```
Hauptmodell → [pod:addr]-Latent → DragonflyRouter (Vektor-Distanz)
            → Executor-Dispatch → Ergebnis-Tokens zurück in den Kontext
```

- **Kein JSON-Parsing auf dem Reflex-Pfad** — die Adresse ist Teil der
  Modellrepräsentation (im Training mitgelernt: wann wird welcher Arm gestreckt)
- Latenzziel: < 5 ms vom Token bis zum Dispatch (Router-Inferenz ist ein
  Embedding-Dot-Product)
- **Templating-Regel aus Gen-6 übernehmen:** jeder Pod bekommt sein eigenes
  Prompt-Template; der Reflex-Kanal transportiert nur die Adresse, nie
  cross-Pod-Prompts

### 1.2 Gelenk (deterministisch, bewacht)

Alles mit Folgen läuft durch die typisierte Barriere — unverändert zum
heutigen Stand: `PodTransport` (Lineage-Validierung), `LifecycleGate`,
`guarded_answer`/Verifier-Konzept. Reflex darf greifen, das Ergebnis
kommt nur durch das Gelenk.

### 1.3 Kortexkarte (das Selbstbild)

`pod_taxonomy.route_candidates` + `Registry` bilden die aktuelle
Körperkarte: welche Pods existieren, welche Generation, welche
Capabilities, welcher Zustand (aktiv/Lease). Das Hauptmodell „weiß",
was es bewegen kann — auch „mein Erkennungs-Arm ist in Version 7".

## 2. Executor-Abstraktion (nativ und schnell für jede Modellfamilie)

Das Protokoll bleibt runtime-agnostisch; die Runtime ist eine Capability
im Pod-Manifest:

```
PodHeader (Erweiterung):
  runtime:   lora_vllm | lora_peft | qlora | xgboost | onnx | gguf
  precision: bf16 | fp16 | int8 | nf4
  batch_profile: latency | throughput | none
  lease_kind: cuda_vram | host_ram | none
```

> **Nicht umgesetzt (Stand 2026-09-20).** Diese vier Felder existieren in
> keinem Header. `neural_pods/pod_contract.py: PodManifest` kennt sie nicht,
> `pod_protocol` dispatcht nicht nach `runtime`, und die unten skizzierte
> `EXECUTORS`-Registry in `model_pod` gibt es nicht — die Zuordnung von
> Laufzeit zu Executor liegt allein in `ExecutorFactory`, und die wird pro
> Aufrufer neu gefüllt. Der Block beschreibt den Zielzustand. Er wurde hier
> gelassen, weil er das Ziel gut beschreibt, und markiert, weil er sonst
> gelesen wird, als sei er erreicht.

Vier-Operationen-Schnittstelle. **Berichtigung (2026-09-20, Pod-Audit):**
„bereits andocken" stimmte für keines der beiden. Der `ResourceGovernor`
dockt seit dem Audit tatsächlich an — `GovernedExecutorPool` delegiert die
Zulassung an ihn, statt einen zweiten Byte-Zähler zu führen, und
`research/benchmark_xgboost_pod.py` misst die Ablehnung. Das `LifecycleGate`
dockt **nicht** an: kein Pfad von `pod_executor.py` führt dorthin, die
Freigabe eines Executors ist an keine Manifest-Transition gebunden. Der
Abschnitt beschreibt insoweit eine Absicht, keinen Zustand.

```python
class PodExecutor(Protocol):
    def activate(self, lease: ResourceLease) -> None: ...
    def infer(self, request: PodRequest) -> PodResponse: ...
    def release(self) -> None: ...
    def health(self) -> bool: ...
```

| Executor | Pool | Schnellster Pfad | Batching |
|---|---|---|---|
| `LoraVllmExecutor` | GPU | vLLM Multi-LoRA (gemessen: identisch zu peft-Eval) | AdaptiveBatcher |
| `TreeExecutor` (XGBoost/LightGBM) | CPU (RAM-Lease) | In-Process, kein Tensor-Transport | nein, direkter Dispatch |
| `OnnxExecutor` | CPU/GPU | ONNX-Runtime (Muster: `ranking.onnx_ranker`) | klein |

**QLoRA ist ein Trainings-Attribut, kein Serving-Format:** 4-bit nf4-Basis
+ Adapter steht in der Provenance (Registry kennt den Artifact-Typ
`quantization`); beim Serving entscheidet das Manifest (merged-fp16 oder
quantisierte Basis + Adapter, vLLM unterstützt AWQ/GPTQ-Basen mit LoRA).
Die Provenance-Kette bleibt unangetastet.

Registrierung als Factory-Registry in `model_pod.ModelPodRuntime` (ist
bereits „server-neutral activation plan"):

```python
EXECUTORS: dict[RuntimeKind, type[PodExecutor]] = {}
def register_executor(kind): ...   # Einzeiler für neue Runtimes
```

Dispatch in `pod_protocol` nach `runtime`: GPU-Familien über
`VllmReplicaRouter`, CPU-Familien direkt (Batcher wird umgangen).

## 3. Sinnesorgane: Perceptions-Pods als Duplex-Streams

Objekterkennung & Co. sind **streamende Sinnesorgane**, keine Request-
Response-Dienste:

- Der Pod **pusht** verdichtete Ereignisse („Objekt X, Position Y,
  Konfidenz Z") — die Verdichtung passiert nah am Sensor, das Hauptmodell
  konsumiert symbolische Zusammenfassungen als Tokens, nie Rohdaten
- `PodFanout` + `HypothesisBranch`/`merge_branches` aus `pod_streams`
  tragen mehrere Wahrnehmungs-Hypothesen parallel
- Backpressure nach dem crossbeam-Prinzip (bounded Queues, Deadline-Drops —
  im `AdaptiveBatcher` bereits implementiert)
- Event-Kontrakt versioniert (Protocol-Version-Feld, siehe Migration)

## 4. Der Selbstverbesserungs-Kreislauf („einen Bereich stärker trainieren")

Alle Einzelstufen existieren als erprobte Skripte (Gen-3→4→5→6); fehlend
ist nur der Orchestrator:

```
1. Erfahrung:      recursive_memory (Experience/DiscoveryReplay) +
                   Fehlerfall-Klassifikation (Gen-3-Muster: 40 Misses →
                   Familien)
2. Datensynthese:  NGU-/weak-family-Oversampling (prepare_generation*-Muster)
3. Training:       preflight → train (train_reader-Pipeline)
4. Bewertung:      frozen dev/test A/B + Verifier (guarded/raw), niemals
                   die Splits verschieben
5. Promotion:      Gate-Check fail-closed → Registry: neue Generation
                   (supersedes-Kette, Lifecycle-Gültigkeit)
6. Umschaltung:    LifecycleGate routet auf die neue Revision; alte
                   Generation bleibt rollback-fähig (SnapshotStore)
```

**Improve-Orchestrator** = Controller-Ebene (nicht im Modell): nimmt
„Familie X ist schwach" entgegen, fährt 1–6, meldet das Ergebnis ins
Event-Log. Das Hauptmodell kann ihn *anstoßen* (Deliberate-Pfad), aber
nie selbst die Promotion entscheiden — die gehört dem Gate.

## 5. Sicherheit: Sperrreflexe des Arms

1. **Promotion nie ohne Gate** — fail-closed ist der A-Bseil
2. **Rückholbarkeit** — Registry-Supersedes + SnapshotStore, jede
   Pod-Generation rollback-fähig
3. **Autonomie-Budget** — Selbst-Trainingsläufe verbrauchen eine im
   Event-Log sichtbare Quote (z. B. max. N pro Tag, max. GPU-Stunden)
4. **Endabschaltung** — Delete-Tokens (`lifecycle_transport`) bleiben
   die höchste Instanz
5. **frozen Splits sind tabu** — die Evidenzbasis darf der
   Selbstverbesserungs-Kreislauf nie anfassen

## 6. Umsetzungsphasen (jede mit eigenem fail-closed Gate-Check)

| Phase | Inhalt | Gate-Check |
|---|---|---|
| P1 | **Failover-Mechanik abgeschlossen, Adressierung offen.** `neural_pods/reflex.py` + GPU-gekoppelter Benchmark (`research/benchmark_reflex_dispatch.py`) über das heterogene Ensemble. Union 126 == Baseline, 0 Fehler, alle 132 Reflex-Misses korrekt zum Default-Pod zurückgezogen. Hit-Rate 0.0: das Basis-Modell emittiert keine Alias-Signale, die Adress-Emission braucht P5. | `reflex_failover` **grün** · `reflex_dispatch` **rot bis P5** |
| P2 | **Abgeschlossen.** Executor-Factory + `TreeExecutor` (XGBoost-Pod, echtes RAM-Lease über den `ResourceGovernor`, deterministisches Replay über zwei unabhängig aktivierte Pods). `research/benchmark_xgboost_pod.py` läuft ohne Server; `tests/test_benchmark_xgboost_pod.py` prüft die Messung selbst. | `xgboost_pod` **grün** |
| P3 | Perceptions-Stream: `neural_pods/perception.py` existiert und ist seit dem Pod-Audit belastbar (In-Flight-Zählung, überlebende Consumer-Fehler, wirksamer Ratenbegrenzer). Der ONNX-Pod fehlt, und die Messung braucht den Broker. | `perception_stream` **existiert nicht** |
| P4 | Improve-Orchestrator: Autonomie-Quote und Event-Log sind gebaut (`CycleBudget`, `dream_cycle`-Event, `research/run_dream_cycle.py`). Der Ein-Zyklus-Durchlauf von Fehleranalyse bis Promotion ist es nicht. | `improve_cycle` **existiert nicht** |
| P5 | Latentes Adress-Training: Symlink-Erweiterung des Gen-6-LoRA auf alle Pod-Adressen. Nicht begonnen. | `latent_addressing` **existiert nicht** |

**Berichtigung (2026-09-20, Pod-Audit).** Die Spalte „Gate-Check" führte für
P2 bis P5 Checks auf, die es nie gegeben hat, und für P1 einen, der das
Falsche prüfte. Das ist derselbe Fehler wie in DREAM-POD-DESIGN D3: eine
Phase gilt als belegt, weil in der Tabelle ein Check-Name steht.

- `reflex_dispatch` war grün, obwohl die aufgezeichnete Evidenz
  `reflex_hits 0 / reflex_misses 132 / failovers 132` sagt — alle 132
  Antworten kamen vom Failover. Der Check las weder die Trefferzahl noch
  einen Fehlerzähler, der je hochgezählt wird. Er ist jetzt geteilt:
  `reflex_failover` belegt, was der Lauf zeigte, `reflex_dispatch` verlangt
  mindestens einen aufgelösten Alias und bleibt bis P5 rot.
- `xgboost_pod` existiert seit heute und ist grün.
- `perception_stream`, `improve_cycle` und `latent_addressing` werden hier
  nicht erfunden: die Perceptions-Messung braucht den MQTT-Broker, P4 und P5
  sind nicht gebaut. Ein Check ohne Evidenz wäre genau die Sorte Zusicherung,
  die dieser Abschnitt korrigiert.

Akzeptanzkriterium über alle Phasen: kein bestehender Check darf einbrechen;
jede Phase liefert ihre Messdatei als Evidenz — **gestempelt**, mit
`producer` und `subject` (`research/evidence.py`), sonst lässt das Gate sie
nicht mehr durch.

## 7. Risiken & Offene Fragen

- **Template-Sensitivität** (Gen-6-Lektion): der Reflex-Kanal darf
  Prompt-Templates nicht mischen — pro Pod ein Template, vertraglich
  im Manifest
- **Trainingslauf-Stabilität**: 5 harte Systemhänge unter GPU-Last;
  kdump-Crash-Capture ist jetzt aktiv — Selbst-Trainingsläufe laufen
  erst mit auswertbarem Crash-Dump
- **Offen**: Priorisierung P2 vs. P3 (Baum-Pod zuerst oder Sinnesorgan
  zuerst?); Event-Kontrakt-Versionierung im `pod_protocol`
