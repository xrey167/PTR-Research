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

Vier-Operationen-Schnittstelle, an die `ResourceGovernor` (Leases) und
`LifecycleGate` (Freigabe) bereits andocken:

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
| P1 | Reflex-Dispatch — **Kern implementiert:** `neural_pods/reflex.py` (ReflexChannel: TemporalPortPlane-Auflösung → Dispatch, guarded Failover auf den default Pod, Latenz-/Failover-Metriken; 5 Tests). Offen: GPU-gekoppelter Benchmark gegen das vLLM-Ensemble | `reflex_dispatch` |
| P2 | Executor-Factory + erster `TreeExecutor` (XGBoost-Pod, RAM-Lease, deterministisches Replay) | `xgboost_pod` |
| P3 | Perceptions-Stream: ONNX-Objekterkennungs-Pod pusht Ereignisse über Duplex-Session; Backpressure gemessen | `perception_stream` |
| P4 | Improve-Orchestrator: Autonomie-Quote, Event-Log, Ein-Zyklus-Durchlauf von Fehleranalyse bis Promotion | `improve_cycle` |
| P5 | Latentes Adress-Training: Symlink-Erweiterung des Gen-6-LoRA auf alle Pod-Adressen | `latent_addressing` |

Akzeptanzkriterium über alle Phasen: keine bestehenden Checks darf
einbrechen; jede Phase liefert ihre Messdatei als Evidenz.

## 7. Risiken & Offene Fragen

- **Template-Sensitivität** (Gen-6-Lektion): der Reflex-Kanal darf
  Prompt-Templates nicht mischen — pro Pod ein Template, vertraglich
  im Manifest
- **Trainingslauf-Stabilität**: 5 harte Systemhänge unter GPU-Last;
  kdump-Crash-Capture ist jetzt aktiv — Selbst-Trainingsläufe laufen
  erst mit auswertbarem Crash-Dump
- **Offen**: Priorisierung P2 vs. P3 (Baum-Pod zuerst oder Sinnesorgan
  zuerst?); Event-Kontrakt-Versionierung im `pod_protocol`
