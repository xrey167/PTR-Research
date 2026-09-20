# Neural Pods – lokales Forschungsexperiment

**Status:** siehe `ARCHITECTURE-MASTER-20260920.md` — dort steht der einzige
gepflegte Zählstand (Gate-Checks, Tests, Module). Diese Datei führt bewusst
keine eigenen Zahlen mehr; README, HANDOVER und Master-Dokument hatten drei
verschiedene, und alle drei stimmten nicht.

Forschungsplattform mit LoRA-Wissens-Pods, Provenance-Registry und
Self-Improvement-Kreislauf. Reader-Adapter: **Gen-7 (NeoHorse-1-4B)**.
Was davon wie gut belegt ist, prüft `research/STATE-DEEP-RESEARCH-20260920.md`
Punkt für Punkt nach.

## Aktuelle Architektur (Kurzfassung)

```
Hauptmodell ──Reflex-Kanal (Symlinks/Dragonfly)──▶ Pods
   │                                               │
   │   ┌───────────────────────────────────────────┤
   │   │ Serving: vLLM Multi-LoRA (2 Basen, 2 GPUs)│
   │   │ Konsens: Raft 3 Knoten (LXD, mTLS)        │
   │   │ Persistenz: Postgres-Quorum (3 Knoten)    │
   │   │ Cache: LRU → Redis-L2 (0.38 ms)           │
   │   └───────────────────────────────────────────┤
   ▼                                               ▼
Dream-Pod ◀── Historie als Replay-Simulator ── Eval-Outcomes
   │      (Dream-RSI-Muster: träumt Curriculum-Strategien,
   │       Vorhersage durch Evidenz validiert: 125 predicted,
   │       125 real)
   ▼
Improve-Kreislauf: Curriculum → preflight → train → frozen A/B
                   → Architecture-Gate (fail-closed) → Promotion
```

Kernkomponenten: `neural_pods/` (47+ Module: Registry/Provenance, Serving,
Retrieval, Raft, Ressourcen), `bindings/raft_binding/` (Rust/PyO3, TiKV
raft-rs), `research/` (86+ Design- und Messdokumente, Benchmarks, Gate).

**Design-Dokumente (Einstieg):**
- `ARCHITECTURE-MASTER-20260920.md` — **Projektstand und Zählstände (Quelle der Wahrheit)**
- `research/STATE-DEEP-RESEARCH-20260920.md` — Prüfbericht: was belegt ist und was nicht
- `HANDOVER-20260917.md` — Workflows, Serverpfade, Betriebswissen
- `research/POD-ARM-DESIGN-20260919.md` — Pod-Arm-Architektur (Phasen P1–P5)
- `research/POD-NERVENSYSTEM-DESIGN-20260920.md` — Mesh, native Protokoll-Sprache (MQTT/TCP ohne Tool-Use), Task-Graph
- `research/DREAM-POD-DESIGN-20260920.md` — Dream-Pod (Dream-RSI-Adaption)
- `research/ARCHITECTURE-VALIDATION-20260917.md` — Gate & Messwerte
- `research/MULTIHOST-CLUSTER-20260917.md` — Multi-Host-Topologie

**Gate:** `research/verify_architecture_gate.py` — fail-closed, Checkzahl im
Master-Dokument. Checks über alle Schichten (Retrieval, Cache, mTLS-Transport, Raft/Multi-Host,
Quorum, vLLM/Failover, Batcher, LoRA-A/B Gen-3…7, Ensemble, Redis, gRPC-,
Dream-Validierung). Betrieb auf xrserver; Git-Flow: lokal → GitHub →
Server-Klon (`git fetch && git reset origin/main`).

## Historischer Ausgangs-Prototyp (v0.1–0.3, unten unveraendert)

Umsetzung der Idee aus [Working Prototype Status](https://chatgpt.com/share/6aa9082a-85d4-83eb-ba37-63f8dfc1398e).
Ein echtes vortrainiertes Qwen-Modell lernt kleine, austauschbare LoRA-Wissens-Pods.
Eine SQLite-Registry kontrolliert deren Herkunft und Lebenszyklus.

## Zusammengeführter Stand des Prototyps (historisch)

Der aktuelle Stack ist mit zwei reproduzierbaren Gates abgesichert:

- System-, Lifecycle-, Retrieval- und J-Space-Gate: **20/20**
- Leakage-kontrolliertes Generalisierungs-/Qwen-Gate: **6/6**

Beide Läufe starten mit einem Befehl:

```powershell
.venv\Scripts\python.exe research\run_all_evaluations.py
```

Der kombinierte Report liegt unter `runs/combined-evaluation-001.json`; die
vollständige Desktop-Ausgabe liegt in `Desktop\Neural-Pods-Public-Pack`.

Model-Pods sind typisiert: `lora`, `distilled`, `moe`, `quantized` und `base`.
`neural_pods/model_pod.py` löst einen gültigen Execution-Manifest-Eintrag in
einen server-neutralen Aktivierungsplan auf; Teacher-Identität und MoE-Experten
werden dabei als harte Metadaten geprüft.

**Semantik-Erweiterung:** [Drei Ebenen, vier Schluessel, Aliasse und Filter](SEMANTIK.md).
Der neue Einstieg ist `run_semantic_experiment.py` mit `ask_semantic.py`.

**Dragonfly-Erweiterung:** [Gelernte Pod-Repräsentationen und Tests](DRAGONFLY.md).
`train_dragonfly.py` trainiert einen eigenen Adressvektor pro Pod;
`ask_semantic.py` erkennt solche Laeufe automatisch.
Seit 0.2.1 trainiert die Bibliothek alle freigegebenen Pod-Aliase automatisch mit;
der fertige Lauf `runs/dragonfly-alias-002` enthaelt dieses Aliaswissen bereits.

**Neuronale Symlinks:** [In Qwen-LoRA trainierte Link- und Cluster-IDs](NEURAL-SYMLINK.md).
`train_symlink_lora.py` trainiert die Verweise und prueft das Aufloesen auf eine
neue Fakten-Generation. `ask_semantic.py` unterstuetzt diesen zweistufigen Modellpfad.

## Architektur

```text
Quellinhalt -> SHA256-OriginKey -> kanonisches Wissen + Generation
                                      |                |
                                      v                v
                                 Qdrant-Vektor     LoRA-Dateien
                                      |                |
Frage -> MiniLM -> Qdrant -> gelernter Adressrouter -> Qwen + Pod
                                                        |
                                      Snapshot -> Commit-Pruefung -> Antwort
```

- Die deterministische semantische Uebersetzung erzeugt typisierte Lieferantenmetriken,
  Entitaeten, Einheiten und Tags aus strukturierten Eingaben. Kein universeller Freitext-Compiler.
- Jeder abgeleitete Knoten traegt Ursprungsschluessel, Wissensschluessel, Generationen,
  Eltern und einen Inhalts-Hash. LoRA-Artefakte enthalten zusaetzlich Dateipruefsummen,
  Basismodell-Version und Trainingsdaten-Hash.
- Quellen koennen mehrere Eltern haben. Die Registry dedupliziert Herkunftswurzeln.
  Die Zahl verschiedener Wurzeln beweist keine statistische Unabhaengigkeit der Quellen.
- Widerruf sperrt die transitive Nachkommenmenge, einschliesslich Vektor, Text, LoRA,
  Cache und gespeicherter Antwort. Der Typ `jspace` ist im Lebenszyklus unterstuetzt;
  ein neuronaler J-Space-Operator ist hier nicht implementiert.
- Neue Generationen machen alte materialisierte Zustaende ungueltig. Wiederherstellung
  erzeugt eine neue Generation; alte Identitaeten bleiben ungueltig.
- Die Commit-Pruefung validiert alle Vorfahren, aktuelle Generationen und ACLs.
  Pruefung und Antwortmaterialisierung laufen in einer SQLite-Schreibtransaktion.
  Ein Widerruf, der vor dem Commit abgeschlossen ist, verhindert diesen Commit.
- Der urspruengliche Router ist ein trainiertes kleines MLP. Zusaetzlich steht
  Dragonfly mit gelernten Pod-Vektoren bereit. Beide lernen Adressen, keine
  Antwortwerte. Die Trainingsdaten und gelernten Artefakte haben eigene Herkunft.

## Lokaler Start (PowerShell)

```powershell
cd C:\Users\ReyDa\neural-pods
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -u run_experiment.py --steps 32 --threads 4 --output runs/mein-lauf
.\.venv\Scripts\python.exe ask.py runs/mein-lauf "How many days does Norvex Supply need for Z47?"
```

Das Ausgabeverzeichnis muss neu sein. Modelle liegen in `models/`; `manifest.json`
enthaelt die exakten Hugging-Face-Revisionen. Modellladung und Experiment laufen
mit `local_files_only=True`, ohne Remote-Code und ohne externe Inferenz-API.
`download_models.py` benoetigt nur fuer den erstmaligen Download eine Internetverbindung.
`requirements-lock.txt` haelt die installierten Paketversionen fest (Windows/Python 3.13).
`ask.py` laedt einen abgeschlossenen Lauf, prueft Basismodell und Adapterdateien,
generiert eine echte Antwort und liefert den Commit-Beleg mit Herkunftsschluesseln.

## Experiment

- Echtes [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
  unveraenderte Basisgewichte, CPU/float32, Rank-8-LoRA auf q_proj und v_proj.
- [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) fuer Embeddings;
  [Qdrant Local](https://github.com/qdrant/qdrant-client) als echte persistente Vektorsuche.
- Drei erfundene Lieferantenfakten, acht Trainingsfragen je Fakt, separate Validierung
  und vier vorher unbenutzte Testformulierungen je Fakt, darunter eine deutsche.
- Assistant-only-Trainingsverlust. Keine Antwortwerte im neuronalen Inferenzprompt.
- Verglichen werden Basis, Qdrant-RAG, ein gemeinsamer dauernd aktiver LoRA-Adapter
  und einzeln geroutete Pods. Alle verwenden dieselbe deterministische Generierung.
- Exact-Match wird streng gemessen; raw outputs bleiben erhalten. Kein Ersetzen
  falscher Modellantworten durch die Sollwerte aus der Datenbank.
- Update 24 -> 18 Tage, Wiederherstellung, Widerruf, alte Cache-Zustaende und
  laufende Snapshots werden geprueft. Alte Modellgewichte werden zusaetzlich bewusst
  ohne Barriere aufgerufen, um Zugriffssperre und neuronales Vergessen zu unterscheiden.
- Mehr-Pod-Test: zwei neuronale Einzelantworten, anschliessend textuelle Komposition
  durch das Basismodell. Keine gleichzeitige Adapterfusion.

Ergebnisse: `runs/<lauf>/report.json`, Rohfragen: `dataset.json`,
Herkunftsgraph: `registry.sqlite3`, Adapter: `adapters/`, Router: `router.pt`.

Weitere Pruefungen:

```powershell
.\.venv\Scripts\python.exe -u verify_saved.py runs/mein-lauf
.\.venv\Scripts\python.exe -u compare_rag.py runs/mein-lauf
.\.venv\Scripts\python.exe summarize_run.py runs/mein-lauf
```

`verify_saved.py` prueft weitere neue Fragen nach einem Prozessneustart.
`compare_rag.py` waehlt einen RAG-Prompt auf reservierten Validierungsfragen und
prueft ihn auf nochmals neuen Fragen. Zusaetzlich misst es einen strukturierten
Lookup ohne LLM. Die unterschiedlichen Fragegruppen sind keine gepaarte Studie.
Der lesbare Ergebnisbericht ist `runs/<lauf>/ERGEBNIS.md`.

## Aussagegrenzen

### Korrekturen nach dem Review (0.1.1)

Neue semantische Identitaeten verwenden `knowledge:v2:SHA256` ueber das
kanonische JSON-Array `[subject_id, component_id oder null, predicate, role]`.
Damit sind Feldgrenzen eindeutig; Anzeigenamen und Antwortwerte bleiben ausserhalb
der Identitaet. Alte v1-Laeufe bleiben lesbar und werden nicht umgeschrieben.
Der Compiler lehnt neue Schreibvorgaenge in v1-Registern ab. Fuer v2 einen frischen
Lauf mit `run_semantic_experiment.py --output runs/<neuer-name>` erstellen;
Relationen muessen auf die dabei zurueckgegebenen neuen KnowledgeKeys zeigen.
Eine automatische Migration vorhandener DAGs und Gewichte ist nicht implementiert.

Die Namensaufloesung verwendet nur aktuelle, zeitgueltige und fuer den Principal
zugelassene Indexeintraege. Der lokale Lieferzeit-Parser hat bewusst einen begrenzten
deutschen/englischen Wortschatz: unbekannte Teile und nicht aufgeloeste Frageinhalte
werden abgewiesen. Auch legitime neue Formulierungen koennen daher eine Enthaltung
ausloesen. Generische Dauerfragen allein begruenden keine Lieferzeit-Absicht.

Regressionen und echter gespeicherter Modelltest:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --junitxml=runs/fix-tests.xml
.\.venv\Scripts\python.exe verify_routing_fixes.py
```

Der Modelltest verwendet eine temporaere Kopie von `runs/semantic-003`, prueft die
Modell-/Adapter-Hashes und veraendert dessen gespeichertes Register nicht.
Ergebnisse: `runs/fix-model-probes.json` und `runs/fix-tests.xml`.

Dies ist ein kleiner funktionaler Pilot mit synthetischen Fakten in einem echten LLM.
Er prueft weder 100–1000 Pods noch die Ueberlegenheit gegen Produktions-RAG.
Die Latenzen stammen aus einem sequenziellen CPU-Durchlauf mit warm geladenen Adaptern;
sie sind keine belastbaren Skalierungs-, GPU- oder Durchsatzbenchmarks unter Parallelitaet.
Trainingskosten werden separat dokumentiert. Das MLP und seine Schwellen sind kein
allgemein kalibrierter Router; unbekannte Entitaeten und adversarielle Fragen erfordern
einen groesseren Testkorpus.

Widerruf ist eine logische Sperre im kontrollierten, nicht streamenden Laufzeitpfad.
Er loescht keine bereits ausgelieferte Antwort und keine kopierten Gewichte und beweist
kein Unlearning im Basismodell. Direkter Gewichts-/DB-Zugriff umgeht diese Grenze.
Die Registry und das lokale Dateisystem sind vertrauenswuerdig vorausgesetzt;
SHA256-Pruefsummen sind keine digitalen Signaturen. ACL-Principals sind hier lokale
Aufrufparameter, kein authentifizierter Mehrbenutzerdienst.

Quellen-Widerruf und Commit werden linearisiert; ein nach erfolgreichem Commit
eintreffender Widerruf kann die fruehere Ausgabe nicht rueckwirkend verhindern.
Kein Streaming und keine KV-Cache-Wiederverwendung zwischen Anfragen.

Vorhandene Serverdienste und GPUs wurden fuer dieses Experiment nicht veraendert.
