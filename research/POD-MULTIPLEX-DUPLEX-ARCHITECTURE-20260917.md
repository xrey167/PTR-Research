# Multiplex- und Duplex-Schicht zwischen Pods

## Entscheidung

Ja, wir bauen beides zwischen Pods, aber mit unterschiedlichen Rollen:

- **Multiplex** bündelt mehrere geprüfte Hypothesen/Toolpfade und merged strukturierte Ergebnisse.
- **Duplex** hält einen laufenden, unterbrechbaren Stream mit geordneten Events.

Raw hidden states oder kontinuierliche Tokens werden nicht zwischen heterogenen Pods geteilt. Das wäre nur sicher, wenn Verifier, Tokenizer, Layer-Schnittstellen und Generation exakt identisch sind. Für unsere gemischten Qwen-, FunctionGemma-, OCR-, Embedding- und Runtime-Pods verwenden wir deshalb typed evidence packets.

## Multiplex-Packet

Jeder Zweig liefert:

```json
{
  "branch_id": "b1",
  "pod_id": "pod:research",
  "generation": "g8",
  "evidence": ["knowledge:K1", "artifact:A2"],
  "answer": "...",
  "confidence": 0.91,
  "latency_ms": 42,
  "verified": true
}
```

Der Merge sortiert deterministisch nach Verifikation, Confidence, Latenz und Branch-ID, dedupliziert Evidenz und begrenzt die Evidenzmenge. Dadurch können ANN/BM25-, Multi-hop-, Math- und Reader-Pods parallel arbeiten, ohne ihre internen Modellrepräsentationen zu vermischen.

Implementierung: `neural_pods/pod_streams.py`, `merge_branches`.

## Duplex-Session

`DuplexSession` ist die transportneutrale Zustandsmaschine für SSH, WebSocket und In-Process:

- monoton steigende Event-Sequenzen;
- `session_id`, `epoch`, `turn_id`;
- Interrupt/Barge-in als neue Epoch;
- Resume ab letztem Event-Cursor;
- explizites Session-Close;
- Audio/Text/Video bleiben Plugin-Nutzlasten.

Implementierung: `neural_pods/pod_streams.py`.

## Lifecycle-Sicherheit

Vor dem Merge oder Forwarding muss jeder Zweig bereits durch `PodLink.validate` geprüft sein: Pod-ID, Generation, Artifact, Capability, ACL, Hop-Budget, Attestation und Zyklusfreiheit. Der Merge darf keine unverified Branches aufnehmen.

## Wann echte Multiplex-Tokens sinnvoll sind

Nur innerhalb eines einzelnen kompatiblen Reasoning-Modells, wenn wir die Multiplex-Thinking-Architektur vollständig integrieren. Dann braucht der Pod einen eigenen Token-/Hidden-State-Vertrag und einen diskreten-CoT-Baseline-Test. Für Pod-zu-Pod-Kommunikation bleiben strukturierte Branches robuster, auditierbarer und rückrufbar.
