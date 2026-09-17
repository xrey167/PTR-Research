# Recursive self-improvement as Pods

Die beiden Referenzen liefern zwei komplementäre Muster:

- **Dream-RSI** nutzt historische Discovery-Trees als Replay-Simulator. Eine
  Explorationsstrategie kann offline bewertet werden, bevor sie erneut online
  ausgeführt wird.
- **RSIAgent** trennt breite Exploration von gezielter Tiefenexploration und
  konsolidiert verifizierte Action-Condition-Outcome-Beziehungen als wieder
  verwendbare Erinnerung.

## Abbildung auf Neural Pods

```text
Explorer / Tools
      │ action + observation
      ▼
Discovery history
      │ verified outcome
      ▼
Experience Pod ── OriginKey / KnowledgeKey / GenerationKey
      │
      ├── local replay simulator
      ├── Dragonfly type/domain routing
      └── Model-Pod activation after lifecycle validation
```

`neural_pods.recursive_memory` implementiert den lokalen Kern:

1. `BroadThenDeepExplorer` sammelt Seeds parallel und expandiert nur unsichere
   oder fehlgeschlagene Äste bis zu einer festen Tiefe.
2. `DiscoveryReplay` bewertet Policies auf verifizierter Historie ohne neue
   externe Ausführung.
3. `consolidate()` erzeugt kompakte `ExperiencePod`s aus wiederholten,
   verifizierten Beziehungen.
4. Ein Experience-Pod wird über das normale Registry-DAG als `context`-Pod
   gebunden und unterliegt damit ACL, Generation und Revocation.

Die Demo `research/demo_recursive_pod_loop.py` erzeugt einen tiefen Login-Ast,
replayed beide verifizierten Fälle mit 2/2 korrekten Ergebnissen, aktiviert den
Experience-Pod vor dem Widerruf und blockiert ihn danach. Der Lauf schreibt
`runs/recursive-pod-loop-001.json`.

Der Search-Agent-Lauf `research/demo_recursive_search_agent.py` verbindet diese
Schicht mit dem echten lokalen iterativen Retriever: Broad erreicht 0,5 Recall,
der ausgelöste Deep-Lauf 1,0 Recall (1 beziehungsweise 3 Suchaufrufe), danach
werden die verifizierten Sucherfahrungen konsolidiert.

## Was damit erreicht ist

- Explorationsergebnisse werden zu adressierbarem, wiederverwendbarem internem
  Wissen statt zu unkontrolliertem Gesprächsverlauf.
- Replay und Online-Erkundung sind getrennt; eine schlechte Policy kann offline
  aussortiert werden.
- Nur verifizierte Ergebnisse werden konsolidiert.
- Die bestehende Model-Pod-Familie kann diese Experience-Pods über Dragonfly
  auswählen und danach generation-bound aktivieren.

## Grenze der aktuellen Implementierung

Das ist ein lokaler, deterministischer Replay- und Konsolidierungskern. Er ist
noch kein Nachweis für die externen OSWorld-/Agent's-Last-Exam-Zahlen der
Referenzen und enthält noch keinen autonomen Qwen-Agenten, der echte Tools in
offenen Umgebungen ausführt. Diese nächste Stufe braucht ein reproduzierbares
Tool-Environment, Verifier und einen festgelegten Held-out-Split.

Referenzen: [Dream-RSI](https://huggingface.co/papers/2609.14858) und
[RSIAgent](https://huggingface.co/papers/2609.15364).
