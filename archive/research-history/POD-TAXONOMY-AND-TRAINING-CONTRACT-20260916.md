# Pod-Arten und Trainingsvertrag

Jede Trainingszeile trägt dieselben Lifecycle-Felder: `origin_keys`,
`knowledge_key`, `generation_key`, `pod_type`, `domain`, `semantic_role`,
`split`, `input` und `target`. Zusätzlich gelten je Pod-Art harte Tags:

| Pod-Art | Zweck | Pflicht-Tags | Typische Payload |
|---|---|---|---|
| `context` | Fakten, Dokumente, Dialogkontext | `pod:context`, `role:knowledge` | content, entities, time, uncertainty |
| `math` | Zahlen, Einheiten, Berechnungen | `pod:math`, `role:calculation`, `has:unit` | expression, operands, unit, constraints |
| `model` | ausführbarer Reader, LoRA, Distilled, MoE oder Quantized | `pod:model`, `role:execution`, `has:interface` | model/adapter ref, input/output schema |
| `reasoning` | mehrstufige Ableitung und Multi-Hop | `pod:reasoning`, `role:inference` | steps, relations, evidence set |
| `retrieval` | ANN, BM25, Filter, RRF und Suchstrategie | `pod:retrieval`, `role:search` | query plan, candidates, rank target |

Freie Tags, Aliases, Topics und LLM-Klassifikationen bleiben zusätzliche
weiche Signale. Sie dürfen die Pflicht-Tags nicht ersetzen. Harte Metadaten
stammen aus Registry bzw. Datenquelle; weiche Metadaten werden mit Quelle und
Confidence gespeichert.

Die neue `validate_training_record()`-Prüfung weist unvollständige Zeilen vor
dem Training zurück. `PodBuilder` unterstützt das Zusammenklicken eines Pods
aus Name, Typ, Vertrag und Metadaten. `pod_identity` bleibt dabei stabil;
`artifact_key` und Generation ändern sich bei einer Branch. Ein Modell muss
dadurch nicht neu verlinkt werden: Die Runtime löst beim Aktivieren die aktive
Revision auf und prüft weiterhin Generation, ACL und Revocation.
