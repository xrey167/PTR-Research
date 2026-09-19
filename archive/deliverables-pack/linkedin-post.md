# LinkedIn-Beitrag

## Was wäre, wenn internes Wissen eine eigene Identität hätte?

Eine Lieferzeit ist für ein Sprachmodell heute meist nur ein Textfragment im Kontext. Bei jeder Anfrage wird gesucht, kopiert und erneut verarbeitet.

Wir testen gerade einen anderen Weg:

**Wissen wird zu einem versionierten Neural Pod.**

Ein Pod kennt seine Herkunft, seine semantische Identität und seine aktuelle Generation. Aliase wie:

> „Wie lange braucht Müller für X12?“  
> „Aktuelle Lieferzeit von Müller“  
> „X12 procurement delay“

zeigen auf denselben KnowledgeKey. Wird der Wert geändert, wird die Generation ersetzt. Wird sie widerrufen, dürfen abgeleitete Vektoren, Adapter und Caches nicht weiter antworten.

Der Ablauf sieht so aus:

```text
Frage
  ↓
Dragonfly: Pod-Typ und Adresse
  ↓
Embedding/BM25: Kandidaten
  ↓
generation-bound LoRA-Pod
  ↓
Qwen3B + Lifecycle-Barrier
```

Wir haben das mit einem echten Qwen3B-Modell auf einer RTX 3090 gemessen. Für dieselbe Frage und denselben Reader:

- kompakter Pod-Kontext: 61 Eingabetoken, 190,9 ms pro Batch mit vier Antworten
- längerer RAG-Kontext: 303 Eingabetoken, 412,5 ms pro Batch mit vier Antworten
- beide Pfade liefern im kontrollierten Fixture exakt „18 days“

Das entspricht **2,16× niedrigerer Latenz** und 80 % weniger Eingabetoken.

Weitere Messwerte:

- 23,9 generierte Token/s auf der RTX 3090
- 9,48 ms Search-p50 bei 2.000 Pods
- 434× schnelleres inkrementelles Operator-Update gegenüber Vollkompilierung
- 20/20 Projekt-Gate-Checks bestanden

RAG bleibt dabei unsere Evidenzschicht. Der Pod ist die kompakte, adressierbare und versionierte Wissensschicht darüber. Genau diese Trennung macht Updates, Provenienz und selektive Revocation ausführbar.

Die Ergebnisse sind bewusst eingegrenzt: Es handelt sich um einen lokalen Forschungsprototypen und kontrollierte Datensätze. 100B-ANN-Skalierung, breite natürliche Multi-Hop-Fragen und ein vollständiger Vergleich gegen optimiertes RAG stehen als nächste Prüfungen an.

Die technische Dokumentation und die reproduzierbaren Reports liegen im Projekt. Mich interessiert besonders: **Würdet ihr wiederholt genutztes Unternehmenswissen lieber als Retrieval-Kontext oder als versionierten Neural Pod behandeln?**

#AI #MachineLearning #KnowledgeSystems #RAG #LoRA #Provenance #MLOps #SemanticSearch
