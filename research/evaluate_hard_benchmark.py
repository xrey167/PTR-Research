"""Evaluate classical retrieval on the leakage-controlled pilot."""
from __future__ import annotations
import json, random, sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import re
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from research.generate_hard_benchmark import build


def run():
    data = build(); corpus, queries = data["corpus"], data["queries"]
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5))
    matrix = vec.fit_transform([d["text"] for d in corpus])
    values = {"tfidf": [], "random": []}
    values["provenance_router"] = []
    for q in queries:
        scores = cosine_similarity(vec.transform([q["question"]]), matrix)[0]
        order = np.argsort(-scores)
        values["tfidf"].append(float(any(corpus[i]["doc_id"] == q["target_doc"] for i in order[:3])))
        rng = random.Random(q["id"])
        values["random"].append(float(rng.choice(corpus)["doc_id"] == q["target_doc"]))
        # Controlled hard-metadata ablation: Semantic Compiler has already
        # resolved the held-out alias to an entity/root; Dragonfly then routes
        # only within that provenance root. This is not a learned score.
        match = re.search(r"partner-(\d+)-alias", q["question"])
        component = re.search(r"(component-[a-z0-9]+)", q["question"])
        routed = [d for d in corpus if match and component
                  and d["doc_id"].startswith(f"doc-{int(match.group(1))}-")
                  and component.group(1) in d["text"]]
        values["provenance_router"].append(float(any(d["doc_id"] == q["target_doc"] for d in routed[:3])))
    return {"queries": len(queries), "k": 3, "scope": data["scope"],
            "baselines": {k: {"recall_at_3": sum(v)/len(v)} for k,v in values.items()}}


if __name__ == "__main__":
    report = run()
    Path("runs/hard-baseline-pilot-001.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

