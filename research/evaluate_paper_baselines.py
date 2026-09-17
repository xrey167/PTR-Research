"""Run reproducible pilot retrieval baselines on the frozen benchmark."""
from __future__ import annotations

import json
from pathlib import Path
import random
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from research.generate_paper_benchmark import build
from research.demo_pod_gallery import classify


def norm(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def bootstrap(values, seed=7, rounds=1000):
    rng = random.Random(seed)
    means = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(rounds)]
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def run(rows=None):
    rows = rows or build(700)
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    docs = [f"{r['question']} {r['knowledge_key']} {r['answer']} {r['pod_type']}" for r in train]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1)
    matrix = vectorizer.fit_transform(docs)
    outcomes = {"random": [], "tfidf": [], "typed_tfidf": []}
    for row in test:
        query = vectorizer.transform([row["question"]])
        scores = cosine_similarity(query, matrix)[0]
        order = np.argsort(-scores)
        typed = classify(row["question"])
        typed_order = [i for i in order if train[i]["pod_type"] == typed]
        for name, candidates in (("tfidf", order), ("typed_tfidf", typed_order)):
            outcomes[name].append(float(any(train[i]["answer"] == row["answer"] for i in candidates[:3])))
        outcomes["random"].append(float(random.Random(row["id"]).choice(train)["answer"] == row["answer"]))
    report = {"rows": len(rows), "train": len(train), "test": len(test), "k": 3,
              "scope": "synthetic provenance-labelled pilot; not external-world QA evidence", "baselines": {}}
    for name, values in outcomes.items():
        report["baselines"][name] = {"recall_at_3": float(sum(values) / len(values)),
                                      "bootstrap_95_ci": bootstrap(values)}
    return report


if __name__ == "__main__":
    report = run()
    Path("runs/paper-baseline-pilot-001.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
