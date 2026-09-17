# Pilot benchmark failure analysis

The first baseline run (`runs/paper-baseline-pilot-001.json`) is intentionally
not a paper result. Character TF-IDF reaches Recall@3 = `1.00` on the test
split, with a 95% bootstrap interval of `[1.00, 1.00]`. The reason is leakage:
the generated questions reuse the same templates, entities and answer-bearing
surface forms across train and test.

This is a useful quality gate. We must not report the pilot as evidence of
Neural Pods performance.

## Required benchmark correction

- split by provenance roots and entities, not by row number;
- hold out supplier aliases and question templates entirely;
- paraphrase questions with an independent generator;
- include adversarial distractors sharing the same vocabulary;
- require multi-hop joins across separate source records;
- keep a locked test generator and publish only hashes for private data.

The corrected benchmark is the next implementation step. Any model that still
saturates the lexical baseline after these controls has not demonstrated a
neural-memory advantage.
