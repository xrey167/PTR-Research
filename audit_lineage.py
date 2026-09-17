"""Audit saved four-key identities and revocation on independent DB copies.

Uses real trained adapter files and previously generated text. No model output is
fabricated here; the cloned lifecycle scenarios do not run additional inference.
"""
import argparse
import json
from pathlib import Path
import sqlite3
from neural_pods.registry import Registry, InvalidState, digest, verify_files


def independent_closure(db, root):
    children = {}
    for child, parent in db.execute("SELECT child,parent FROM edges"):
        children.setdefault(parent, set()).add(child)
    seen, pending = set(), [root]
    while pending:
        node = pending.pop()
        if node in seen: continue
        seen.add(node)
        pending.extend(children.get(node, ()))
    return seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    if report["status"] != "completed": raise RuntimeError("Use completed training results")
    source = sqlite3.connect((run / "registry.sqlite3").as_uri() + "?mode=ro", uri=True)
    audit = {"status": "running", "nodes_verified": 0, "adapter_artifacts_verified": 0, "scenarios": []}
    for node_id, kind, payload_json in source.execute("SELECT id,kind,payload FROM nodes"):
        payload = json.loads(payload_json)
        parents = sorted(r[0] for r in source.execute("SELECT parent FROM edges WHERE child=?", (node_id,)))
        expected = kind + ":" + digest({"kind": kind, "payload": payload, "parents": parents})
        assert node_id == expected, f"Immutable node content changed: {node_id}"
        if "parent_artifact_keys" in payload:
            assert payload["parent_artifact_keys"] == parents
            assert payload["derivation_hash"] == digest(payload["payload"])
            ancestors, pending = set(), list(parents)
            while pending:
                ancestor = pending.pop()
                if ancestor in ancestors: continue
                ancestors.add(ancestor)
                pending.extend(r[0] for r in source.execute("SELECT parent FROM edges WHERE child=?", (ancestor,)))
            roots, generations = [], {}
            for ancestor in ancestors:
                a_kind, a_json = source.execute("SELECT kind,payload FROM nodes WHERE id=?", (ancestor,)).fetchone()
                a = json.loads(a_json)
                if a_kind == "origin": roots.append(ancestor)
                if a_kind == "knowledge":
                    generations[ancestor] = {"knowledge_key": a["knowledge_key"], "generation": a["generation"]}
            assert sorted(roots) == payload["origin_keys"]
            assert generations == payload["generations"]
        if kind == "lora":
            p = payload["payload"]
            path = (run / "adapters" / p["adapter"]).resolve()
            assert path.parent == (run / "adapters").resolve()
            verify_files(path, p["files"])
            assert p["base_sha256"] == report["base_weights_sha256"]
            audit["adapter_artifacts_verified"] += 1
        audit["nodes_verified"] += 1

    # The same semantic identity has distinct versions and distinct derived artifacts.
    selections = [next(a for a in report["answers"] if a["phase"] == phase)["selection"] for phase in ["g1", "g2"]]
    selections.append(report["restored"]["selection"])
    assert len({s["knowledge_key"] for s in selections}) == 1
    assert len({s["generation_key"] for s in selections}) == 3
    assert len({s["artifact_key"] for s in selections}) == 3
    audit["stable_knowledge_distinct_generations_and_artifacts"] = True

    restored = report["restored"]["selection"]
    for scenario, target in [("generation_revocation", restored["generation_key"]),
                             ("source_revocation", report["canonical_knowledge_object"]["origin_keys"][0])]:
        clone = Registry(":memory:")
        source.backup(clone.db)
        cache = clone.artifact("cache", {"recorded_output": report["restored"]["text"]},
                               [restored["artifact_key"]], principal="buyer")
        snapshot = clone.snapshot([cache], principal="buyer")
        answer = clone.commit(snapshot, report["restored"]["text"])["answer_id"]
        unrelated = [r[0] for r in clone.db.execute("SELECT node_id FROM heads WHERE knowledge_key NOT LIKE 'routing:%' AND knowledge_key!=?", (restored["knowledge_key"],))]
        expected = independent_closure(clone.db, target)
        actual = set(clone.revoke(target))
        assert actual == expected
        assert {restored["artifact_key"], cache, answer} <= actual
        try:
            clone.commit(snapshot, report["restored"]["text"])
        except InvalidState:
            rejected = True
        else:
            rejected = False
        assert rejected
        for node in unrelated: clone.snapshot([node], principal="buyer")
        audit["scenarios"].append({"scenario": scenario, "closure_size": len(actual),
                                  "exact_closure": True, "stale_commit_rejected": rejected,
                                  "unrelated_generations_preserved": len(unrelated)})
        clone.close()
    source.close()
    audit["status"] = "passed"
    (run / "lineage-audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__": main()
