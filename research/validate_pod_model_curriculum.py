"""Validate the modular Pod/model curriculum before any training run."""
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8-sig"))
required = set(manifest["required_fields"])
prov_required = set(manifest["required_provenance"])
seen = set()
by_split = {}
tasks = set()
tracks = set()
for split, expected in manifest["splits"].items():
    rows = json.loads((root / "inputs" / f"{split}.json").read_text(encoding="utf-8"))
    assert len(rows) == expected, (split, len(rows), expected)
    by_split[split] = {r["id"] for r in rows}
    assert len(by_split[split]) == len(rows), f"duplicate ids in {split}"
    for row in rows:
        assert required <= row.keys(), row["id"]
        assert prov_required <= row["provenance"].keys(), row["id"]
        json.loads(row["target"])
        assert row["assessment"]["capability"] == row["task"]
        tasks.add(row["task"]); tracks.add(row["model_track"])
    seen |= by_split[split]
assert by_split["train"].isdisjoint(by_split["dev"])
assert by_split["train"].isdisjoint(by_split["test"])
assert by_split["dev"].isdisjoint(by_split["test"])
assert len(tasks) == manifest["capability_count"]
print(f"PASS rows={len(seen)} capabilities={len(tasks)} tracks={len(tracks)} split_disjoint=true")
