"""Verify that the public pack contains the assembled evidence and is synced."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "deliverables" / "neural-pods-public-pack.zip"
DESKTOP = Path(r"C:/Users/ReyDa/Desktop/Neural-Pods-Public-Pack/neural-pods-public-pack.zip")
REQUIRED = [
    "project-gate-001.json", "generalization-gate-001.json",
    "combined-evaluation-001.json", "neohorse-real-checkpoint-probe-001.json",
    "local-stack-benchmark-002.json", "xrserver-reference-serving-probe-001.json",
    "PROJECT-SYNTHESIS-20260916.md", "COMPLETION-AUDIT-20260916.md",
    "routing_harness.py", "test_routing_harness.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def zip_payload_hash(path: Path) -> str:
    """Hash sorted archive contents, excluding this self-referential report."""
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if name == "public-pack-verification-001.json":
                continue
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(archive.read(name))
            digest.update(b"\0")
    return digest.hexdigest()


def main() -> dict[str, object]:
    with zipfile.ZipFile(PACK) as archive:
        names = set(archive.namelist())
        required = {name: name in names for name in REQUIRED}
        report: dict[str, object] = {
            "schema": "public-pack-verification:v1",
            "entries": len(names),
            "zip_test": archive.testzip(),
            "required": required,
            "all_required": all(required.values()),
            "workspace_inventory_hash": sha256(ROOT / "research" / "project_inventory.json"),
            "pack_inventory_hash": hashlib.sha256(
                archive.read("project-inventory-20260916.json")
            ).hexdigest(),
            "pack_payload_hash": zip_payload_hash(PACK),
            "desktop_payload_hash": zip_payload_hash(DESKTOP),
        }
    report["inventory_equal"] = (
        report["workspace_inventory_hash"] == report["pack_inventory_hash"]
    )
    report["desktop_equal"] = report["pack_payload_hash"] == report["desktop_payload_hash"]
    report["passed"] = bool(
        report["zip_test"] is None
        and report["all_required"]
        and report["inventory_equal"]
        and report["desktop_equal"]
    )
    output = ROOT / "runs" / "public-pack-verification-001" / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    raise SystemExit(0 if main()["passed"] else 1)
