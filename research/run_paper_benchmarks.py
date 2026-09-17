"""Paper-compatible benchmark plan and environment audit.

The full OSWorld/ALE runs are intentionally not silently substituted by the
small local fixture.  This command emits a manifest that can be handed to the
official runners and records exactly which prerequisites are present.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    release: str
    tasks: int
    cohort: str
    runner: str
    prerequisites: tuple[str, ...]


SPECS = (
    BenchmarkSpec("osworld_v2", "v2026.08.08", 108, "official 108-task batch", "run_osworld.py",
                  ("docker", "qemu-system-x86_64")),
    BenchmarkSpec("agents_last_exam", "d10fb61a14f9719774c3520c5763068b28ef554d", 67,
                  "Near-term: 64 CPU + 3 completed GPU baselines", "run_ale.py", ("qemu-system-x86_64",)),
)


def command_version(command: str) -> str | None:
    path = shutil.which(command)
    if not path:
        return None
    try:
        proc = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10)
        return (proc.stdout or proc.stderr).splitlines()[0][:200]
    except (OSError, subprocess.SubprocessError):
        return "present (version probe failed)"


def storage_audit() -> dict:
    candidates = [Path("/srv/ai/workspaces"), Path("/tmp")]
    result = []
    for path in candidates:
        if path.exists():
            usage = shutil.disk_usage(path)
            result.append({"path": str(path), "free_gib": round(usage.free / 1024**3, 2),
                           "total_gib": round(usage.total / 1024**3, 2)})
    snap_qemu = Path("/snap/lxd/current/bin/qemu-system-x86_64")
    return {"mounts": result, "snap_qemu": str(snap_qemu) if snap_qemu.exists() else None,
            "kvm": Path("/dev/kvm").exists()}


def build_manifest(arm: str, output: Path) -> dict:
    if arm not in {"baseline", "rsi", "both"}:
        raise ValueError("arm must be baseline, rsi or both")
    rows = []
    for spec in SPECS:
        checks = {p: command_version(p) for p in spec.prerequisites}
        rows.append({"spec": asdict(spec), "prerequisites": checks,
                     "ready": all(v is not None for v in checks.values())})
    manifest = {
        "schema": "paper-benchmark-manifest:v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": {"system": platform.system(), "release": platform.release(), "python": sys.version.split()[0]},
        "arms": ["baseline", "rsi"] if arm == "both" else [arm],
        "evaluation_protocol": {
            "verifier_outside_learning": True,
            "memory_frozen_for_test": True,
            "matched_task_deltas_only": True,
            "metrics": ["partial_score", "success", "wall_time_s", "tool_calls", "input_tokens", "memory_pods"],
        },
        "storage_audit": storage_audit(),
        "benchmarks": rows,
        "status": "prepared_not_run",
        "reason": "Official VM assets/runners and model credentials are required; no score is fabricated by this dry-run.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["baseline", "rsi", "both"], default="both")
    parser.add_argument("--output", type=Path, default=Path("runs/paper-benchmark-manifest-001.json"))
    parser.add_argument("--dry-run", action="store_true", help="audit prerequisites and write a manifest")
    args = parser.parse_args()
    if not args.dry_run:
        parser.error("full execution must be invoked through the pinned OSWorld/ALE runners; use --dry-run to prepare")
    manifest = build_manifest(args.arm, args.output)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
