"""Atomic, content-addressed namespace snapshots with a compactable WAL.

The filesystem layout is intentionally object-storage friendly: immutable
compressed snapshots are addressed by namespace/revision and a small manifest
points at the latest revision.  The API can later be backed by S3-compatible
put/get without changing callers.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import gzip, json, os, re, tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

_SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class SnapshotRef:
    namespace: str
    revision: int
    digest: str
    path: str


class SnapshotStore:
    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _check(namespace: str) -> None:
        if not _SAFE.fullmatch(namespace):
            raise ValueError("invalid namespace")

    def _dir(self, namespace: str) -> Path:
        self._check(namespace)
        p = self.root / namespace
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def _encode(rows: Iterable[Mapping[str, Any]]) -> bytes:
        body = json.dumps(list(rows), sort_keys=True, separators=(",", ":"), default=str).encode()
        return gzip.compress(body, mtime=0)

    def write(self, namespace: str, revision: int, rows: Iterable[Mapping[str, Any]]) -> SnapshotRef:
        if revision < 0:
            raise ValueError("revision must be non-negative")
        data = self._encode(rows)
        digest = sha256(data).hexdigest()
        directory = self._dir(namespace)
        final = directory / f"snapshot-{revision:020d}-{digest[:16]}.json.gz"
        if not final.exists():
            fd, tmp = tempfile.mkstemp(prefix=".snapshot-", dir=directory)
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(data); f.flush(); os.fsync(f.fileno())
                os.replace(tmp, final)
            finally:
                if os.path.exists(tmp): os.unlink(tmp)
        manifest = directory / "LATEST"
        mtmp = manifest.with_suffix(".tmp")
        mtmp.write_text(json.dumps({"revision": revision, "digest": digest, "file": final.name}), encoding="utf-8")
        os.replace(mtmp, manifest)
        return SnapshotRef(namespace, revision, digest, str(final))

    def read(self, namespace: str, revision: int | None = None) -> tuple[SnapshotRef, list[dict[str, Any]]]:
        directory = self._dir(namespace)
        if revision is None:
            latest = json.loads((directory / "LATEST").read_text(encoding="utf-8"))
            revision, expected, name = int(latest["revision"]), latest["digest"], latest["file"]
        else:
            matches = sorted(directory.glob(f"snapshot-{int(revision):020d}-*.json.gz"))
            if not matches: raise FileNotFoundError(f"snapshot revision {revision} not found")
            name, expected = matches[-1].name, None
        path = directory / name
        data = path.read_bytes()
        digest = sha256(data).hexdigest()
        if expected and digest != expected: raise ValueError("snapshot digest mismatch")
        rows = json.loads(gzip.decompress(data).decode())
        return SnapshotRef(namespace, int(revision), digest, str(path)), rows

    def append_wal(self, namespace: str, event: Mapping[str, Any]) -> None:
        path = self._dir(namespace) / "wal.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(event), sort_keys=True, separators=(",", ":"), default=str) + "\n")
            f.flush(); os.fsync(f.fileno())

    def compact(self, namespace: str, keep: int = 2) -> int:
        if keep < 1: raise ValueError("keep must be positive")
        files = sorted(self._dir(namespace).glob("snapshot-*.json.gz"))
        removed = 0
        for path in files[:-keep]: path.unlink(); removed += 1
        return removed
