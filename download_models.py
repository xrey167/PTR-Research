"""Fetch public, revision-pinned weights; no remote model code is executed."""
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

root = Path(__file__).resolve().parent / "models"
root.mkdir(exist_ok=True)
resolved = {}
for repo, name in [("Qwen/Qwen2.5-0.5B-Instruct", "qwen"),
                   ("sentence-transformers/all-MiniLM-L6-v2", "encoder")]:
    revision = HfApi().model_info(repo).sha
    print(f"Downloading {repo}@{revision}", flush=True)
    path = snapshot_download(repo, revision=revision, local_dir=root / name,
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "README.md"])
    resolved[name] = {"repo": repo, "revision": revision, "path": str(Path(path).resolve())}
    (root / "manifest.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
print(json.dumps(resolved, indent=2), flush=True)
