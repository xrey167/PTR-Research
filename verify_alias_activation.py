"""Check persisted neural alias recognition and a fresh-process Qwen answer."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from sentence_transformers import SentenceTransformer
from neural_pods.dragonfly import DragonflyRouter
from neural_pods.registry import Registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / "report.json").read_text(encoding="utf-8"))
    info = manifest["models"]["encoder"]
    encoder = SentenceTransformer(info["path"], device="cpu", local_files_only=True)
    registry = Registry(run / "registry.sqlite3")
    router = DragonflyRouter(registry, encoder, run, encoder_id=info["repo"] + "@" + info["revision"])
    checks = []
    try:
        router.load_weights()
        for alias in ["Müller", "Mueller", "Muller", "Müller GmbH", "Mueller GmbH", "Muller GmbH", "Müller Werke"]:
            expected = "supplier:muller-werke" if alias == "Müller Werke" else "supplier:muller"
            result = router.recognize_alias(alias, principal="buyer")
            assert result["subject"] == expected, (alias, result)
            checks.append({"alias": alias, "subject": result["subject"], "score": result["score"]})
    finally:
        router.close()
        registry.close()
    process = subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).with_name("ask_semantic.py")), str(run),
                              "X12 procurement delay from Mueller GmbH?"], capture_output=True, text=True, encoding="utf-8")
    if process.returncode: raise RuntimeError(process.stderr)
    answer = json.loads(process.stdout)
    assert answer["status"] == "committed" and answer["text"] == "24 days", answer
    assert answer["routing_mode"] == "dragonfly-pod-address:v2"
    assert "Mueller" in answer["trained_aliases"]
    assert answer["representation_key"] in answer["receipt"]["dependencies"]
    report = {"status": "passed", "aliases": checks, "cli_answer": answer}
    (run / "activation-check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "aliases_correct": len(checks), "cli_output": answer["text"]}))


if __name__ == "__main__": main()
