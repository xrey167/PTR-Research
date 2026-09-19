"""Publish trained reader adapters as versioned model Pods in the Registry.

Records the full provenance chain per adapter (dataset protocol hash ->
training report hash -> adapter file hashes) and writes a machine-readable
inventory for the serving/benchmark stage. Gen-5 is marked promoted; Gen-3
stays available as ensemble fallback.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.train_reader import file_sha
from neural_pods.registry import Registry


ADAPTERS = [
    {"name": "reader-gen3", "run": "runs/qwen3b-lora-generation3-20260917",
     "inputs": "runs/reader-training-inputs-generation3", "promoted": False},
    {"name": "reader-gen5", "run": "runs/qwen3b-lora-generation5-20260919",
     "inputs": "runs/reader-training-inputs-generation5", "promoted": True},
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    output = args.output or (args.project_root / 'runs' / 'adapter-pod-registry-001')
    output.mkdir(parents=True, exist_ok=False)
    registry = Registry(str(output / 'registry.sqlite3'))
    inventory = []
    for adapter in ADAPTERS:
        run_dir = args.project_root / adapter['run']
        inputs_dir = args.project_root / adapter['inputs']
        report = json.loads((run_dir / 'report.json').read_text(encoding='utf-8'))
        protocol_sha = file_sha(inputs_dir / 'protocol.json')
        if report['protocol_sha256'] != protocol_sha:
            raise ValueError(f"{adapter['name']}: training protocol mismatch")
        artifact_hashes = {name: file_sha(run_dir / 'adapter' / name)
                           for name in report['adapter_files']}
        origin = registry.origin('reader_adapters', adapter['name'], 1,
                                 {'kind': 'reader_lora_adapter',
                                  'base_model': 'qwen3b',
                                  'dataset_protocol_sha256': protocol_sha,
                                  'training_report_sha256': file_sha(run_dir / 'report.json'),
                                  'adapter_files': artifact_hashes,
                                  'reader_identity_sha256': report['reader_identity']['sha256']})
        published = registry.publish('adapter:' + adapter['name'],
                                     {'pod_type': 'lora', 'promoted': adapter['promoted']},
                                     parents=[origin])
        artifact = registry.artifact('lora',
                                     {'pod': published,
                                      'adapter_dir': str(run_dir / 'adapter'),
                                      'files': artifact_hashes},
                                     parents=[published])
        inventory.append({'name': adapter['name'], 'origin': origin,
                          'pod': published, 'artifact': artifact,
                          'promoted': adapter['promoted'],
                          'adapter_dir': str(run_dir / 'adapter')})
    (output / 'inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    print(json.dumps({'published': [i['name'] for i in inventory],
                      'output': str(output)}))


if __name__ == '__main__':
    main()
