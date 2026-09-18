"""Freeze reader-training inputs and an explicit schedule; does not train a model."""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.reader_training_data import build_data


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(Path('research/reader-lora-candidate.json').read_text(encoding='utf-8'))
    training = config['training']
    if training['micro_batch_size'] != 1 or training['world_size'] != 1:
        raise ValueError('The proposed training primitive currently supports single-process microbatch 1')
    assert training['effective_batch_size'] == training['micro_batch_size'] * training['gradient_accumulation_steps'] * training['world_size']
    data = build_data()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    inputs = output/'inputs'; inputs.mkdir()
    for split, rows in data.items():
        (inputs/f'{split}.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copyfile('research/reader-lora-candidate.json', inputs/'config.json')
    model = Path(config['model_path']).resolve()
    shutil.copyfile(model/'download-manifest.json', inputs/'model-manifest.json')
    model_manifest = json.loads((inputs/'model-manifest.json').read_text(encoding='utf-8'))
    snapshot = output/'source_snapshot'; snapshot.mkdir()
    source_files = [Path('research')/name for name in (
        'prepare_reader_training.py', 'reader_training_data.py', 'reader_prompt.py',
        'reader_training.py', 'reader_identity.py', 'prefix_capsule.py', 'progress_json.py')]
    source_files.append(Path('neural_pods/registry.py'))
    for source in source_files:
        target = snapshot/source; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    updates_per_epoch = math.ceil(len(data['train']) / training['gradient_accumulation_steps'])
    total_updates = updates_per_epoch * training['epochs']
    protocol = {
        'status': 'prepared_not_trained', 'full_research_goal_complete': False,
        'rows': {split: len(rows) for split, rows in data.items()},
        'input_hashes': {p.name: sha(p) for p in inputs.iterdir()},
        'source_hashes': {p.relative_to(snapshot).as_posix(): sha(p) for p in snapshot.rglob('*.py')},
        'model_path': str(model), 'model_repo': model_manifest['repo'], 'model_revision': model_manifest['revision'],
        'model_weight_files_reverified_in_this_preparation': False,
        'packages': {name: version(name) for name in ('torch', 'transformers', 'peft')},
        'max_sequence_tokens': 512, 'evaluation_max_new_tokens': 64,
        'optimizer_updates_per_epoch': updates_per_epoch, 'optimizer_updates': total_updates,
        'warmup_updates': math.ceil(total_updates * training['warmup_ratio']),
        'last_window_microbatches': len(data['train']) % training['gradient_accumulation_steps'] or training['gradient_accumulation_steps'],
        'selection_policy': 'Two fixed epochs for this candidate; development used for diagnosis, final test not for checkpoint selection',
        'evaluation_policy': 'Save real unconstrained outputs; exact target matching is a format metric, not a complete semantic-quality metric',
        'limitations': 'Synthetic procurement data. Training CLI, actual-device memory preflight, model training and integrated trained-reader evaluation remain to be executed.',
    }
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print(json.dumps({k: protocol[k] for k in ('status', 'rows', 'optimizer_updates', 'warmup_updates', 'last_window_microbatches')}))


if __name__ == '__main__': main()
