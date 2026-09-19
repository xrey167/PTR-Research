"""Generation-6 reader inputs: train the reader LoRA on a heterogeneous base.

Copies the Gen-5 inputs and repoints them at NeoHorse-1-4B (a different base
model family), building a fresh verified model manifest. The frozen dev/test
cases stay byte-identical, so the resulting adapter is a true heterogeneous
fallback pod: it answers the same frozen questions from a different model's
weights, which is exactly what the Gen-5 ensemble fallback lacks.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

NEOHORSE = Path('/srv/ai/models/NeoHorse-1-4B')
MANIFEST_FILES = ['config.json', 'model-00001-of-00002.safetensors',
                  'model-00002-of-00002.safetensors', 'model.safetensors.index.json',
                  'tokenizer.json', 'chat_template.jinja']


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    if out.exists():
        raise SystemExit(f'output directory already exists: {out}')
    shutil.copytree(src, out)

    for name in ('dev', 'test', 'train'):
        if (out / 'inputs' / f'{name}.json').read_bytes() != (src / 'inputs' / f'{name}.json').read_bytes():
            raise SystemExit(f'{name} split was modified — refusing to continue')

    files = {}
    for name in MANIFEST_FILES:
        target = NEOHORSE / name
        if not target.is_file():
            raise SystemExit(f'missing NeoHorse file: {target}')
        data = target.read_bytes()
        files[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    manifest = {'status': 'verified', 'repo': 'TokenRhythm/NeoHorse-1-4B',
                'revision': 'local-copy-verified', 'files': files}
    (out / 'inputs' / 'model-manifest.json').write_text(json.dumps(manifest, indent=2),
                                                        encoding='utf-8')

    protocol = json.loads((out / 'protocol.json').read_text(encoding='utf-8'))
    protocol['model_path'] = str(NEOHORSE)
    protocol['status'] = 'prepared_not_trained_generation6_full_load'
    protocol['purpose'] = ('Heterogeneous fallback pod: same frozen Gen-5 cases, '
                           'NeoHorse-1-4B base for ensemble diversity')
    protocol['input_hashes']['model-manifest.json'] = hashlib.sha256(
        (out / 'inputs' / 'model-manifest.json').read_bytes()).hexdigest()
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print(json.dumps({'model_path': protocol['model_path'],
                      'train_rows': protocol['rows']['train']}))


if __name__ == '__main__':
    main()
