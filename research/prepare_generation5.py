"""Generation-5 reader inputs: concept oversampling PLUS a typed-precision anchor.

The Gen-4 experiment showed the trade-off: 3x concept oversampling lifted
concept fidelity to 43/44 but cost 4 typed raw matches (74 -> 70 on test).
The trainer shuffles per epoch, so a family's gradient share is its row
share. Gen-5 keeps the Gen-4 concept oversampling and adds one extra copy
of the lookup family, restoring the typed share while keeping the concept
signal. dev/test stay byte-identical (frozen, comparable to Gen-3/Gen-4).
"""
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

from prepare_generation4 import WEAK_FAMILY_MARKERS, is_weak_family


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    if out.exists():
        raise SystemExit(f'output directory already exists: {out}')
    shutil.copytree(src, out)

    train_path = out / 'inputs' / 'train.json'
    rows = json.loads(train_path.read_text(encoding='utf-8'))
    added = {'concept': 0, 'lookup_anchor': 0}
    variants = []
    for repeat in range(3):
        for row in rows:
            if not is_weak_family(row.get('id', '')):
                continue
            variant = dict(row)
            variant['id'] = f"train:gen5c:{repeat}:{row['id']}"
            variant['assessment'] = {'generation5': True, 'weak_family_repeat': repeat + 1}
            variants.append(variant)
            added['concept'] += 1
    for row in rows:
        if is_weak_family(row.get('id', '')) or ':ngu:' in row.get('id', ''):
            continue
        variant = dict(row)
        variant['id'] = f"train:gen5a:{row['id']}"
        variant['assessment'] = {'generation5': True, 'lookup_anchor': True}
        variants.append(variant)
        added['lookup_anchor'] += 1
    rows.extend(variants)
    train_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')

    for name in ('dev', 'test'):
        if (out / 'inputs' / f'{name}.json').read_bytes() != (src / 'inputs' / f'{name}.json').read_bytes():
            raise SystemExit(f'{name} split was modified — refusing to continue')

    config_path = out / 'inputs' / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    training = config['training']
    protocol = json.loads((out / 'protocol.json').read_text(encoding='utf-8'))
    protocol['rows'] = {k: len(json.loads((out / 'inputs' / f'{k}.json').read_text(encoding='utf-8')))
                        for k in ('train', 'dev', 'test')}
    protocol['optimizer_updates_per_epoch'] = math.ceil(
        protocol['rows']['train'] / training['effective_batch_size'])
    protocol['optimizer_updates'] = protocol['optimizer_updates_per_epoch'] * training['epochs']
    protocol['warmup_updates'] = math.ceil(protocol['optimizer_updates'] * training['warmup_ratio'])
    protocol['input_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (out / 'inputs').iterdir()}
    protocol['status'] = 'prepared_not_trained_generation5_full_load'
    protocol['purpose'] = ('Gen-4 concept oversampling plus a 1x lookup-family anchor: '
                           'both signals in one curriculum; dev/test frozen')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print(json.dumps({'train_rows': protocol['rows']['train'], 'added': added,
                      'optimizer_updates': protocol['optimizer_updates']}))


if __name__ == '__main__':
    main()
