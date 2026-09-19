"""Generation-7 reader inputs: the DREAMED curriculum.

The Dream Pod's replay simulator (research/runs/dream-cycle-20260920.json)
predicted the best strategy: concept_oversample 3 + lookup_anchor 2 on the
NeoHorse-1-4B base. This script materializes that dream: Gen-6 inputs plus
a SECOND lookup anchor copy. The frozen dev/test splits stay byte-identical
— the dream prediction is evaluated against them unchanged.
"""
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

from prepare_generation4 import is_weak_family


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    if out.exists():
        raise SystemExit(f'output directory already exists: {out}')
    shutil.copytree(src, out)

    for name in ('dev', 'test'):
        if (out / 'inputs' / f'{name}.json').read_bytes() != (src / 'inputs' / f'{name}.json').read_bytes():
            raise SystemExit(f'{name} split was modified — refusing to continue')

    train_path = out / 'inputs' / 'train.json'
    rows = json.loads(train_path.read_text(encoding='utf-8'))
    anchors = 0
    for row in rows:
        # The first anchor (gen5-style) is already in src; add one MORE copy
        # of every non-dream lookup row (ids without the gen5a/gen5c prefixes).
        if (':gen5a:' in row.get('id', '') or ':gen5c:' in row.get('id', '')
                or ':ngu:' in row.get('id', '') or is_weak_family(row.get('id', ''))):
            continue
        variant = dict(row)
        variant['id'] = f"train:gen7a:{row['id']}"
        variant['assessment'] = {'generation7': True, 'dreamed': True, 'lookup_anchor': 2}
        rows.append(variant)
        anchors += 1
    train_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')

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
    protocol['status'] = 'prepared_not_trained_generation7_dreamed'
    protocol['purpose'] = ('Dreamed curriculum (dream-cycle-20260920 winner): concept '
                           'oversample 3 + lookup anchor 2 on NeoHorse base; '
                           'dev/test frozen — dream prediction vs evidence')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print(json.dumps({'train_rows': protocol['rows']['train'], 'anchors_added': anchors,
                      'optimizer_updates': protocol['optimizer_updates']}))


if __name__ == '__main__':
    main()
