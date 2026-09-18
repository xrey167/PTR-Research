"""Generation-4 reader inputs: oversample the concept-question families that
caused the Generation-3 guarded-match failures.

Failure analysis of the frozen Gen-3 dev/test evaluations (40 misses each,
identical distribution): 32 misses from model_pod_generation2 rows
(retrieval/reasoning/model tasks) and 8 from typed_topic rows. The Gen-3
train split contained only 20 rows from these families while dev/test
carry 40 cases each, so the adapter never saw enough exact-target variants.

Gen-4 keeps dev/test byte-identical (frozen, comparable A/B against the
Gen-3 numbers) and 3x-oversamples the weak families in train only.
"""
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

WEAK_FAMILY_MARKERS = ('model_pod_generation2', ':topic:', ':generation2:', ':model:')


def is_weak_family(row_id: str) -> bool:
    return ':ngu:' not in row_id and any(m in row_id for m in WEAK_FAMILY_MARKERS)


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    if out.exists():
        raise SystemExit(f'output directory already exists: {out}')
    shutil.copytree(src, out)

    train_path = out / 'inputs' / 'train.json'
    rows = json.loads(train_path.read_text(encoding='utf-8'))
    weak = [r for r in rows if is_weak_family(r.get('id', ''))]
    added = 0
    for repeat in range(3):
        for row in weak:
            variant = dict(row)
            variant['id'] = f"train:gen4:{repeat}:{row['id']}"
            variant['assessment'] = {'generation4': True, 'weak_family_repeat': repeat + 1}
            rows.append(variant)
            added += 1
    train_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')

    # dev/test must stay byte-identical to the frozen Gen-3 splits; verify.
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
    protocol['status'] = 'prepared_not_trained_generation4_full_load'
    protocol['purpose'] = ('3x oversampling of the weak concept families '
                           '(model_pod_generation2, typed_topic) identified by the '
                           'Gen-3 guarded-match failure analysis; dev/test frozen')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print(json.dumps({'train_rows': protocol['rows']['train'], 'oversampled': added,
                      'optimizer_updates': protocol['optimizer_updates']}))


if __name__ == '__main__':
    main()
