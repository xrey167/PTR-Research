"""Repeat real-model checks on a saved run; mutate only an in-memory registry."""
import argparse
import json
from pathlib import Path
import statistics
import time

from sentence_transformers import SentenceTransformer
from neural_pods.dragonfly import DragonflyRouter
from neural_pods.model import PodModel
from neural_pods.registry import Registry, InvalidState, digest, verify_files
from neural_pods.symlink import NeuralSymlinks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    manifest = json.loads((args.run / 'report.json').read_text(encoding='utf-8'))
    original = json.loads((args.run / 'symlink-report.json').read_text(encoding='utf-8'))
    result = {'status': 'running', 'links': [], 'answers': [], 'aliases': [], 'blocked': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    source = Registry(args.run / 'registry.sqlite3')
    reg = Registry(':memory:')
    source.db.backup(reg.db)
    source.close()
    router = None
    try:
        info = manifest['models']['encoder']
        encoder = SentenceTransformer(info['path'], device='cpu', local_files_only=True)
        router = DragonflyRouter(reg, encoder, args.run, encoder_id=info['repo'] + '@' + info['revision'])
        router.load_weights()
        links = NeuralSymlinks(reg)
        model = PodModel(manifest['models']['qwen']['path'])
        assert model.frozen_hash() == manifest['base_weights_sha256']
        result['load_s'] = time.perf_counter() - started
        verified = 0
        for row in reg.db.execute('SELECT id FROM nodes').fetchall():
            node = reg.node(row['id'])
            parents = sorted(r['parent'] for r in reg.db.execute('SELECT parent FROM edges WHERE child=?', (row['id'],)))
            assert node['id'] == node['kind'] + ':' + digest({'kind': node['kind'], 'payload': node['payload'], 'parents': parents})
            if node['kind'] == 'lora':
                p = node['payload']['payload']
                verify_files(args.run / 'adapters' / p['adapter'], p['files'])
                verified += 1
        result['integrity'] = {'nodes': reg.db.execute('SELECT count(*) FROM nodes').fetchone()[0], 'lora_artifacts': verified}

        def load(artifact):
            p = reg.node(artifact)['payload']['payload']
            if p['adapter'] not in model.model.peft_config:
                model.model.load_adapter(args.run / 'adapters' / p['adapter'], adapter_name=p['adapter'], is_trainable=False)
            return p['adapter']

        for row in original['link_predictions']:
            binding = links.active_binding(row['knowledge_key'], 'buyer')
            predicted = model.generate(row['question'], adapter=load(binding['adapter_key']), task='link')
            expected = links.target_text(row['knowledge_key'])
            result['links'].append({'question': row['question'], 'expected': expected, **predicted})
            assert predicted['text'] == expected
            links.resolve(predicted['text'], expected_knowledge_key=row['knowledge_key'], principal='buyer')
        print('18 saved held-out link/cluster cases checked', flush=True)

        questions = list(dict.fromkeys(r['question'] for r in original['answers']))
        for question in questions:
            t = time.perf_counter()
            selected = router.select(question, principal='buyer')
            binding = links.active_binding(selected['knowledge_key'], 'buyer')
            predicted = model.generate(question, adapter=load(binding['adapter_key']), task='link')
            resolved = links.resolve(predicted['text'], expected_knowledge_key=selected['knowledge_key'], principal='buyer')
            fresh = router.select(question, principal='buyer')
            assert fresh['generation_key'] == resolved['generation_key']
            snap = reg.snapshot([*fresh['snapshot'].artifacts, *resolved['snapshot'].artifacts], 'buyer')
            answer = model.generate(question, adapter=load(fresh['artifact_key']))
            receipt = reg.commit(snap, answer['text'])
            result['answers'].append({'question': question, 'link': predicted['text'], **answer,
                                      'total_s': time.perf_counter() - t, 'receipt': receipt})
            assert answer['text'] == '18 days'
            assert binding['adapter_key'] in receipt['dependencies']
        for alias in ['Mueller', 'Mueller GmbH', 'Muller GmbH', 'M\u00fcller', 'M\u00fcller GmbH', 'M\u00fcller Werke', 'Mueller Werke']:
            t = time.perf_counter()
            recognized = router.recognize_alias(alias, principal='buyer')
            expected = 'supplier:muller-werke' if 'Werke' in alias else 'supplier:muller'
            assert recognized['subject'] == expected, recognized
            result['aliases'].append({'alias': alias, 'subject': expected, 'elapsed_s': time.perf_counter() - t})
        for question, principal in [('How long does Mueller need for X99?', 'buyer'), ('How old is Mueller GmbH?', 'buyer'), (questions[0], 'outsider')]:
            try:
                router.select(question, principal=principal)
            except InvalidState:
                result['blocked'].append({'question': question, 'principal': principal})
            else:
                raise AssertionError('Expected blocked query')
        pending = links.resolve(predicted['text'], expected_knowledge_key=selected['knowledge_key'], principal='buyer')['snapshot']
        origin = reg.node(selected['generation_key'])['payload']['parents'][0] if 'parents' in reg.node(selected['generation_key'])['payload'] else reg.db.execute('SELECT parent FROM edges WHERE child=?', (selected['generation_key'],)).fetchone()[0]
        reg.revoke(origin)
        for label, operation in [('revoked_commit', lambda: reg.commit(pending, '18 days')), ('revoked_resolve', lambda: links.resolve(predicted['text'], expected_knowledge_key=selected['knowledge_key'], principal='buyer'))]:
            try:
                operation()
            except InvalidState:
                result['blocked'].append({'operation': label})
            else:
                raise AssertionError(label)
        result['base_unchanged'] = model.frozen_hash() == manifest['base_weights_sha256']
        assert result['base_unchanged']
        result['answer_median_s'] = statistics.median(r['total_s'] for r in result['answers'])
        result['status'] = 'passed'
    except Exception as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        result['elapsed_s'] = time.perf_counter() - started
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        if router is not None:
            router.close()
        reg.close()
    print(json.dumps({k: result[k] for k in ['status', 'elapsed_s', 'answer_median_s', 'integrity']}))


if __name__ == '__main__':
    main()
