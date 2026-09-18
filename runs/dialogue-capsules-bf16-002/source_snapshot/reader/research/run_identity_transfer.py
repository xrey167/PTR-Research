"""Real frozen MiniLM + trained identity addresses; value update with no gradients.

Qwen link LoRA is reused and actually invoked before and after the update.
This experiment tests addresses, not generation of the new factual answer.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from sentence_transformers import SentenceTransformer
from neural_pods.registry import Registry, InvalidState, verify_files
from neural_pods.model import PodModel
from research.identity_dragonfly import IdentityDragonfly
from run_semantic_experiment import TRAIN, HELD_OUT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path('runs/embedded-symlink-002'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); out = args.output.resolve(); source = args.source.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source/'report.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'completed'
    src = sqlite3.connect((source/'registry.sqlite3').as_uri()+'?mode=ro', uri=True)
    dst = sqlite3.connect(out/'registry.sqlite3'); src.backup(dst); src.close(); dst.close()
    for name in ['qdrant','adapters']: shutil.copytree(source/name, out/name)
    report = {'status':'running', 'scope':'Existing fictional lookup regression; identity training once, then zero gradients on value update',
              'training':[], 'predictions':[], 'full_research_goal_complete':False}
    started = time.perf_counter()
    def save():
        report['elapsed_s'] = time.perf_counter()-started
        (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    reg = Registry(out/'registry.sqlite3'); router = None
    try:
        torch.set_num_threads(4)
        info = manifest['models']['encoder']
        encoder = SentenceTransformer(info['path'], device='cpu', local_files_only=True)
        router = IdentityDragonfly(reg, encoder, out, encoder_id=info['repo']+'@'+info['revision'])
        items = [i for i in router.bindings() if i['node']['semantic']['role']=='FACT'
                 and reg.head(i['node']['knowledge_key'])==i['generation_key']]
        items.sort(key=lambda i:i['node']['semantic']['subject'])
        assert [i['node']['semantic']['subject'] for i in items] == ['supplier:muller','supplier:muller-werke']
        def other(q): return re.sub(r'Müller GmbH|Muller GmbH|Mueller GmbH|Müller|Muller|Mueller', 'Müller Werke', q)
        questions = [TRAIN, [other(q) for q in TRAIN]]
        for i, item in enumerate(items):
            metric = router.fit_pod(item['generation_key'], questions[i], questions[1-i], principal='buyer',
                negative_aliases=items[1-i]['node']['semantic']['retrieval']['trusted_aliases'])
            report['training'].append(metric)
        save()
        before = router.select(HELD_OUT[0], principal='buyer')
        identity = router.links.binding(before['knowledge_key'])['identity_key']
        saved_rep = reg.node(before['representation_key'])
        assert before['generation_key'] not in {n['id'] for n in reg.ancestors(before['representation_key'])}
        model = PodModel(manifest['models']['qwen']['path'])
        assert model.frozen_hash() == manifest['base_weights_sha256']
        adapter = router.links.active_binding(before['knowledge_key'], 'buyer')['adapter_key']
        payload = reg.node(adapter)['payload']['payload']
        path = (out/'adapters'/payload['adapter']).resolve()
        assert path.parent == (out/'adapters').resolve()
        verify_files(path, payload['files'])
        model.model.load_adapter(path, adapter_name=payload['adapter'], is_trainable=False)
        def predictions(stage):
            for q in HELD_OUT:
                selected = router.select(q, principal='buyer')
                generated = model.generate(q, adapter=payload['adapter'], task='link')
                resolved = router.links.resolve(generated['text'], expected_knowledge_key=selected['knowledge_key'], principal='buyer')
                assert selected['generation_key'] == resolved['generation_key']
                snapshot = reg.snapshot([*selected['snapshot'].artifacts, *resolved['snapshot'].artifacts], 'buyer')
                receipt = reg.commit(snapshot, generated['text'])
                report['predictions'].append({'stage':stage, 'question':q, **generated,
                    'generation_key':selected['generation_key'], 'representation_key':selected['representation_key'],
                    'adapter_key':resolved['adapter_key'], 'receipt':receipt})
                save()
        predictions('before')
        def forbidden(*a, **kw): raise AssertionError('Unexpected gradient/training after initial address fit')
        router.fit_pod = forbidden
        model.train_adapter = forbidden
        original = reg.node(before['generation_key'])['payload']
        semantic = json.loads(json.dumps(original['semantic'])); semantic['object']['value'] = 27
        origin = reg.origin('fixture:identity-transfer','new-value','1',semantic,acl=['buyer'])
        generation = reg.publish(before['knowledge_key'],semantic,[origin],'buyer',['buyer'],
                                 lifecycle=original['lifecycle'])
        artifact = reg.artifact('text',{'scope':'Canonical value update; no answer LoRA retrained'},[generation],'buyer')
        router.index(generation,artifact,'buyer')
        predictions('after')
        after = router.select(HELD_OUT[0], principal='buyer')
        assert after['representation_key']==before['representation_key'] and reg.node(after['representation_key'])==saved_rep
        assert after['score']==before['score']
        assert router.links.binding(before['knowledge_key'])['identity_key']==identity
        try: reg.commit(before['snapshot'],'old value')
        except InvalidState: report['stale_commit_blocked']=True
        else: raise AssertionError('Stale snapshot accepted')
        router.close(); router=IdentityDragonfly(reg,encoder,out,encoder_id=info['repo']+'@'+info['revision'])
        router.load_weights()
        fresh=router.select(HELD_OUT[0],principal='buyer')
        assert fresh['representation_key']==after['representation_key'] and fresh['score']==after['score']
        report.update(reload_equal=True, representation_unchanged=True, link_adapter_unchanged=True,
            base_unchanged=model.frozen_hash()==manifest['base_weights_sha256'],
            new_value=27, new_generation=generation, factual_answer_generation_tested=False)
        assert report['base_unchanged']
        report['status']='completed'; save(); print(json.dumps({k:v for k,v in report.items() if k not in ['training','predictions']}))
    except Exception as exc:
        report.update(status='failed',error=repr(exc));save();raise
    finally:
        if router is not None: router.close()
        reg.close()


if __name__=='__main__': main()
