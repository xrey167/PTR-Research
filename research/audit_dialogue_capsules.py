"""Verify shared lineage/tokens and replay revocations on isolated DB copies.

Answer semantics are reported for separate review, not scored by this auditor.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from neural_pods.registry import Registry,InvalidState,digest,verify_files
from research.audit_prefix_probe import validate_continuation


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    read=lambda name:json.loads((a.run/name).read_text(encoding='utf-8'))
    links,reader,manifest,cases=map(read,['links.json','reader.json','manifest.json','cases.json'])
    assert links['status']==reader['status']=='completed'
    source_count=None
    if 'source_snapshot' in reader:
        snapshot=(a.run/reader['source_snapshot']).resolve()
        assert snapshot.is_relative_to(a.run.resolve())
        sources=json.loads((snapshot/'manifest.json').read_text(encoding='utf-8'))
        assert 'research/run_dialogue_capsules.py' in sources
        verify_files(snapshot,{**sources,'manifest.json':reader['source_manifest_sha256']})
        source_count=len(sources)
    assert links['base_unchanged'] is True and reader['base_unchanged'] is True
    if reader.get('reader_precision')=='bf16':
        assert reader['quantization'] is None
        assert reader['compiled_capsules']
    assert [r['id'] for r in links['rows']]==[r['id'] for r in reader['rows']]==[r['id'] for r in cases['cases']]
    tok=AutoTokenizer.from_pretrained(manifest['reader_model'],local_files_only=True)
    config=json.loads((Path(manifest['reader_model'])/'generation_config.json').read_text())
    eos=config['eos_token_id'];eos=set(eos if isinstance(eos,list) else [eos]);eos.add(tok.eos_token_id)
    raw=load_file(a.run/'reader-logits.safetensors');r=Registry(a.run/'registry.sqlite3')
    try:
        resumed_count=0
        if 'resumed_from' in reader:
            prior_file=a.run/'resume-source-report.json'
            assert hashlib.sha256(prior_file.read_bytes()).hexdigest()==reader['resume_source_sha256']
            prior=json.loads(prior_file.read_text(encoding='utf-8'))
            assert prior['status']=='failed'
            assert prior['reader_model_sha256']==reader['reader_model_sha256']
            assert prior['reader_precision']==reader['reader_precision']
            assert prior['reader_input_format']==reader['reader_input_format']
            resumed_count=len(prior['rows'])
            assert reader['resumed_rows']==[row['id'] for row in prior['rows']]
            assert reader['rows'][:resumed_count]==prior['rows']
            prior_raw=load_file(a.run/'resume-source-logits.safetensors')
            assert all(key in raw and torch.equal(value,raw[key]) for key,value in prior_raw.items())
            prior_snapshot=a.run/'resume-source-snapshot'/Path(prior['source_snapshot']).relative_to('source_snapshot')
            prior_sources=json.loads((prior_snapshot/'manifest.json').read_text(encoding='utf-8'))
            verify_files(prior_snapshot,{**prior_sources,'manifest.json':prior['source_manifest_sha256']})
        node_count=0
        for key, in r.db.execute('SELECT id FROM nodes'):
            node=r.node(key);parents=sorted(x[0] for x in r.db.execute('SELECT parent FROM edges WHERE child=?',(key,)))
            assert key==node['kind']+':'+digest({'kind':node['kind'],'payload':node['payload'],'parents':parents})
            node_count+=1
        invoked=0
        for link,row,case in zip(links['rows'],reader['rows'],cases['cases']):
            assert row['planner_proof']==link['planner_proof']
            proof=r.node(row['planner_proof'])['payload']['payload']
            assert proof['task']=='dialogue-plan' and proof['question']==case['question'] and proof['history']==case['history']
            assert proof['text']==row['planner_text'] and proof['planner_key']==row['planner_key']
            receipt=row['receipt'];answer=r.node(receipt['answer_id'])
            assert answer['payload']['payload']['text']==row['text']==receipt['text']
            assert sorted(receipt['dependencies'])==answer['payload']['parent_artifact_keys']
            r.snapshot(receipt['dependencies'],'buyer')
            lineage={n['id'] for n in r.ancestors(receipt['answer_id'])}
            assert {row['planner_key'],row['planner_proof']}<=lineage
            pp=r.node(row['planner_key'])['payload']['payload']
            verify_files(a.run/'adapters'/pp['adapter'],pp['files'])
            if row['deferred']:
                assert row['text']==proof['text']=='UNKNOWN' and not row['reader_invoked']
                continue
            invoked+=1
            assert {row['link_adapter'],row['link_proof'],row['representation_key'],row['capsule_key'],row['generation_key']}<=lineage
            lp=r.node(row['link_proof'])['payload']['payload']
            assert lp['question']==row['canonical_lookup']==proof['canonical_lookup']
            assert lp['text']==row['link_text'] and lp['adapter_key']==row['link_adapter']
            cap=r.node(row['capsule_key'])['payload']['payload']
            assert row['generation_key'] in r.node(row['capsule_key'])['payload']['parent_artifact_keys']
            if 'reader_model_sha256' in reader:
                assert cap['model_sha256']==reader['reader_model_sha256']
            if reader.get('reader_precision')=='bf16':
                assert reader['compiled_capsules'][row['generation_key']]==row['capsule_key']
            verify_files(a.run/'capsule_states'/cap['directory'],cap['files'])
            cached,reference=raw[row['id']+'_cached'],raw[row['id']+'_reference']
            assert torch.isfinite(cached).all() and torch.equal(cached,reference)
            validate_continuation(row,cached,tok,eos)
        assert len(raw)==2*invoked
        controls={};sample=next(x for x in reader['rows'] if x['reader_invoked'])
        training_origin=next(n['id'] for n in r.ancestors(sample['planner_key']) if n['kind']=='origin' and n['payload']['namespace']=='planner-training')
        request_origin=next(n['id'] for n in r.ancestors(sample['planner_proof']) if n['kind']=='origin' and n['payload']['namespace']=='fixture:dialogue-request' and n['payload']['record_id']==sample['id'])
        for name,target in [('planner_training',training_origin),('request',request_origin),('link',sample['link_adapter']),('fact',sample['generation_key'])]:
            clone=Registry(':memory:');r.db.backup(clone.db)
            try:
                pending=clone.snapshot(sample['receipt']['dependencies'],'buyer');clone.revoke(target)
                try:clone.commit(pending,sample['text'])
                except InvalidState:controls[name+'_blocks_commit']=True
                else:raise AssertionError('Revocation did not block')
                if name=='request':
                    other=reader['rows'][1];clone.snapshot(other['receipt']['dependencies'],'buyer')
                    controls['request_revocation_selective']=True
                if name=='planner_training':
                    clone.snapshot([sample['link_adapter'],sample['capsule_key']],'buyer')
                    controls['planner_revocation_preserves_other_payloads']=True
            finally:clone.close()
        result={'audit_passed':True,'graph_nodes_verified':node_count,'reader_sequences':invoked,
            'source_files_verified':source_count,
            'inherited_saved_rows_verified':resumed_count,
            'full_logit_pairs_equal':invoked,'controls':controls,'answers':[{ 'id':x['id'],'text':x['text']} for x in reader['rows']],
            'answer_quality_gate_evaluated':False,'full_research_goal_complete':False,
            'limits':'Stored evidence audit and local clone revocations; no independent model replay or answer-semantic scoring. Local imported planner snapshot has no future source synchronization. Inherited rows retain their interrupted source-run evidence; its final weight hash was not measured.'}
        (a.run/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
    finally:r.close()


if __name__=='__main__':main()
