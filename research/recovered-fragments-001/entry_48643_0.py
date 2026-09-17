\"\"\"Independent recount of reported case-level outcomes and selection rules.\"\"\"
import argparse,hashlib,json,re
from pathlib import Path
from collections import defaultdict
from cases import datasets,gold,score,strict_ok


def main(root):
    d=json.loads((root/'run/results.json').read_text());data=datasets();saved=json.loads((root/'run/cases.json').read_text());assert saved==data
    assert {(r['method'],r['seed']) for r in d['runs']}=={(m,s) for m in ['affine','bounded','coupling'] for s in [17,29]}
    for name,sha in d['source_hashes'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
    summaries=[]
    for run in d['runs']:
        for name in ['generation','shifted_generation']:
            g=run[name];assert [r['case'] for r in g['rows']]==data['test']
            for row in g['rows']:
                c=row['case'];expected=gold(c['spec'],c['value']);shifted=gold(c['spec'],(c['value']+1)%4)
                assert row['expected']==expected and row['shifted_expected']==shifted
                assert row['correct']==strict_ok(row['text'],expected,row['eos_seen'])
                assert row['shifted_correct']==strict_ok(row['text'],shifted,row['eos_seen'])
            assert score(g['rows'])==g['score']
        selected=max(run['history'],key=lambda r:(r['worst_family'],r['score']['overall']['accuracy'],-r['step']))
        assert selected==run['selected']
        eligible=[i for i,c in enumerate(data['test']) if gold(c['spec'],c['value'])"'!=gold(c['"'spec'],(c['value']+1)%4)]
        correct=sum(run['generation']['rows'][i]['correct'] and run['shifted_generation']['rows'][i]['shifted_correct'] for i in eligible)
        assert run['causal_pairs']==dict(n=len(eligible),correct=correct,accuracy=correct/len(eligible))
        worlds=defaultdict(list)
        for row in run['generation']['rows']:
            c=row['case'];worlds[(c['identity'],c['spec'],c['wording'])].append(row)
        assert len(worlds)==60 and all({r['case']['value'] for r in rows}==set(range(4)) for rows in worlds.values())
        num=run['numerical'];pairmax=max(p['max_relative_error'] for p in num['pairwise']);seqmax=max(max(s['all_step_max_relative_errors']) for s in num['sequence'])
        for s in num['sequence']:assert len(s['all_step_max_relative_errors'])==256 and max(s['all_step_max_relative_errors'])==s['maximum']
        assert sum(p['n'] for p in num['pairwise'])==375 and num['unique_queries']==15
        numerical=pairmax<1e-5 and seqmax<1e-5 and num['sequence_top_agreement']==15 and all(p['top_agreement']==p['n'] for p in num['pairwise'])
        assert num['numerical_gate']==numerical
        assert run['autoregressive_check']['passed']==(run['autoregressive_check']['max_logit_error']<1e-3)
        pilot=all(s['accuracy']>=.9 for s in run['generation']['score']['operations'].values()) and correct/len(eligible)>=.9 and numerical and run['autoregressive_check']['passed']
        assert run['interface_gate_pass']==pilot
        summaries.append(dict(method=run['method'],seed=run['seed'],selected_step=run['selected']['step'],score=run['generation']['score'],causal_pairs=run['causal_pairs'],all_worlds=dict(n=60,correct=sum(all(r['correct'] for r in rows) for rows in worlds.values())),max_pair_error=pairmax,max_sequence_error=seqmax,numerical_gate=numerical,autoreg_error=run['autoregressive_check']['max_logit_error'],parameters=run['parameters'],pilot_pass=pilot))
    for g in d['controls'].values():
        for r in g['rows']:assert r['correct']==strict_ok(r['text'],gold(r['case']['spec'],r['case']['value']),r['eos_seen'])
        assert score(g['rows'])==g['score']
    expected=(root.parent/'So_QC_B_Conditional_ABI/FROZEN_DOD.md').read_bytes();assert (root/'FROZEN_DOD.md').read_bytes()==expected
    output=dict(verification_passed=True,source_hashes_verified=True,selection_recounted=True,generation_and_pair_scores_recounted=True,numerical_gates_recounted=True,frozen_dod_unchanged=True,full_retraining_repeated=False,total_optimizer_steps=sum(r['steps'] for r in d['runs']),runs=summaries)
    (root/'verified_summary.json').write_text(json.dumps(output,indent=2)+'\\n')
    for r in summaries:print(r['method'],r['seed'],r['score']['groups'],r['causal_pairs'],r['all_worlds'],'repair',r['max_pair_error'],r['max_sequence_error'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).parent);main(p.parse_args().root)
