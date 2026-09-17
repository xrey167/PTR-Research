"""Recount delivered observations; no training, no test-selected checkpoint."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from cases import score,strict_ok,gold


def inspect(run):
    path=run/'results.json';d=json.loads(path.read_text());cases=json.loads((run/'cases.json').read_text())
    ids=[{c['identity'] for c in cases[k]} for k in ['train','valid','test']]
    assert all(not ids[i]&ids[j] for i in range(3) for j in range(i))
    checks={name:hashlib.sha256((run.parent/name).read_bytes()).hexdigest()==value for name,value in d['source_hashes'].items()}
    assert all(checks.values())
    out=[]
    for r in d['runs']:
        for key in ['generation','shifted_generation']:
            g=r[key]
            for row in g['rows']:
                assert row['expected']==gold(row['case']['spec'],row['case']['value'])
                assert row['correct']==strict_ok(row['text'],row['expected'],row['eos_seen'])
                assert row['shifted_correct']==strict_ok(row['text'],row['shifted_expected'],row['eos_seen'])
            assert score(g['rows'])==g['score']
        # Confirm validation-only family-first selection, earliest in ties.
        selected=max(r['history'],key=lambda x:(x['worst_family'],x['score']['overall']['accuracy'],-x['step']))
        assert selected==r['selected']
        eligible=[i for i,c in enumerate(cases['test']) if gold(c['spec'],c['value'])!=gold(c['spec'],(c['value']+1)%4)]
        pair=sum(r['generation']['rows'][i]['correct'] and r['shifted_generation']['rows'][i]['shifted_correct'] for i in eligible)
        assert r['causal_pairs']['n']==len(eligible) and r['causal_pairs']['correct']==pair
        worlds=defaultdict(list)
        for row in r['generation']['rows']:
            c=row['case'];worlds[(c['identity'],c['spec'],c['wording'])].append(row)
        assert all(len(rs)==4 and {r['case']['value'] for r in rs}==set(range(4)) for rs in worlds.values())
        out.append(dict(method=r['method'],seed=r['seed'],selected_step=r['selected']['step'],
            full_answer_score=r['generation']['score'],causal_pairs=r['causal_pairs'],
            all_four_worlds_correct=dict(correct=sum(all(x['correct'] for x in rows) for rows in worlds.values()),n=len(worlds)),
            parameters=r['shared_parameters'],preliminary_repair_gate=r['numerical']['numerical_gate'],
            interface_gate_pass=r['interface_gate_pass'],teacher_autoreg_error=r['teacher_vs_autoregressive_eos_logit_error']))
    return dict(experiment=d['experiment'],sources_verified=checks,ids_per_split=list(map(len,ids)),runs=out)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).parent)
    args=parser.parse_args();results=[inspect(args.root/name) for name in ['run_c1','run_c2']]
    (args.root/'analysis.json').write_text(json.dumps(results,indent=2)+'\n')
    for d in results:
        for r in d['runs']:
            print(d['experiment'],r['method'],r['seed'],r['selected_step'],r['full_answer_score']['groups'],r['causal_pairs'],r['all_four_worlds_correct'])
