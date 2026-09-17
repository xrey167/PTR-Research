"""Standalone raw-tensor audit, no import of candidate evaluators."""
from pathlib import Path
from fractions import Fraction
import hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    seal=json.loads((HERE/'results/seal.json').read_text())
    for n,h in seal['sources'].items():assert hashlib.sha256((HERE/n).read_bytes()).hexdigest()==h
    data=np.load(HERE/'results/raw_tensors.npz',allow_pickle=False)
    rows=json.loads((HERE/'results/rows.json').read_text());assert len(rows)==36
    cells=0
    for row in rows:
        m,d,k=row['shape'];tag=f"{row['seed']}_{m}_{d}_{k}"
        mats={n:data[tag+'_'+n] for n in ('B','S','R','C','Y','X')}
        o=data[tag+'_output'];new=data[tag+'_new_output']
        for i in range(m):
            for j in range(d):
                # Independent integer identity: all entries are multiples of 1/16.
                base=sum(int(mats[n][i,j]*16) for n in ('B','S','R'))*16
                correction=sum(int(mats['C'][i,l]*16)*(int(mats['Y'][l,j]*16)-int(mats['X'][l,j]*16)) for l in range(k))
                expected=Fraction(base+correction,256)
                assert expected==Fraction.from_float(float(o[i,j]))
                edited=expected+(Fraction.from_float(float(mats['C'][i,0])) if j==0 else 0)
                assert edited==Fraction.from_float(float(new[i,j]));cells+=1
        assert all(row[n] for n in ('fresh_accept','exact_fraction_match','stale_rejected','relabelled_old_output_rejected',
                                    'new_output_accepted','undeclared_input_rejected','aba_old_evidence_rejected',
                                    'aba_new_evidence_accepted','aba_output_unchanged'))
    structural=json.loads((HERE/'results/structural_controls.json').read_text())
    assert len(structural)==6 and all(r['rejected'] for r in structural)
    cert=json.loads((HERE/'cert_results/results.json').read_text())
    field=json.loads((HERE/'field_results/results.json').read_text())
    for c in cert['costs']:
        assert c['direct_mac_terms']==c['m']*c['k']*c['d']
        assert c['verify_mac_terms']==40*(c['k']*c['d']+c['m']*c['k']+c['m']*c['d'])
    for c in field['costs']:
        m=c['pair_count'];assert c['direct_mac_terms']==m*6*256
        assert c['verification_mac_terms']==2*(6*256+m*6+m*256)
    assert field['on_query_cost_gate'] is False
    result={'passed':True,'cases':36,'output_cells_independently_checked':cells,
      'edited_output_cells_independently_checked':cells,'structural_control_rows':6,
      'scope':'same-assistant independently implemented integer/Fraction audit of saved numerical data',
      'not_audit_of':'LLM output, genuine project tensors, production sandbox, remote attestation',
      'full_dod_pass':False}
    (HERE/'results/audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
