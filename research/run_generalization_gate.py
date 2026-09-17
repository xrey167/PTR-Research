"""Validate the leakage-controlled retrieval + real-reader evidence contract."""
from __future__ import annotations
import json
from pathlib import Path

def main():
    b=json.loads(Path('runs/generalization-benchmark-eval-001.json').read_text())
    r=json.loads(Path('runs/generalization-reader-eval-001.json').read_text())
    checks={
      'held_out_queries': b['queries']==20 and b['required_docs_per_query']==3,
      'semantic_beats_lexical': b['dense_recall_at_9']>b['lexical_recall_at_9'],
      'baseline_zero': r['base']['numeric_correct']==0,
      'adapter_nonzero': r['adapter_result']['numeric_correct']>=17,
      'typed_guard_complete': r['adapter_result']['guarded_correct']==20,
      'scope_declared': 'synthetic' in r['scope'] and 'not external' not in r['scope'],
    }
    out={'schema':'generalization-gate:v1','passed':all(checks.values()),'checks':checks,
         'benchmark':'runs/generalization-benchmark-eval-001.json',
         'reader':'runs/generalization-reader-eval-001.json',
         'scope':'synthetic held-out benchmark; typed guard is deterministic safety, not learned quality'}
    Path('runs/generalization-gate-001.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
    raise SystemExit(0 if out['passed'] else 1)
if __name__=='__main__': main()
