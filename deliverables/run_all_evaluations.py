"""Run the complete local acceptance and generalization evaluation stack."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def main():
    commands=[[sys.executable,'research/run_project_gate.py'],
              [sys.executable,'research/run_generalization_gate.py']]
    results=[]
    for cmd in commands:
        p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
        results.append({'command':cmd,'returncode':p.returncode,
                        'output_tail':(p.stdout+p.stderr)[-2000:]})
    out={'schema':'neural-pods-combined-evaluation:v1','passed':all(x['returncode']==0 for x in results),
         'system_gate':'runs/project-gate-001.json','generalization_gate':'runs/generalization-gate-001.json',
         'results':results}
    path=ROOT/'runs/combined-evaluation-001.json'; path.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':out['passed'],'gates':len(results),'report':str(path.relative_to(ROOT))},indent=2))
    return 0 if out['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
