\"\"\"Reproduce the posthoc DS1 root migration check against actual CC4 QueryBridge.\"\"\"
import json,gzip,sys
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(r))
from query_abi_bridge.run_validation import runtime_from_cc4
from query_abi_bridge.core import QueryBridge
from full_shell_code.core import ShellABICompiler
from decoder_semantics.compiler import ProvenanceCompiler
if __name__=='__main__':
 runtime=runtime_from_cc4();a=np.load(r/'query_abi_bridge/development_run/alias_index.npz');bridge=QueryBridge(runtime,a['digests'],a['ids'])
 with gzip.open(r/'query_abi_bridge/run/questions.jsonl.gz','rt') as f:q=next(json.loads(x)['question'] for x in f if json.loads(x)['operator']=='equal')
 qh=bridge.issue(q);old=ShellABICompiler(r);new=ProvenanceCompiler(r);rows=[]
 for name in old.bases:
  oh,ob=old.compile_query(bridge,qh,name);nh,nb=new.compile_query(bridge,qh,name)
  rows.append({'abi':name,'same_tensor_bytes':ob==nb,'old_handle_accepted_by_old':old.admit(bridge,oh,ob),'old_handle_rejected_by_new':not new.admit(bridge,oh,ob),'fresh_handle_accepted_by_new':new.admit(bridge,nh,nb)})
 assert all(all(v for k,v in x.items() if k"'!='"'abi') for x in rows)
 (r/'decoder_semantics/root_migration_test.json').write_text(json.dumps(rows,indent=2)+'\\n');print(json.dumps(rows,indent=2))
