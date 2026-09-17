import hashlib, json, shutil, sys
from pathlib import Path
b=Path(sys.argv[1]); root=Path(sys.argv[2]); snap=b/'source_snapshot'
for n in ['reader_prompt.py','reader_training.py','reader_identity.py','prepare_reader_training.py','prefix_capsule.py','progress_json.py']:
    (snap/'research').mkdir(parents=True,exist_ok=True); shutil.copy2(root/'research'/n,snap/'research'/n)
(snap/'neural_pods').mkdir(parents=True,exist_ok=True); shutil.copy2(root/'neural_pods/registry.py',snap/'neural_pods/registry.py')
p=json.loads((b/'protocol.json').read_text())
p['source_hashes']={x.relative_to(snap).as_posix():hashlib.sha256(x.read_bytes()).hexdigest() for x in snap.rglob('*.py')}
(b/'protocol.json').write_text(json.dumps(p,indent=2))
