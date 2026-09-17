\"\"\"Create a checksummed research snapshot, without pretrained weights/dependencies.\"\"\"
import argparse,hashlib,json,zipfile
from pathlib import Path

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 root=Path(__file__).parent
 files=sorted(f for f in root.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix not in ['.tmp','.zip'] and f.name"'!='"'BUNDLE_MANIFEST.json')
 entries=[dict(path=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files]
 manifest=dict(status='complete' if (root/'run/results.json').exists() else 'partial',files=entries)
 # Bundle mutable checkpoints from byte snapshots so hash and bytes agree.
 with zipfile.ZipFile(a.output,'w',zipfile.ZIP_DEFLATED) as z:
  for entry,f in zip(entries,files):
   data=f.read_bytes();entry.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());z.writestr(root.name+'/'+entry['path'],data)
  z.writestr(root.name+'/BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2)+'\\n')
 with zipfile.ZipFile(a.output) as z:assert z.testzip() is None
 print(json.dumps(dict(path=str(a.output.resolve()),bytes=a.output.stat().st_size,status=manifest['status'],files=len(files))))
