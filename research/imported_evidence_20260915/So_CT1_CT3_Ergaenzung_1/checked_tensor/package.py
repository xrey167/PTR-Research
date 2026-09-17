from pathlib import Path
import hashlib,json,zipfile

root=Path(__file__).resolve().parent
output=root.parent/'So_CT1_CT3_Ergaenzung.zip'
manifest=[]
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts:continue
        data=p.read_bytes();name=str(p.relative_to(root))
        manifest.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        z.writestr('checked_tensor/'+name,data)
    z.writestr('MANIFEST.json',json.dumps({'files':manifest,'primary_archive_modified':False,'full_dod_pass':False},indent=2))
with zipfile.ZipFile(output) as z:
    assert z.testzip() is None
    for e in manifest:
        data=z.read('checked_tensor/'+e['path'])
        assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256']
print(json.dumps({'path':str(output),'files':len(manifest),'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))
