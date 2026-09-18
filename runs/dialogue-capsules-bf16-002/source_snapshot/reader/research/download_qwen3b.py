"""Download a pinned public model and verify published weight checksums."""
import hashlib
import json
from pathlib import Path
import time
from huggingface_hub import snapshot_download


def main():
    metadata = json.loads(Path('research/qwen3b-upstream.json').read_text(encoding='utf-8'))
    destination = Path('models/qwen3b')
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / 'download-manifest.json'
    result = {'status':'downloading','repo':metadata['id'],'revision':metadata['sha'],'files':{},
              'original_project_revision_match':'unverified; original model manifest unavailable'}
    start = time.perf_counter()
    def save():
        result['elapsed_s'] = time.perf_counter() - start
        manifest_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    save()
    selected = [x for x in metadata['siblings'] if x['rfilename'].endswith(('.safetensors','.json')) or x['rfilename'] in ['merges.txt','LICENSE']]
    try:
        snapshot_download(repo_id=result['repo'], revision=result['revision'], local_dir=destination,
                          allow_patterns=[x['rfilename'] for x in selected], max_workers=2)
        for row in selected:
            path = destination / row['rfilename']
            with path.open('rb') as f: actual = hashlib.file_digest(f,'sha256').hexdigest()
            if row.get('lfs'):
                assert actual == row['lfs']['sha256'], row['rfilename']
                assert path.stat().st_size == row['lfs']['size'], row['rfilename']
            result['files'][row['rfilename']] = {'sha256':actual,'bytes':path.stat().st_size,
                                                'upstream_sha256_verified':bool(row.get('lfs'))}
            save()
        result['status'] = 'verified'
    except Exception as exc:
        result.update(status='failed',error=repr(exc))
        raise
    finally: save()
    print(json.dumps({'status':result['status'],'revision':result['revision'],'files':len(result['files']),'elapsed_s':result['elapsed_s']}))


if __name__ == '__main__': main()
