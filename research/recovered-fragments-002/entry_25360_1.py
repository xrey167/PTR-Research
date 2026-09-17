"""DS1 compiler: archival model provenance, new root, unchanged CC8 coordinates."""
import json,hashlib
from pathlib import Path
from full_shell_code.core import ShellABICompiler
from operator_forest.core import canonical,sha

class ProvenanceCompiler(ShellABICompiler):
    def __init__(self,root):
        root=Path(root)
        seal=json.loads((root/'full_shell_code/candidate_seal.json').read_text())
        for path,digest in {**seal['sources'],**seal['assets']}.items():
            if hashlib.sha256((root/path).read_bytes()).hexdigest()!=digest:
                raise ValueError('archived candidate changed: '+path)
        super().__init__(root)
        a=json.loads((root/'model_attestation.json').read_text())
        b=json.loads((root/'stronger/model_download_manifest.json').read_text())
        if a['downloaded']['sha256']!=a['upstream_weights']['lfs']['sha256']:
            raise ValueError('Qwen3 archival weight mismatch')
        records={'qwen3_0_6b':a,'qwen2_5_3b':b}
        contracts={}
        for name,d in records.items():
            self.models[name]=(d['model'],d['revision'])
            weights=({'model.safetensors':a['downloaded']['sha256']} if name=='qwen3_0_6b' else
                     {k:v['sha256'] for k,v in b['files'].items() if k.endswith('.safetensors')})
            if not weights:raise ValueError('missing weights')
            payload={'schema':'ds1-provenance-abi-v1','name':name,'model':d['model'],'revision':d['revision'],
                     'weights':weights,'attestation_sha256':sha(canonical(d)).hex(),
                     'asset_sha256':[sha(p.read_bytes()).hex() for p in self.assets[name]],
                     'cc8_source_sha256':sha((root/'full_shell_code/core.py').read_bytes()).hex(),
                     'compiler_source_sha256':sha(Path(__file__).read_bytes()).hex(),
                     'gain':self.gains[name],'code':'CC8 norm-one E8 shell, first 10005 ranks','gap_sq':0.01}
            contracts[name]=payload;self.roots[name]=sha(canonical(payload))
        self.provenance=contracts
        self.contract_bytes=canonical({'schema':'ds1-contract-v1','abis':contracts})
