import json
from pathlib import Path

def test_neohorse_reference_manifest_and_probe():
    m=json.loads(Path('runs/neohorse-reference-manifest-001.json').read_text())
    assert m['model']=='TokenRhythm/NeoHorse-1-4B'
    assert m['published_macro_average']['delta']==5.93
    assert len(m['published_benchmarks'])==10
    probe=json.loads(Path(m['local_probe']).read_text())
    assert probe['schema']=='neohorse-real-checkpoint-probe:v1'
    assert probe['total_generated_tokens']>0
    assert probe['mean_tokens_per_s']>0
