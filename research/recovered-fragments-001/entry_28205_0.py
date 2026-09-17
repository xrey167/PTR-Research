\"\"\"Analytic byte accounting for actual tensor shapes, not measured peak RSS.\"\"\"
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent.parent
c=json.loads((ROOT.parent.parent/'models/Qwen2.5-3B-Instruct/config.json').read_text())
layers=c['num_hidden_layers'];heads=c['num_key_value_heads'];width=c['hidden_size']//c['num_attention_heads'];tokens=49
assert (layers,heads,width)==(36,2,128)
elements=layers*2*heads*tokens*width
image_bf16=elements*2;image_f64=elements*8
factor_elems=layers*2*tokens*(heads*width)*6
out=dict(scope='exact dense tensor payload counts from prototype shapes; excludes metadata, weights, allocator and temporary workspaces; not a serving benchmark',image_bf16_bytes=image_bf16,target_eight_calibration_images_bf16_bytes=8*image_bf16,naive_all_32_world_images_bf16_bytes=32*image_bf16,inverse_plus_shift_f64_bytes=2*factor_elems*8,materialized_15_pair_terms_f64_bytes=15*image_f64,explicit_local_square_operator_f64_bytes=layers*2*tokens*(heads*width)**2*8,claim='Implicit application avoids the full local square operator, but no memory Pareto advantage over ordinary prefix sharing has been demonstrated.',full_dod_pass=False)
(ROOT/'binding_transfer/tensor_costs.json').write_text(json.dumps(out,indent=2)+'\\n');print(json.dumps(out))
