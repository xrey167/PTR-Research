import importlib.util, torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
for name in ('vllm','causal_conv1d','fla','flash_linear_attention','sglang'):
    print(name, bool(importlib.util.find_spec(name)))
