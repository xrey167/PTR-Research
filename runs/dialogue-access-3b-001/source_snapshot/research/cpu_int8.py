"""Layerwise PyTorch dynamic Int8 conversion for the local CPU experiment.

No gradients. Embeddings remain BF16 in storage and return FP32 activations;
Linear operations use PyTorch's dynamic Int8 CPU kernels. This is a separately
reported numerical variant, not bit-equivalent to the original BF16 model.
"""
import torch
from torch.ao.nn.quantized.dynamic import Linear as Int8Linear
from torch.ao.quantization import default_dynamic_qconfig


class FloatOutputEmbedding(torch.nn.Embedding):
    def forward(self, ids):
        return super().forward(ids).float()


def quantize_in_place(model, progress=None):
    if 'onednn' in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = 'onednn'
    elif 'x86' in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = 'x86'
    elif 'fbgemm' in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = 'fbgemm'
    else:
        raise RuntimeError('No supported x86 dynamic Int8 backend')
    stats = {'schema':'torch-layerwise-dynamic-int8:v1', 'linear_layers':0,
             'quantized_weight_elements':0, 'embedding_storage':'bfloat16',
             'activations':'float32', 'engine':torch.backends.quantized.engine}
    def convert(parent):
        for name, child in list(parent.named_children()):
            if type(child) is torch.nn.Linear:
                if child.bias is not None:
                    child.bias = torch.nn.Parameter(child.bias.detach().float(), requires_grad=False)
                child.qconfig = default_dynamic_qconfig
                quantized = Int8Linear.from_float(child)
                setattr(parent, name, quantized)
                stats['linear_layers'] += 1
                stats['quantized_weight_elements'] += child.in_features * child.out_features
                if progress is not None and stats['linear_layers'] % 24 == 0:
                    progress(dict(stats))
            elif isinstance(child, torch.nn.Embedding):
                replacement = FloatOutputEmbedding(child.num_embeddings, child.embedding_dim,
                    padding_idx=child.padding_idx, _weight=child.weight.detach().to(torch.bfloat16), _freeze=True)
                setattr(parent, name, replacement)
            else:
                convert(child)
                # Small remaining norm parameters compute with float activations.
                for key, parameter in list(child.named_parameters(recurse=False)):
                    if parameter.is_floating_point():
                        setattr(child, key, torch.nn.Parameter(parameter.detach().float(), requires_grad=False))
    convert(model)
    model.eval().requires_grad_(False)
    return stats
