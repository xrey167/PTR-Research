"""Identity for a future LoRA reader; not retrofitted to existing capsule runs."""
import hashlib
import json
from research.prefix_capsule import weights_hash


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(v) for v in value)
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f'Unsupported identity metadata type: {type(value).__name__}')


def reader_identity(model):
    """Bind weights and effective LoRA routing/scaling for this local reader.

    This is not an execution-environment attestation or cross-backend ABI proof.
    """
    adapters = getattr(model, 'peft_config', {})
    layers = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'scaling'):
            layers[name] = {'scaling': dict(module.scaling),
                            'active': list(module.active_adapters),
                            'disabled': bool(module.disable_adapters),
                            'merged': list(module.merged_adapters)}
    config = model.config.to_dict()
    config.pop('_name_or_path', None)
    config.pop('transformers_version', None)
    payload = {'schema': 'research-reader-identity:v1', 'weights_sha256': weights_hash(model),
               'model_config': config,
               'attention_implementation': getattr(model.config, '_attn_implementation', None),
               'adapter_configs': {k: v.to_dict() for k, v in adapters.items()},
               'lora_runtime': layers}
    payload = _plain(payload)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return {'sha256': hashlib.sha256(encoded.encode('utf-8')).hexdigest(), 'payload': payload}
