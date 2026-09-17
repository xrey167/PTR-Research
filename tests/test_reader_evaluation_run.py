"""Execute the reader evaluator on an actual tiny local model/tokenizer bundle."""
import json
from pathlib import Path
import shutil
import sys

from safetensors.torch import load_file
from tokenizers.pre_tokenizers import ByteLevel
from transformers import Qwen2Tokenizer, Qwen2Config, Qwen2ForCausalLM

from research.evaluate_reader import main
from research.train_reader import file_sha


def test_evaluator_generates_without_target_in_prompt(tmp_path, monkeypatch):
    model_path = tmp_path / 'model'
    vocab = {'[PAD]': 0, '[UNK]': 1, '[EOS]': 2,
             'SECRET_TARGET': 3, 'SECRET_ASSESSMENT': 4}
    vocab.update({character: i + 5 for i, character in enumerate(sorted(ByteLevel.alphabet()))})
    config = Qwen2Config(vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        eos_token_id=2, pad_token_id=0, bos_token_id=1)
    Qwen2ForCausalLM(config).save_pretrained(model_path)
    tokenizer = Qwen2Tokenizer(vocab=vocab, merges=[], pad_token='[PAD]',
        unk_token='[UNK]', eos_token='[EOS]', additional_special_tokens=['SECRET_TARGET', 'SECRET_ASSESSMENT'])
    assert tokenizer.encode('SECRET_TARGET', add_special_tokens=False) == [3]
    assert tokenizer.encode('SECRET_ASSESSMENT', add_special_tokens=False) == [4]
    tokenizer.chat_template = "{% for message in messages %}{{message['role']}}: {{message['content']}}\n{% endfor %}assistant: "
    tokenizer.save_pretrained(model_path)
    bundle = tmp_path / 'bundle'
    inputs = bundle / 'inputs'
    inputs.mkdir(parents=True)
    row = {'id': 'heldout_case', 'evidence': None, 'history': [],
           'question': 'What is the lead time?', 'target': 'SECRET_TARGET',
           'assessment': 'SECRET_ASSESSMENT'}
    for split in ('train', 'dev', 'test'):
        (inputs / (split + '.json')).write_text(json.dumps([row]), encoding='utf-8')
    candidate = json.loads(Path('research/reader-lora-candidate.json').read_text(encoding='utf-8'))
    (inputs / 'config.json').write_text(json.dumps(candidate), encoding='utf-8')
    manifest = {'files': {p.name: {'sha256': file_sha(p)} for p in model_path.iterdir()}}
    (inputs / 'model-manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    snapshot = bundle / 'source_snapshot' / 'research'
    snapshot.mkdir(parents=True)
    for name in ('reader_prompt.py', 'reader_training.py', 'reader_identity.py'):
        shutil.copyfile(Path('research') / name, snapshot / name)
    protocol = {'input_hashes': {p.name: file_sha(p) for p in inputs.iterdir()},
        'source_hashes': {'research/' + p.name: file_sha(p) for p in snapshot.iterdir()},
        'optimizer_updates': 2, 'rows': {'train': 1}, 'model_path': str(model_path),
        'evaluation_max_new_tokens': 2}
    (bundle / 'protocol.json').write_text(json.dumps(protocol), encoding='utf-8')
    output = tmp_path / 'evaluation'
    monkeypatch.setattr(sys, 'argv', ['evaluate_reader.py', '--inputs', str(bundle),
                                     '--split', 'test', '--output', str(output)])
    main()
    report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
    assert report['status'] == 'completed'
    assert report['reader_unchanged']
    assert report['total'] == 1
    actual = report['rows'][0]
    assert 3 not in actual['prefix_ids'] + actual['suffix_ids']
    assert 4 not in actual['prefix_ids'] + actual['suffix_ids']
    assert 1 <= len(actual['tokens']) <= 2
    first_logits = load_file(output / 'first-logits.safetensors')['heldout_case']
    assert first_logits.numel() == len(vocab)
    assert int(first_logits.argmax()) == actual['tokens'][0]
    assert file_sha(output / 'first-logits.safetensors') == report['first_logits_sha256']
    assert report['exact_target_matches'] == int(actual['text'] == 'SECRET_TARGET')
