import pytest
import torch
from research.audit_prefix_probe import validate_continuation


class Tokenizer:
    def decode(self, ids, skip_special_tokens):
        return ''.join({1:'18', 2:'24', 3:''}[i] for i in ids)


def test_valid_evidence():
    validate_continuation({'tokens':[1,3],'eos':True,'text':'18'}, torch.tensor([[0.,1.,0.,0.]]), Tokenizer(), {3})


@pytest.mark.parametrize('row', [
    {'tokens':[1,3],'eos':True,'text':'24'},
    {'tokens':[2,3],'eos':True,'text':'24'},
    {'tokens':[1,3],'eos':False,'text':'18'},
    {'tokens':[1,3,2],'eos':True,'text':'1824'},
    {'tokens':[-1,3],'eos':True,'text':'18'},
])
def test_forged_evidence_is_rejected(row):
    with pytest.raises(ValueError):
        validate_continuation(row, torch.tensor([[0.,1.,0.,0.]]), Tokenizer(), {3})
