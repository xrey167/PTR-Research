"""Actual autograd equivalence, with unequal masked answer lengths."""
import copy
import weakref
from types import SimpleNamespace
import pytest
import torch
from research.reader_training import token_normalized_backward


class ToyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(11, 5, dtype=torch.float64)
        self.head = torch.nn.Linear(5, 11, dtype=torch.float64)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


@pytest.mark.parametrize('count', [3, 2])
def test_accumulation_matches_full_batch_for_unequal_answers(count):
    torch.manual_seed(13)
    accumulated = ToyDecoder()
    full = copy.deepcopy(accumulated)
    ids = torch.tensor([[1, 2, 3, 4, 0], [1, 3, 5, 7, 9], [2, 4, 6, 8, 0]])[:count]
    labels = torch.tensor([[-100, -100, -100, 4, -100],
                           [-100, 3, 5, 7, 9],
                           [-100, -100, 6, 8, -100]])[:count]
    metrics = token_normalized_backward(accumulated,
        [{'input_ids': ids[i:i+1], 'labels': labels[i:i+1]} for i in range(count)])
    logits = full(ids).logits[:, :-1, :]
    reference_loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 11),
        labels[:, 1:].reshape(-1), ignore_index=-100, reduction='mean')
    reference_loss.backward()
    assert metrics['supervised_tokens'] == (7 if count == 3 else 5)
    assert metrics['loss'] == pytest.approx(float(reference_loss.detach()), abs=1e-12)
    for actual, expected in zip(accumulated.parameters(), full.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-12, atol=1e-12)


def test_empty_supervision_rejected_before_forward():
    model = ToyDecoder()
    with pytest.raises(ValueError, match='no supervised'):
        token_normalized_backward(model, [{'input_ids': torch.tensor([[1, 2]]),
                                          'labels': torch.tensor([[-100, -100]])}])
    assert all(p.grad is None for p in model.parameters())


def test_previous_logits_released_before_next_microbatch():
    class LifetimeDecoder(ToyDecoder):
        previous = None

        def forward(self, *args, **kwargs):
            if self.previous is not None:
                assert self.previous() is None, 'Previous output still retained at next forward'
            output = super().forward(*args, **kwargs)
            self.previous = weakref.ref(output.logits)
            return output

    model = LifetimeDecoder()
    batch = {'input_ids': torch.tensor([[1, 2, 3]]),
             'labels': torch.tensor([[-100, 2, 3]])}
    token_normalized_backward(model, [batch, batch])
    assert model.previous() is None
