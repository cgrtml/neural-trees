"""TensorBatches must reproduce DataLoader's batches exactly for a given seed."""
import torch
from torch.utils.data import DataLoader, TensorDataset

from neural_trees._batching import TensorBatches


def _tensors(n=23):
    return torch.arange(n * 2.0).reshape(n, 2), torch.arange(n), torch.ones(n)


def test_same_batches_as_dataloader_for_a_seeded_generator():
    X, y, w = _tensors()
    dl = DataLoader(TensorDataset(X, y, w), batch_size=5, shuffle=True,
                    generator=torch.Generator().manual_seed(7))
    tb = TensorBatches((X, y, w), 5, torch.Generator().manual_seed(7))
    for _ in range(4):                       # several epochs, fully consumed
        a = [yb.tolist() for _, yb, _ in dl]
        b = [yb.tolist() for _, yb, _ in tb]
        assert a == b


def test_covers_every_sample_once_per_epoch():
    X, y, w = _tensors(50)
    tb = TensorBatches((X, y, w), 8, None)
    seen = sorted(i for _, yb, _ in tb for i in yb.tolist())
    assert seen == list(range(50)) and len(tb) == 7


def test_exposes_the_tensors_like_a_loader():
    X, y, w = _tensors()
    tb = TensorBatches((X, y, w), 4, None)
    assert tb.dataset.tensors[2] is w
    assert tb.batch_size == 4 and TensorBatches((X, y, w), 999, None).batch_size == 23
