"""
Mini-batches straight from tensors.

`DataLoader(TensorDataset(...))` indexes the dataset one sample at a time and
collates the results, which costs about a fifth of a soft tree's training
time for nothing: the tensors are already in memory. This iterator draws the
same permutation `RandomSampler` would draw from the same generator, so with a
`random_state` the batches, and therefore the fitted parameters, are
bit-identical to the loader's; it just slices instead of gathering.
"""
from typing import Optional, Sequence

import torch


class _Tensors:
    def __init__(self, tensors):
        self.tensors = tuple(tensors)


class TensorBatches:
    """Shuffled mini-batches over aligned tensors; `.dataset.tensors` as a loader has."""

    def __init__(self, tensors: Sequence[torch.Tensor], batch_size: int,
                 generator: Optional[torch.Generator] = None):
        self.dataset = _Tensors(tensors)
        self.n = int(tensors[0].shape[0])
        self.batch_size = max(1, min(int(batch_size), self.n))
        self.generator = generator

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.generator is None:
            perm = torch.randperm(self.n)
        else:
            # DataLoader draws one int64 from the generator when it builds its
            # iterator (its _base_seed) before RandomSampler draws the
            # permutation. Drawing it here too keeps the batches, and so the
            # fitted parameters, bit-identical to the loader's for a given seed.
            torch.empty((), dtype=torch.int64).random_(generator=self.generator)
            perm = torch.randperm(self.n, generator=self.generator)
        perm = perm.to(self.dataset.tensors[0].device)
        for start in range(0, self.n, self.batch_size):
            idx = perm[start:start + self.batch_size]
            yield tuple(t[idx] for t in self.dataset.tensors)
        if self.generator is not None:
            # RandomSampler's generator function, once its permutation is used
            # up, draws a second permutation for the (empty) remainder before
            # it stops. A fully consumed loader therefore advances the
            # generator by one more randperm per epoch; match it.
            torch.randperm(self.n, generator=self.generator)
