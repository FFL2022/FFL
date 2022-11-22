from typing import Tuple, List, Union
import random
from collections import Sequence
from itertools import accumulate


class NxDataset(Sequence):
    def __init__(self):
        if self.has_cache():
            self.load()
        else:
            self.process()
            self.save()

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def has_cache(self):
        raise NotImplementedError


class NxDataloader(Sequence):
    def __init__(self, nx_dataset: NxDataset, idxs):
        self.idxs = idxs
        self.nx_dataset = nx_dataset

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        return self.nx_dataset[self.idxs[i]]

    def get_dataset(self) -> NxDataset:
        return self.nx_dataset


def split_nx_dataset(nx_dataset: NxDataset,
                     ratio: Union[int, List[int]]=0.8,
                     shuffle=True) -> Tuple[NxDataset, NxDataset]:
    N = len(nx_dataset)
    idxs = list(range(N))
    if shuffle:
        random.shuffle(idxs)
    if isinstance(ratio, list):
        assert len(ratio), "Ratio must not be empty List"
        assert sum(ratio) == 1, "Sum of ratio must be 1"
        cumsum = list(accumulate(ratio))
        out = []
        for i, c in enumerate(cumsum):
            if i:
                out.append(NxDataloader(
                    nx_dataset, idxs[int(cumsum[i-1]*N):int(c*N)]))
            else:
                out.append(NxDataloader(nx_dataset, idxs[:int(c*N)]))
        return out
    return NxDataloader(nx_dataset, idxs[:int(ratio*N)]),\
        NxDataloader(nx_dataset, idxs[int(N*ratio):])
