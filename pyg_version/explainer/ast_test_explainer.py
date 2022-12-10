from pyg_version.explainer.explainer import Explainer
from utils.iterators import UniformRandomIterator
import torch
import torch.nn.functional as F
from pyg_version.codeflaws.dataloader_cfl_pyg import CodeflawsCFLPyGStatementDataset
from typing import Sequence
import random


# Loop
class StatementNodeIterator(Sequence):
    def __init__(self, data, stmt_nodes):
        self.stmt_nodes = stmt_nodes

    def __len__(self):
        return len(self.stmt_nodes)

    def __getitem__(self, i):
        return self.stmt_nodes[i]


class RandomStatementGraphIterator(object):
    def __init__(self, data, stmt_nodes, lower, upper):
        self.stmt_nodes = stmt_nodes
        self.uniform_rand_iter = UniformRandomIterator(len(self.stmt_nodes), lower, upper)
        self.data = data

    def __iter__(self):
        self.iter = self.uniform_rand_iter
        return self

    def __next__(self):
        return self.data, self.stmt_nodes[next(self.iter)]


class RandomStatementDatasetIterator(object):
    def __init__(self, dataloader: Sequence, lower: float, upper: float):
        self.dataloader = dataloader
        self.lower, self.upper = lower, upper

    def __iter__(self):
        self.i = 0
        self.curr = None
        return self

    def __next__(self):
        if self.curr is None:
            if self.i > len(self.dataloader):
                raise StopIteration
            self.curr = RandomStatementGraphIterator(self.dataloader[self.i][0],
                                                     self.dataloader[self.i][1],
                                                     self.lower, self.upper)
        try:
            n = next(self.curr)
            return n
        except StopIteration:
            self.i += 1
            return next(self)


class AstTestWeight(object):
    def __init__(self):
        pass


def ast_consistency_loss(ast_pred_ori, ast_pred_perturb, all_weights, instance: tuple[_, int]):
    _, target_node = instance
    closs = F.cross_entropy(torch.tensor([ast_pred_perturb[target_node]]),
                            torch.tensor([torch.max(ast_pred_ori, dim=1)[target_node]]).long())
    # also has to
    return closs


class AstTestExplainer(Explainer):
    def __init__(self, model, save_dir, epochs):
        super().__init__(model, save_dir, epochs)
