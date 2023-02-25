from utils.meta_data_extractor import get_meta_data
from utils.numerize_graph import numerize_graph
from typing import List
import networkx as nx
import torch
import torch_geometric


class PytorchAdapter:

    def __init__(self, meta_data):
        self.meta_data = meta_data

    def get_pytorch_graph(self, graph):
        raise NotImplementedError


class PytorchExplainerConstructor:
    def __init__(self, model_class, meta_data):
        pass
    
# Building a PytorchAdapter automatically using a networkx dataset
class PytorchAdapterBuilder:

    def __init__(self, nx_gs: List[nx.Graph]):
        self.nx_gs = nx_gs
        self.meta_data = get_meta_data(self.nx_gs)

    def build(self, save_path):
        pass
