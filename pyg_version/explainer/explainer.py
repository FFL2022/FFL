from pyg_version.explainer.common import EdgeWeights, NodeFeatureWeights, NodeFeatureWeightsBalanced
from graph_algos.nx_shortcuts import where_node, where_edge, \
        neighbors_in, update_nodes
from typing import Iterable, Tuple, List
import torch
import tqdm
import networkx as nx


def data_forward(model, data):
    if isinstance(data, tuple) or isinstance(data, list):
        return model(*data)
    elif isinstance(data, dict):
        return model(**data)
    else:
        return model(data)


class Explainer(object):
    def __init__(self, model, loss_func, epochs=5000):
        self.model = model
        self.loss = loss_func
        self.epochs = epochs

    def get_data(self, instance):
        raise NotImplementedError

    def get_perturber(self, data) -> torch.nn.Module:
        raise NotImplementedError

    def data_to_model(self, data):
        raise NotImplementedError

    def data_to_perturber(self, data):
        raise NotImplementedError

    def prepare(self, perturber: torch.nn.Module):
        self.opt = torch.optim.AdamW(perturber.parameters())

    def explain_instance(self, instance) -> torch.nn.Module:
        data = self.get_data(instance)
        perturber = self.get_perturber(data)
        self.prepare(perturber)
        orig_pred = self.model(*self.data_to_model(data))
        bar = tqdm.trange(self.epochs)
        for i in bar:
            perturbed_data = data_forward(perturber, self.data_to_perturber(data))
            perturbed_pred = data_forward(self.model, perturbed_data)
            loss = self.loss(perturbed_pred, orig_pred, perturber, instance)
            bar.set_postfix(loss=loss.item())
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return perturber

    def explain(self, instance_iterator) -> Iterable[torch.nn.Module]:
        for instance in instance_iterator:
            yield instance, self.explain_instance(instance)


class InflExtractor(object):
    def __init__(self, where_n, where_e, thres_n, thres_e):
        self.where_e = where_e
        self.where_n = where_n
        self.thres_n = thres_n
        self.thres_e = thres_e

    def extract_infl_structure(self, nx_g, target_node):
        es = [(u, v, k, e) for u, v, k, e in nx_g.edges()
              if e['explain_weight'] >= self.thres_e and
              self.where_e(nx_g, (u, v, k, e))]
        ns = list(n for n in nx_g.nodes() if self.where_n(nx_g, n) and
                  nx_g.nodes[n]['explain_weight'] >= self.thres_n)
        kept_n = set(ns) + set(sum([u, v] for u, v, _, _ in es))
        kept_n.add(target_node)
        kept_n.remove(target_node)
        n2i = {n: i for i, n in enumerate(kept_n)}
        n2i[target_node] = 'y'
        kept_n.add(target_node)
        substruct = nx_g.subgraph(kept_n).copy()
        for u, v in list(substruct.edges()):
            if (u, v) not in es:
                substruct.remove_edge(u, v)
        for n in list(substruct.nodes()):
            if not neighbors_in(n, nx_g) and n != target_node:
                substruct.remove_node(n)
        substruct = nx.relabel_nodes(substruct, n2i)
        update_nodes(substruct, is_target=0)
        substruct.nodes['y']['is_target'] = 1
        return substruct
