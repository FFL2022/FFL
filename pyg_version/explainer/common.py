import torch
import torch.nn as nn
import math
import networkx as nx
import os
import tqdm
import pickle as pkl
from typing import List, Tuple, Union


device = torch.device('cuda'
                      if torch.cuda.is_available() else 'cpu')

class EdgeWeights(nn.Module):
    def __init__(self, num_nodes, num_edges, init='random'):
        super().__init__()
        self.num_edges = num_edges
        self.params = nn.Parameter(
            torch.FloatTensor(self.num_edges).unsqueeze(-1))
        self.sigmoid = nn.Sigmoid()
        self.num_nodes = num_nodes
        if init == 'random':
            nn.init.normal_(
                self.params,
                nn.init.calculate_gain("relu")*math.sqrt(2.0)/(num_nodes*2))
        else:
            nn.init.constant_(self.params, 7.0)

    def forward(self):
        return self.sigmoid(self.params)


class NodeFeatureWeights(nn.Module):
    def __init__(self, num_nodes, init='random'):
        super().__init__()
        self.num_nodes = num_nodes
        self.params = nn.Parameter(
            torch.FloatTensor(self.num_nodes, 1))
        if init == 'random':
            nn.init.normal_(self.params, nn.init.calculate_gain(
                "relu")*math.sqrt(2.0)/(num_nodes*2))
        else:
            nn.init.constant_(self.params, 7.0)
        self.sigmoid = nn.Sigmoid()

    def forward(self):
        return self.sigmoid(self.params.to(device))


class EdgeWeightsBalanced(nn.Module):
    def __init__(self, num_nodes, num_edges, init='random'):
        super().__init__()
        self.num_edges = num_edges
        self.params = nn.Parameter(
            torch.FloatTensor(self.num_edges).unsqueeze(-1))
        self.sigmoid = nn.Sigmoid()
        self.num_nodes = num_nodes
        if init == 'random':
            nn.init.normal_(
                self.params,
                nn.init.calculate_gain("relu")*math.sqrt(2.0)/(num_nodes*2))
        else:
            nn.init.constant_(self.params, 7.0)

    def forward(self):
        return self.sigmoid(self.params)*2 - 1


class NodeFeatureWeightsBalanced(nn.Module):
    def __init__(self, num_nodes, init='random'):
        super().__init__()
        self.num_nodes = num_nodes
        self.params = nn.Parameter(
            torch.FloatTensor(self.num_nodes, 1))
        if init == 'random':
            nn.init.normal_(self.params, nn.init.calculate_gain(
                "relu")*math.sqrt(2.0)/(num_nodes*2))
        else:
            nn.init.constant_(self.params, 7.0)
        self.sigmoid = nn.Sigmoid()

    def forward(self):
        return self.sigmoid(self.params.to(device)) * 2 - 1


class AllWeightsSingleGraph(nn.Module):
    def __init__(self, num_nodes, num_edges, init='random'):
        super().__init__()
        self.nweights = NodeFeatureWeights(num_nodes, init)
        self.eweights = EdgeWeights(num_nodes, num_edges, init)

    def forward(self, *feats):
        nweights = self.nweights()
        eweights = self.eweights()
        return [nweights*f for f in feats] + [eweights]


class AllWeightsSingleGraphBalanced(nn.Module):
    def __init__(self, num_nodes, num_edges, init='random'):
        super().__init__()
        self.nweights = NodeFeatureWeightsBalanced(num_nodes, init)
        self.eweights = EdgeWeightsBalanced(num_nodes, num_edges, init)

    def forward(self, *feats):
        nweights = self.nweights()
        eweights = self.eweights()
        return [nweights*f for f in feats] + [eweights]

def entropy_loss(masking):
    return torch.mean(
        -torch.sigmoid(masking) * torch.log(torch.sigmoid(masking)) -
        (1 - torch.sigmoid(masking)) * torch.log(1 - torch.sigmoid(masking)))


def entropy_loss_mask(nweights, eweights, coeff_n=0.1, coeff_e=0.3):
    e_e_loss = 0
    e_e_loss += entropy_loss(eweights)
    e_e_loss = coeff_e * e_e_loss
    n_e_loss = coeff_n * entropy_loss(nweights)
    return n_e_loss + e_e_loss


def entropy_loss_mask_balanced(nweights, eweights, coeff_n=0.1, coeff_e=0.3):
    e_e_loss = 0
    e_e_loss += entropy_loss(eweights*0.5+0.5)
    e_e_loss = coeff_e * e_e_loss
    n_e_loss = coeff_n * entropy_loss(nweights*0.5+0.5)
    return n_e_loss + e_e_loss



def entropy_loss_wrapper_single(wrapper):
    return entropy_loss_mask(wrapper.a_weight.nweights(),
                             wrapper.a_weight.eweights())


def size_loss(nweights, eweights, coeff_n=0.002, coeff_e=0.005):
    feat_size_loss = coeff_n * torch.sum(nweights)
    edge_size_loss = 0
    edge_size_loss += eweights.sum()
    edge_size_loss = coeff_e * edge_size_loss
    return feat_size_loss + edge_size_loss


def size_loss_balanced(nweights, eweights, coeff_n=0.002, coeff_e=0.005):
    feat_size_loss = coeff_n * torch.sum(torch.abs(nweights))
    edge_size_loss = 0
    edge_size_loss += torch.abs(eweights).sum()
    edge_size_loss = coeff_e * edge_size_loss
    return feat_size_loss + edge_size_loss



def size_loss_wrapper_single(wrapper):
    return size_loss(wrapper.a_weight.nweights(),
                     wrapper.a_weight.eweights())


def get_specific_output(model, args, output_idxs, output_maps,
                        output_names=None):
    '''Get specific output out of all modesl output and map to a dict
    output_idxs: indexing among the output
    '''

    if output_names is None:
        output_names = output_idxs
    with torch.no_grad():
        os = model(*args)
        return {nm: mfunc(os[i]) for i, mfunc, nm in zip(
            output_idxs, output_maps, output_names)}


def extract_infl_substruct_n_e(nx_g, target_node, ns, es, nweights, eweights,
                              thres_e=0.3, thres_n=0.5, target_eidx=None):
    print(torch.max(nweights), torch.max(eweights))
    # print(nweights, eweights)
    # Extract all nodes and edges that have high influences
    if target_eidx is not None:
        target_edge = (es[0, target_eidx].item(), es[1, target_eidx].item())
    # print(torch.max(nweights), torch.max(eweights))
    es = es[:, (eweights >= thres_e)[:, 0]]
    edges = [(u.item(), v.item()) for u, v in zip(es[0, :], es[1, :])]
    if target_eidx is not None:
        if target_edge not in edges:
            edges.append(target_edge)
    ns = ns[(nweights >= thres_n)[:, 0]]
    kept_n = set(es[0, :].cpu().tolist()).union(
        set(es[1, :].cpu().tolist())).union(set(ns))
    if target_eidx is not None:
        kept_n = kept_n.union(
            set([target_edge[0], target_edge[1]]))
    kept_n.add(target_node)
    kept_n.remove(target_node)
    # rename nodes
    mapping = dict((n, i) for i, n in enumerate(sorted(kept_n)))
    mapping[target_node] = 'y'
    kept_n.add(target_node)
    substruct = nx_g.subgraph(kept_n).copy()
    for u, v in list(substruct.edges()):
        if (u, v) not in edges:
            substruct.remove_edge(u, v)
    if target_eidx is not None:
        substruct.edges[target_edge[0], target_edge[1], 0]['is_target'] = 1
    # print("before: ", substruct.nodes(), target_node)
    substruct = nx.relabel_nodes(substruct, mapping)
    for n in substruct.nodes():
        substruct.nodes[n]['is_target'] = 0

    for u, v, k, e in substruct.edges(keys=True, data=True):
        if 'is_target' not in e:
            e['is_target'] = 0
    # print(substruct.edges())
    substruct.nodes['y']['is_target'] = 1
    return substruct


def extract_infl_substruct_n(nx_g, target_node, ns, es, nweights, eweights,
                              thres_e=0.3, thres_n=0.5):
    # Extract all nodes and edges that have high influences
    es = es[:, (eweights >= thres_e)[:, 0]]
    edges = [(u, v) for u, v in zip(es[0, :], es[1, :])]
    ns = ns[(nweights >= thres_n)[:, 0]]
    kept_n = set(es[0, :].cpu().tolist()).union(
        set(es[1, :].cpu().tolist())).union(set(ns))
    kept_n.add(target_node)
    kept_n.remove(target_node)
    # rename nodes
    mapping = dict((n, i) for i, n in enumerate(sorted(kept_n)))
    mapping[target_node] = 'y'
    kept_n.add(target_node)
    substruct = nx_g.subgraph(kept_n).copy()
    for u, v in list(substruct.edges()):
        if (u, v) not in edges:
            substruct.remove_edge(u, v)
    substruct = nx.relabel_nodes(substruct, mapping)
    for n in substruct.nodes():
        substruct.nodes[n]['is_target'] = 0
    substruct.nodes['y']['is_target'] = 1
    return substruct


class Explainer(object):
    ## TODO: Document
    def __init__(self, DataAdapter, ModelMetadata, save_dir="."):
        self.save_dir = save_dir
        self.da = DataAdapter()
        self.mmd = ModelMetadata()

    def make_savedir(self):
        """Creating save directory"""
        if self.save_dir:
            save_dir = self.save_dir
            os.makedirs(f'{save_dir}/explain_log', exist_ok=True)
            os.makedirs(f'{save_dir}/substructure', exist_ok=True)
            os.makedirs(f'{save_dir}/substructure/images', exist_ok=True)

    def preprocess_data(self, data):
        """Preproccess the data before anything"""
        if isinstance(data, tuple) or isinstance(data, list):
            return tuple(e.to(device) for e in data)
        return data.to(device)

    def get_ori_pred_data(self, model, data):
        """Get the targeted original prediction"""
        raise NotImplementedError

    def explain_has_cache(self, i, target: Union[Tuple[int], List[int]]):
        """Check if the explaination has cache"""
        s = '_'.join([str(t) for t in target])
        if os.path.exists(os.path.join(self.save_dir, f"explain_log/g{i}_{s}")):
            return True

    def get_ne(self, data):
        """Get number of edges from data"""
        raise NotImplementedError

    def init_optimizer(self, wrapper):
        """Optimizer for explanation process"""
        return torch.optim.AdamW(wrapper.a_weight.parameters(), 1e-2)

    def closs_func(self, ori_pred_data, wrapper_out, target):
        """Definition of consistency loss function w.r.t to original and
        masked output"""
        raise NotImplementedError


    def data2nx(self, data, ori_pred_data, wrapper) -> nx.MultiDiGraph:
        """Convert from data sample in the data wrapper to networkx for dumpling substruct"""
        raise NotImplementedError

    def extract_infl_substruct(self, nx_g, data, target, wrapper) -> nx.MultiDiGraph:
        """Extract influential substructure"""
        raise NotImplementedError

    def dump_substructure(self, substruct, outs,
                          i:int, target: Union[List[int],
                          Tuple[int]]):
        s = '_'.join([str(t) for t in target])
        pkl.dump((substruct, outs),
                open(os.path.join(self.save_dir,
                                  f'substructure/{i}_{s}.pkl'), 'wb'))

    def draw_substruct(self, substruct: nx.MultiDiGraph, save_path: str):
        """Draw substruct and save to savepath"""
        raise NotImplementedError

    def get_target_iter(self, data):
        """Iterate through wanted component of the target data sample,
        e.g., nodes or edges"""
        raise not NotImplementedError

    def get_wrapper(self, model, data) -> nn.Module:
        raise NotImplementedError

    def get_size_loss(self, wrapper):
        return size_loss_wrapper_single(wrapper) * 3

    def get_entropy_loss(self, wrapper):
        return entropy_loss_wrapper_single(wrapper) * 7

    def explain(self, model, dataloader, opt_iters=5000,
                start_idx=None, end_idx=None):
        self.make_savedir()
        for i, data in enumerate(dataloader):
            if start_idx is not None and i < start_idx:
                continue
            if end_idx is not None and i > end_idx:
                break
            self.preprocess_data(data)
            ori_pred_data = self.get_ori_pred_data(model, data)
            target_iter = self.get_target_iter(data)
            model_args = self.da(data)
            for t in target_iter:
                if self.explain_has_cache(i, t):
                    continue
                wrapper = self.get_wrapper(model, data).to(device)
                opt = self.init_optimizer(wrapper)
                titers = tqdm.tqdm(range(opt_iters))
                titers.set_description(f'Instance {i} {t}')
                for _ in titers:
                    outs = wrapper(*model_args)

                    loss_e = self.get_size_loss(wrapper)
                    loss_c = self.closs_func(ori_pred_data, outs, t)
                    loss_s = self.get_entropy_loss(wrapper)
                    loss = loss_e + loss_c + loss_s
                    titers.set_postfix(loss_e=loss_e.item(),
                                   loss_c=loss_c.item(),
                                   loss_s=loss_s.item(),
                                   )
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        wrapper.a_weight.parameters(), 1.0)
                    opt.step()
                nx_g = self.data2nx(data, ori_pred_data, wrapper)
                substruct = self.extract_infl_substruct(nx_g, data, t, wrapper)
                self.dump_substructure(substruct, outs, i, t)
                s = '_'.join([str(te) for te in t])
                print(
                    os.path.join(
                        self.save_dir, f'substructure/images/{i}_{s}.png'))
                self.draw_substruct(
                    substruct,
                    os.path.join(self.save_dir, f'substructure/images/{i}_{s}.png'))
