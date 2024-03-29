from codeflaws.dataloader_gumtree_node import CodeflawsGumtreeDGLNodeDataset
from utils.explain_utils import map_explain_with_nx
from dgl_version.model import GCN_A_L_T_1
from utils.draw_utils import ast_to_agraph

import torch.nn.functional as F
import torch.nn as nn
import torch
import tqdm
import math
import copy
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class EdgeWeights(nn.Module):
    def __init__(self, num_nodes, num_edges, etype):
        super(EdgeWeights, self).__init__()

        self.num_edges = num_edges
        self.params = nn.Parameter(torch.FloatTensor(self.num_edges).unsqueeze(-1).to(device))
        self.sigmoid = nn.Sigmoid()

        nn.init.normal_(self.params,
            nn.init.calculate_gain("relu")*math.sqrt(2.0)/(num_nodes*2))

        self.etype = etype

    def forward(self, g):
        g.edges[self.etype].data['weight'] = self.sigmoid(self.params)
        return g

class NodeWeights(nn.Module):

    def __init__(self, num_nodes, num_node_feats):
        super(NodeWeights, self).__init__()
        self.params = nn.Parameter(
            torch.FloatTensor(num_nodes, num_node_feats).to(device))
        nn.init.normal_(self.params, nn.init.calculate_gain(
            "relu")*math.sqrt(2.0)/(num_nodes*2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, g):
        g.nodes['ast'].data['weight'] = self.sigmoid(self.params)
        return g

class HeteroGraphWeights(nn.Module):

    def __init__(self, num_nodes, num_edges_dict, num_node_feats):
        super(HeteroGraphWeights, self).__init__()
        self.nweights = NodeWeights(num_nodes, num_node_feats)
        self.eweights = {}
        for etype in num_edges_dict:
            self.eweights[etype] = EdgeWeights(num_nodes, num_edges_dict[etype], etype)

        self.etypes = list(num_edges_dict.keys())

    def forward(self, g):
        g = self.nweights(g)
        for etype in self.etypes:
            g = self.eweights[etype](g)
        return g


class WrapperModel(nn.Module):
    def __init__(self, model, num_nodes, num_edges_dict, num_node_feats):
        super(WrapperModel, self).__init__()
        self.hgraph_weights = HeteroGraphWeights(num_nodes, num_edges_dict, num_node_feats)
        self.model = model

    def forward_old(self, g):
        self.model.add_default_nweight = True
        self.model.add_default_eweight = True
        self.model.eval()
        return self.model(g).nodes['ast'].data['logits']

    def forward(self, g):
        self.model.add_default_nweight = False
        self.model.add_default_eweight = False
        self.model.eval()

        g = self.hgraph_weights(g)

        return self.model(g).nodes['ast'].data['logits'].softmax(dim=1)


def entropy_loss(masking):
    return torch.mean(
        -torch.sigmoid(masking) * torch.log(torch.sigmoid(masking)) -
        (1 - torch.sigmoid(masking)) * torch.log(1 - torch.sigmoid(masking)))


def entropy_loss_mask(g, etypes, coeff_n=0.2, coeff_e=0.5):
    e_e_loss = coeff_e * torch.tensor([entropy_loss(g.edges[_].data['weight'])
        for _ in etypes]).mean()
    n_e_loss = coeff_n * entropy_loss(g.nodes['ast'].data['weight'])
    return n_e_loss + e_e_loss

def consistency_loss(preds, labels):
    # print(preds, labels)
    # print(preds.shape, labels.shape)
    loss = F.cross_entropy(preds, labels)
    return loss



def size_loss(g, etypes, coeff_n=0.005, coeff_e=0.005):
    feat_size_loss = coeff_n * torch.sum(g.nodes['ast'].data['weight'])
    edge_size_loss = coeff_e * torch.tensor([g.edges[_].data['weight'].sum()
        for _ in etypes]).sum()
    return feat_size_loss + edge_size_loss



def explain(model, dataloader, iters=10):

    lr = 1e-3
    bar = range(len(dataloader))
    for i in bar:

        g = dataloader[i]
        nx_g_id = dataloader.active_idxs[i]
        nx_g = dataloader.nx_dataset[nx_g_id]

        if g is None:
            continue
        g = g.to(device)

        nidxs = g.nodes(ntype='ast')

        num_edges_dict = {}
        for etype in g.etypes:
            if g.number_of_edges(etype) > 0:
                num_edges_dict[etype] = g.number_of_edges(etype)
        etypes = list(num_edges_dict.keys())

        wrapper = WrapperModel(model,
                               len(nidxs),
                               num_edges_dict,
                               model.hidden_feats).to(device)
        wrapper.hgraph_weights.train()
        # print([p.shape for p in wrapper.hgraph_weights.parameters()])
        # exit()
        # wrapper(g)
        # # exit()
        opt = torch.optim.Adam(wrapper.hgraph_weights.parameters(), lr)

        with torch.no_grad():
            ori_logits = wrapper.forward_old(g)
            _, ori_preds = torch.max(ori_logits.detach().cpu(), dim=1)

        for nidx in nidxs:
            if ori_preds[nidx] == 0:
                continue
            print(num_edges_dict)

            gi = copy.deepcopy(g)
            nx_gi = copy.deepcopy(nx_g)

            titers = tqdm.tqdm(range(iters))
            titers.set_description(f'Graph {i}, Node {nidx}, Label {ori_preds[nidx]}')
            for _ in titers:
                preds = wrapper(gi).detach().cpu()
                # preds1 = preds[mask_stmt].detach().cpu()

                loss_e = entropy_loss_mask(gi, etypes)
                loss_c = consistency_loss(preds[nidx].unsqueeze(0), ori_preds[nidx].unsqueeze(0))
                loss_s = size_loss(gi, etypes) * 5e-2

                loss = loss_e + loss_c + loss_s

                titers.set_postfix(loss_e=loss_e.item(), loss_c=loss_c.item(), loss_s=loss_s.item())

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    wrapper.hgraph_weights.parameters(), 1.0)
                opt.step()

            visualized_nx_g = map_explain_with_nx(gi, nx_gi)
            n_asts = [n for n in visualized_nx_g if
                      visualized_nx_g.nodes[n]['graph'] == 'ast']
            visualized_ast = nx_gi.subgraph(n_asts)

            # color = red
            visualized_ast.nodes[n_asts[nidx]]['status'] = 4

            os.makedirs(f'visualize_ast_explained/codeflaws/node_level/{i}', exist_ok=True)
            ast_to_agraph(visualized_ast,
                          f'visualize_ast_explained/codeflaws/node_level/{i}/{nidx}.png')


if __name__ == '__main__':
    dataset = CodeflawsGumtreeDGLNodeDataset()
    meta_graph = dataset.meta_graph

    model = GCN_A_L_T_1(
        256, meta_graph,
        device=device,
        num_ast_labels=len(dataset.nx_dataset.ast_types),
        num_classes_ast=3)

    model.load_state_dict(torch.load('trained/codeflaws/Nov-28-2021/model_last_gumtree_node.pth', map_location=device))
    explain(model, dataset, iters=10000)
