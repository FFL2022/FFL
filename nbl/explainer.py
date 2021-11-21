from nbl.dataloader_gumtree import NBLGumtreeDGLStatementDataset
from utils.explain_utils import map_explain_with_nx
from model import GCN_A_L_T_1
from utils.draw_utils import ast_to_agraph

import torch.nn.functional as F
import torch.nn as nn
import torch
import tqdm
import math
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class EdgeWeights(nn.Module):
    def __init__(self, num_nodes, num_edges, etype):
        super().__init__()

        self.num_edges = num_edges
        self.params = nn.Parameter(torch.FloatTensor(self.num_edges)).unsqueeze(-1)
        self.sigmoid = nn.Sigmoid()

        nn.init.normal_(self.params,
            nn.init.calculate_gain("relu")*math.sqrt(2.0)/(num_nodes*2))

        self.etype = etype

    def forward(self, g):
        g.edges[self.etype].data['weight'] = self.sigmoid(self.params)
        return g

class NodeWeights(nn.Module):

    def __init__(self, num_nodes, num_node_feats):
        super().__init__()
        self.num_nodes = num_nodes
        self.params = nn.Parameter(
            torch.FloatTensor(self.num_nodes, num_node_feats))
        nn.init.normal_(self.params, nn.init.calculate_gain(
            "relu")*math.sqrt(2.0)/(num_nodes*2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, g):
        g.nodes['ast'].data['weight'] = self.sigmoid(self.params)
        return g

class HeteroGraphWeights(nn.Module):

    def __init__(self, num_nodes, num_edges_dict, num_node_feats):
        super().__init__()
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
        super().__init__()
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

        return self.model(g).nodes['ast'].data['logits']


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
    loss = F.cross_entropy(preds, labels)
    return loss



def size_loss(g, etypes, coeff_n=0.005, coeff_e=0.005):
    feat_size_loss = coeff_n * torch.sum(g.nodes['ast'].data['weight'])
    edge_size_loss = coeff_e * torch.tensor([g.edges[_].data['weight'].sum()
        for _ in etypes]).sum()
    return feat_size_loss + edge_size_loss



def explain(model, dataloader, iters=10):

    lr = 3e-3
    os.makedirs('explain_log', exist_ok=True)

    # bar = tqdm.trange(len(dataloader))
    bar = range(len(dataloader))
    for i in bar:

        g, mask_stmt = dataloader[i]
        nx_g_id = dataloader.active_idxs[i]
        nx_g = dataloader.nx_dataset[nx_g_id][0]
        if g is None:
            continue
        g = g.to(device)
        mask_stmt = mask_stmt.to(device)

        num_edges_dict = {}
        for etype in g.etypes:
            if g.number_of_edges(etype) > 0:
                num_edges_dict[etype] = g.number_of_edges(etype)
        print(num_edges_dict, g.etypes)
        etypes = list(num_edges_dict.keys())

        wrapper = WrapperModel(model,
                               g.number_of_nodes(),
                               num_edges_dict,
                               model.hidden_feats).to(device)
        wrapper.hgraph_weights.train()
        # wrapper(g)
        # exit()
        opt = torch.optim.Adam(wrapper.hgraph_weights.parameters(), lr)

        with torch.no_grad():
            ori_logits = wrapper.forward_old(g)
            _, ori_preds = torch.max(ori_logits[mask_stmt].detach().cpu(), dim=1)

        titers = tqdm.tqdm(range(iters))
        titers.set_description(f'Graph {i}')
        # titers = range(iters)
        for it in titers:
            preds = wrapper(g)
            preds = preds[mask_stmt].detach().cpu()

            loss_e = entropy_loss_mask(g, etypes)
            loss_c = consistency_loss(preds, ori_preds.squeeze(-1))
            loss_s = size_loss(g, etypes) * 5e-3

            loss = loss_e + loss_c + loss_s

            titers.set_postfix(loss_e=loss_e.item(), loss_c=loss_c.item(), loss_s=loss_s.item())

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                wrapper.hgraph_weights.parameters(), 1.0)
            opt.step()
        visualized_nx_g = map_explain_with_nx(g, nx_g)
        # for n in visualized_nx_g:
        #     print(n)
        # exit()
        # Visualizing only ast:
        n_asts = [n for n in visualized_nx_g if
                  visualized_nx_g.nodes[n]['graph'] == 'ast']
        visualized_ast = nx_g.subgraph(n_asts)
        os.makedirs(f'visualize_ast_explained/nbl', exist_ok=True)
        ast_to_agraph(visualized_ast,
                      f'visualize_ast_explained/nbl/graph_{i}.png')


if __name__ == '__main__':
    dataset = NBLGumtreeDGLStatementDataset()
    meta_graph = dataset.meta_graph

    model = GCN_A_L_T_1(
        128, meta_graph,
        device=device,
        num_ast_labels=len(dataset.nx_dataset.ast_types),
        num_classes_ast=2)

    model.load_state_dict(torch.load('model_last.pth', map_location=device))
    explain(model, dataset, iters=5000)
