from nbl.dataloader_key_only import NBLFullDGLDataset

from utils.explain_utils import entropy_loss_mask, consistency_loss, size_loss
from utils.explain_utils import map_explain_with_nx
from utils.explain_utils import WrapperModel
from graph_algos.nx_shortcuts import nodes_where

from model import GCN_A_L_T_1
from utils.draw_utils import ast_to_agraph

import torch.nn.functional as F
import torch.nn as nn
import torch
import tqdm
import math
import copy
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def explain(model, dataloader, iters=10, epsilon=5e-1):

    lr = 5e-3
    bar = range(len(dataloader))
    for i in bar:

        g = dataloader[i]
        nx_g_id = dataloader.active_idxs[i]
        nx_g = dataloader.nx_dataset[nx_g_id][0]
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
        # wrapper(g)
        # # exit()
        opt = torch.optim.AdamW(wrapper.hgraph_weights.parameters(), lr)

        with torch.no_grad():
            ori_logits = wrapper.forward_old(g)
            _, ori_preds = torch.max(ori_logits, dim=1)
            # print(len(nodes_where(nx_g, graph='ast')))
            # exit()

        for nidx in nidxs:
            if nidx == 0:
                continue

            if ori_preds[nidx] == 0:
                continue

                
            print(num_edges_dict)

            gi = copy.deepcopy(g)
            nx_gi = copy.deepcopy(nx_g)

            titers = tqdm.tqdm(range(iters))
            titers.set_description(f'Graph {i}, Node {nidx}')
            for _ in titers:
                preds = wrapper(gi)
                # preds1 = preds[mask_stmt].detach().cpu()

                loss_e = entropy_loss_mask(gi, etypes)
                loss_c = 3 * consistency_loss(preds[nidx].unsqueeze(0), ori_preds[nidx].unsqueeze(0))
                loss_s = 7 * size_loss(gi, etypes) * epsilon

                loss = loss_e + loss_c + loss_s

                titers.set_postfix(loss_e=loss_e.item(), loss_c=loss_c.item(), loss_s=loss_s.item())

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    wrapper.hgraph_weights.parameters(), 1.0)
                opt.step()

            visualized_nx_g = map_explain_with_nx(gi, nx_gi)
            n_asts = nodes_where(visualized_nx_g, graph='ast')
            visualized_ast = nx_gi.subgraph(n_asts)

            # color = red
            visualized_ast.nodes[n_asts[nidx]]['status'] = 4

            os.makedirs(f'visualize_ast_explained/nbl/node_level', exist_ok=True)
            ast_to_agraph(visualized_ast,
                          f'visualize_ast_explained/nbl/node_level/Graph{i}_Node{nidx}_{epsilon}.png')


if __name__ == '__main__':
    dataset = NBLFullDGLDataset()
    meta_graph = dataset.meta_graph

    model = GCN_A_L_T_1(
        128, meta_graph,
        device=device,
        num_ast_labels=len(dataset.nx_dataset.ast_types),
        num_classes_ast=3)

    model.load_state_dict(torch.load('trained/nbl/Nov-29-2021/model_3_bestf1.pth', map_location=device))
    explain(model, dataset, iters=5000, epsilon=0.5)
