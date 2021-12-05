from nbl.dataloader_gumtree import NBLGumtreeDGLStatementDataset

from utils.explain_utils import entropy_loss_mask, consistency_loss, size_loss
from utils.explain_utils import map_explain_with_nx
from utils.explain_utils import WrapperModel

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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def explain(model, dataloader, iters=10):

    lr = 1e-3
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
        etypes = list(num_edges_dict.keys())

        wrapper = WrapperModel(model,
                               g.number_of_nodes(),
                               num_edges_dict,
                               model.hidden_feats).to(device)

        wrapper.model.eval()

        wrapper.hgraph_weights.train()
        # wrapper(g)

        # for name, param in wrapper.hgraph_weights.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.shape)


        # print([wrapper.hgraph_weights.nweights.parameters()] + [wrapper.hgraph_weights.eweights[_].parameters() for _ in etypes])
        # print('wrapper', count_parameters(wrapper))
        # print('hgraph_weights', count_parameters(wrapper.hgraph_weights))
        # print('nweights', count_parameters(wrapper.hgraph_weights.nweights))
        # print('eweights', [count_parameters(wrapper.hgraph_weights.eweights[_]) for _ in etypes])
        # exit()
        opt = torch.optim.Adam(wrapper.hgraph_weights.parameters(), lr)

        with torch.no_grad():
            ori_logits = wrapper.forward_old(g)
            _, ori_preds = torch.max(ori_logits.detach().cpu(), dim=1)

        for nidx in mask_stmt:
            if ori_preds[nidx] == 0:
                continue
            print(num_edges_dict)

            gi = copy.deepcopy(g)

            titers = tqdm.tqdm(range(iters))
            titers.set_description(f'Graph {i}, Node {nidx}')
            for _ in titers:
                preds = wrapper(gi)
                # preds1 = preds[mask_stmt].detach().cpu()

                loss_e = 3 * entropy_loss_mask(gi, etypes)
                loss_c = consistency_loss(preds[nidx].unsqueeze(0), ori_preds[nidx].unsqueeze(0))
                loss_s = 5e-1 * size_loss(gi, etypes) 

                loss = loss_e + loss_c + loss_s

                # print(loss_c, preds[nidx], ori_preds[nidx])
                # titers.set_postfix(loss_e=loss_e.item(), pred=preds[nidx])
                titers.set_postfix(loss_e=loss_e.item(), loss_c=loss_c.item(), loss_s=loss_s.item(), pred=preds[nidx].data)

                # a = list(wrapper.hgraph_weights.parameters())[3].clone()

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(wrapper.hgraph_weights.parameters(), 1.0)
                opt.step()
                # b = list(wrapper.hgraph_weights.parameters())[3].clone()
                # print(a.shape, torch.equal(a.data, b.data))
                # exit()


            visualized_nx_g = map_explain_with_nx(gi, nx_g)
            n_asts = [n for n in visualized_nx_g if
                      visualized_nx_g.nodes[n]['graph'] == 'ast']
            visualized_ast = nx_g.subgraph(n_asts)

            # color = red
            visualized_ast.nodes[n_asts[nidx]]['status'] = 2

            os.makedirs(f'visualize_ast_explained/nbl/stmt_level', exist_ok=True)
            ast_to_agraph(visualized_ast,
                          f'visualize_ast_explained/nbl/stmt_level/Graph{i}_Node{nidx}.png')


if __name__ == '__main__':
    dataset = NBLGumtreeDGLStatementDataset()
    meta_graph = dataset.meta_graph

    model = GCN_A_L_T_1(
        128, meta_graph,
        device=device,
        num_ast_labels=len(dataset.nx_dataset.ast_types),
        num_classes_ast=2)

    model.load_state_dict(torch.load('trained/nbl/Nov-29-2021/model_79_best_top3_gumtree_stmt.pth', map_location=device))
    model.eval()
    explain(model, dataset, iters=10000)
