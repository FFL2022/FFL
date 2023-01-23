import networkx as nx
from collections import Counter, defaultdict
from utils.data_utils import NxDataloader, AstGraphMetadata
from pyg_version.codeflaws.dataloader_cfl_pyg import CodeflawsCFLPyGStatementDataset, CodeflawsCFLNxStatementDataset, CodeflawsCFLStatementGraphMetadata

import torch
from graph_algos.nx_shortcuts import neighbors_out
import pickle as pkl
from pyg_version.model import MPNNModel_A_T_L
import tqdm
import glob
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def attack(nx_g, nx_stmt_nodes, model, pattern_set):
    # 0. convert nx_g to pyg graph
    data, data_stmt_nodes = CodeflawsCFLPyGStatementDataset.nx_to_pyg(
        nx_g, None, nx_stmt_nodes)
    # 1. get model's output
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        out = model(data.xs, data.ess)
    # 2. get the top-k predictions
    topk = 10
    topk_pred = out[data_stmt_nodes].topk(topk, dim=1)[1]
    # 3. Map back to the nx
    topk_nx = set([nx_g.nodes[nx_stmt_nodes[n]] for n in topk_pred[0]])
    orig_nx = nx_g.copy()
    # 4. Attack for each pattern
    for pattern in pattern_set:
        for target in topk_nx:
            nx_g = orig_nx.copy()
            # get the parent
            parent = list(nx_g.predecessors(target))[0]
            # get the relation from parent to target
            relation = nx_g[parent][target][0]['etype']

            # get the top node in the pattern
            top_node = [
                n for n in pattern.nodes if not neighbors_out(
                    n, pattern, lambda u, v, k, e: '_reverse' not in e['etype']
                ) and neighbors_out(n, pattern)
            ][0]
            # 4.2. insert the pattern
            for n in pattern.nodes:
                nx_g.add_node(n)
            for u, v, k, e in pattern.edges(keys=True, data=True):
                nx_g.add_edge(u, v, key=k, **e)
            # 4.3. connect the pattern to the target's parent
            nx_g.add_edge(parent, top_node, key=0, etype=relation)
            '''
            nx_g.add_edge(top_node, target, key=0, etype='next_sibling')
            nx_g.add_edge(target, top_node, key=0, etype='next_sibling_reverse')
            '''
            # 4.2 Get the new output
            data, data_stmt_nodes = CodeflawsCFLPyGStatementDataset.nx_to_pyg(
                nx_g, None, nx_stmt_nodes)
            model.eval()
            with torch.no_grad():
                data = data.to(device)
                out = model(data.xs, data.ess)
            # 4.3. Get the new top-k predictions
            topk_pred = out[data_stmt_nodes].topk(topk, dim=1)[1]
            # 4.4. Map back to the nx
            new_topk_nx = set(
                [nx_g.nodes[nx_stmt_nodes[n]] for n in topk_pred[0]])
            # 4.5. Check if any of the topk changed by iou
            for new_target in new_topk_nx:
                if new_target != target:
                    print('Attack success!')
                    print('Target:', target)
                    print('New target:', new_target)
                    print('Pattern:', pattern)
                    print('Parent:', parent)
                    print('Relation:', relation)
                    print('Top node:', top_node)
                    print('New topk:', new_topk_nx)
                    print('Old topk:', topk_nx)
                    print('New topk pred:', topk_pred)
                    print('Old topk pred:',
                          out[data_stmt_nodes].topk(topk, dim=1)[1])
                    print('New out:', out)
                    print('Old out:', out)
                    return True
            # 4.3 Compare the output with the original one
    return False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--k", type=int, default=10)
    # parser.add_argument("--loss_func",
    #                     type=str,
    #                     default="total_loss_size_stmt_entropy")
    return parser.parse_args()


def main():
    args = get_args()
    # 1. load the pattern set
    pattern_set = [
        pkl.load(fp) for fp in glob.glob(
            'experiments/result/codeflaws_pyc_cfl_stmt_pyg/influential_substructure/*.pkl'
        )
    ]
    # 2. Load meta data and dataset
    nx_dataset = CodeflawsCFLNxStatementDataset()
    meta_data = CodeflawsCFLStatementGraphMetadata(nx_dataset)
    t2id = {'ast': 0, 'test': 1}
    # 2. load the model
    model = MPNNModel_A_T_L(dim_h=64,
                            netypes=len(meta_data.meta_graph),
                            t_srcs=[t2id[e[0]] for e in meta_data.meta_graph],
                            t_tgts=[t2id[e[2]] for e in meta_data.meta_graph],
                            n_al=len(meta_data.t_asts),
                            n_layers=5,
                            n_classes=2).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # 3. Attack and note the result
    attack_success = 0
    bar = tqdm.tqdm(total=len(nx_dataset))
    for i in bar:
        nx_g, stmt_nodes = nx_dataset[i]
        if attack(nx_g, stmt_nodes, model, pattern_set):
            attack_success += 1
        bar.set_description(f'Attack success rate: {attack_success / (i + 1)}')
    print('Attack success rate:', attack_success / len(nx_dataset))
