import networkx as nx
from codeflaws.dataloader_cfl import CodeflawsCFLNxStatementDataset
from collections import Counter, defaultdict
from utils.data_utils import NxDataloader, AstGraphMetadata
from utils.train_utils import BinFullMeter, AverageMeter
from pyg_version.dataloader_cfl_pyg import PyGStatementDataset, AstGraphMetadata

import torch
from graph_algos.nx_shortcuts import neighbors_out
import pickle as pkl
from pyg_version.model import MPNNModel_A_T_L
import tqdm
import glob
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def attack(nx_g, nx_stmt_nodes, model, pattern_set, meta_data):
    # 0. convert nx_g to pyg graph
    data, data_stmt_nodes = PyGStatementDataset.nx_to_pyg(
        meta_data, nx_g, None, nx_stmt_nodes)
    # 1. get model's output
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        out = model(data.xs, data.ess)[1]
    # 2. get the top-k predictions
    topk = min(10, data_stmt_nodes.shape[0])
    topk_pred = out[data_stmt_nodes.long(), 1].topk(topk, dim=0)[1]
    # 3. Map back to the nx
    topk_nx = set([nx_stmt_nodes[n] for n in topk_pred])
    orig_nx = nx_g.copy()
    # 4. Attack for each pattern
    success = False

    ast_lb = data.lbl[data_stmt_nodes.long()]
    non_zeros_lbs = torch.nonzero(ast_lb).detach()
    ast_lbidxs = set(torch.flatten(non_zeros_lbs).detach().cpu().tolist())
    min_recs = [1] * 4
    top_updates = [1, 3, 5, 10]
    top_updates = [min(top_updates[i], topk) for i in range(len(top_updates))]
    for pattern in pattern_set:
        for target in topk_nx:
            nx_g = orig_nx.copy()
            # get the parent
            parent = list(nx_g.predecessors(target))[0]
            # get the relation from parent to target
            relation = nx_g[parent][target][0]['label']

            # get the top node in the pattern
            top_node = [
                n for n in pattern.nodes if not neighbors_out(
                    n, pattern, lambda u, v, k, e: '_reverse' not in e['etype']
                ) and neighbors_out(n, pattern)
            ][0]
            # 4.2. insert the pattern
            for n, d  in pattern.nodes(data=True):
                nx_g.add_node(n, graph='ast', **d, status=0)
            for u, v, k, e in pattern.edges(keys=True, data=True):
                nx_g.add_edge(u, v, key=k, label=e['etype'], **e)
            # 4.3. connect the pattern to the target's parent
            nx_g.add_edge(parent, top_node, key=0, label=relation)
            '''
            nx_g.add_edge(top_node, target, key=0, label='next_sibling')
            nx_g.add_edge(target, top_node, key=0, label='next_sibling_reverse')
            '''
            # 4.2 Get the new output
            new_nx_stmt_nodes = nx_stmt_nodes[:] + [top_node]
            data, data_stmt_nodes = PyGStatementDataset.nx_to_pyg(
                meta_data, nx_g, None, new_nx_stmt_nodes)
            model.eval()
            with torch.no_grad():
                data = data.to(device)
                out = model(data.xs, data.ess)[1]
            # 4.3. Get the new top-k predictions
            # use the old topk
            topk_pred = out[data_stmt_nodes.long(), 1].topk(topk, dim=0)[1]
            for i, (k, curr_rec) in enumerate(zip(top_updates, min_recs)):
                topk_val = topk_pred[:k].tolist()
                new_rec = int(any([i in ast_lbidxs for i in topk_val]))
                min_recs[i] = min(new_rec, curr_rec)

            # 4.4. Map back to the nx
            new_topk_nx = set(
                [new_nx_stmt_nodes[n] for n in topk_pred])
            # 4.5. Check if any of the topk changed by iou
            if new_topk_nx != topk_nx:
                success = True
            '''
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
            '''
    return success, min_recs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern_dir", type=str, default='experiments/result/codeflaws_pyc_cfl_stmt_pyg/influential_substructure/*.pkl')
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
        pkl.load(open(fp, 'rb')) for fp in glob.glob(args.pattern_dir)
    ]
    # 2. Load meta data and dataset
    nx_dataset = CodeflawsCFLNxStatementDataset()
    meta_data = AstGraphMetadata(nx_dataset)
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
    top_1_rec, top_3_rec, top_5_rec, top_10_rec = [
        AverageMeter() for _ in range(4)
    ]
    # 3. Attack and note the result
    attack_success = 0
    bar = tqdm.trange(len(nx_dataset))
    for i in bar:
        nx_g, stmt_nodes = nx_dataset[i]
        success, min_recs = attack(nx_g, stmt_nodes, model, pattern_set, meta_data)
        if success:
            attack_success += 1
        top_1_rec.update(min_recs[0])
        top_3_rec.update(min_recs[1])
        top_5_rec.update(min_recs[2])
        top_10_rec.update(min_recs[3])
        bar.set_description(f'Attack success: {attack_success}/{i+1}')
        bar.set_postfix(top_1=top_1_rec.avg,
                        top_3=top_3_rec.avg,
                        top_5=top_5_rec.avg,
                        top_10=top_10_rec.avg)
    print('Attack success rate:', attack_success / len(nx_dataset))


if __name__ == '__main__':
    main()
