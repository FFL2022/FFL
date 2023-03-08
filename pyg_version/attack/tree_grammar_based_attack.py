import os
import networkx as nx
import pickle as pkl
from codeflaws.dataloader_cfl import CodeflawsCFLNxStatementDataset
import tree_grammars.utils
import torch
from pyg_version.dataloader_cfl_pyg import PyGStatementDataset, AstGraphMetadata
from numerize_graph.numerize_graph import get_meta_data, get_node_type_mapping, get_edge_type_mapping
from utils.train_utils import BinFullMeter, AverageMeter
import tqdm
from graph_algos.nx_shortcuts import neighbors_out
import argparse
from miner.gspan_cork_utils import remove_self_loops, remove_inverse_edge, to_gspan_format, convert_graph_attrs_to_int

from pyg_version.model import MPNNModel_A_T_L

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# define a named tuple
TreeGrammarMeta = NamedTuple('GrammarMetadata', ['node_attr_names', 'edge_attr_names', 'node_types', 'edge_types'])


def attack_independent(nx_g, nx_stmt_nodes, model, tree_grammar, meta_data, tree_grammar_meta: GrammarMetadata, max_attempt=4):
    n2ntype = get_node_type_mapping(nx_g, tree_grammar_meta.node_attr_names,
                                       ["graph", "ntype", "is_target"])
    e2etype = get_edge_type_mapping(nx_g, tree_grammar_meta.node_attr_names,
                                       tree_grammar_meta.edge_attr_names,
                                       ["graph", "ntype", "is_target"],
                                       ["label"])

    ntype_mapping = {i: node_type for i, node_type in enumerate(tree_grammar_meta.node_types)}
    etype_mapping = {i: edge_type for i, edge_type in enumerate(tree_grammar_meta.edge_types)}
    rev_ntype_mapping = {node_type: i for i, node_type in enumerate(tree_grammar_meta.node_types)}
    rev_etype_mapping = {edge_type: i for i, edge_type in enumerate(tree_grammar_meta.edge_types)}
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

    for target in topk_nx:
        parent = list(nx_g.predecessors(target))[0]
        relation = nx_g[parent][target][0]['label']
        parent_label = rev_ntype_mapping[n2ntype[parent]]
        relation_label = rev_etype_mapping[e2etype[parent, target, 0]]
        possible_edges = tree_grammar.sample_independent_rules(label_u=parent_label, label_e=relation_label, k = max_attempt)
        if possible_edges is None:
            continue
        # denumerize these edges
        for srclabel, dstlabel, elabel in possible_edges:
            nx_g = orig_nx.copy()
            # add another node
            new_node = nx_g.number_of_nodes()
            dst_ntype = ntype_mapping[dstlabel]
            dst_attrs = {
                attr_name: attr_val
                for attr_name, attr_val, has_attr in dst_ntype if has_attr
            }
            nx_g.add_node(f"ast_{new_node}", **dst_attrs)
            e_attrs = etype_mapping[elabel][0]
            e_attrs = {
                attr_name: attr_val
                for attr_name, attr_val, has_attr in e_attrs if has_attr
            }
            nx_g.add_edge(parent, f"ast_{new_node}", **e_attrs)
            ''' 
            nx_g.add_edge(top_node, target, key=0, label='next_sibling')
            nx_g.add_edge(target, top_node, key=0, label='next_sibling_reverse')
            '''
            # 4.2 Get the new output
            new_nx_stmt_nodes = nx_stmt_nodes[:] + [f"ast_{new_node}"]
            data, data_stmt_nodes = PyGStatementDataset.nx_to_pyg(
                meta_data, nx_g, None, new_nx_stmt_nodes)
            # print shape of data.xs and max of data.ess
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
            new_topk_nx = set([new_nx_stmt_nodes[n] for n in topk_pred])
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
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--attack_type", type=str, default='independent'
    )  # can be independent, breath-first, or contextual-conditioned random
    # parser.add_argument("--loss_func",
    #                     type=str,
    #                     default="total_loss_size_stmt_entropy")
    return parser.parse_args()


def main():
    args = get_args()
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

    # 3. Build the tree grammar
    os.makedirs('tree_grammar_save', exist_ok=True)
    if os.path.exists('tree_grammar_save/tree_grammar.pkl'):
        tree_grammar = pkl.load(
            open('tree_grammar_save/tree_grammar.pkl', 'rb'))
        node_attr_names, edge_attr_names, node_types, edge_types = pkl.load(
            open('tree_grammar_save/tree_grammar_meta.pkl', 'rb'))
        tree_grammar_meta = TreeGrammarMeta(node_attr_names, edge_attr_names,
                                            node_types, edge_types)
    else:
        graphs = remove_self_loops(nx_dataset)
        graphs = remove_inverse_edge(graphs)
        node_attr_names, edge_attr_names, node_types, edge_types = get_meta_data(
            graphs, ["graph", "ntype"], ["label"])
        pkl.dump((node_attr_names, edge_attr_names, node_types, edge_types),
                 open("tree_grammar_save/tree_grammar_meta.pkl", 'wb'))
        numerized_graphs = convert_graph_attrs_to_int(
            graphs,
            node_attr_names=node_attr_names,
            edge_attr_names=edge_attr_names,
            node_types=node_types,
            edge_types=edge_types)
        tree_grammar = tree_grammars.utils.ProbabilisticTreeGrammar(
                    numerized_graphs
                )
        pkl.dump(tree_grammar,
                    open('tree_grammar_save/tree_grammar.pkl', 'wb'))
        
    # 4. Attack
    attack_success = 0
    bar = tqdm.trange(len(nx_dataset))
    for i in bar:
        nx_g, stmt_nodes = nx_dataset[i]
        success, min_recs = attack_independent(nx_g, stmt_nodes, model, tree_grammar,
                                               meta_data,
                                               tree_grammar_meta, max_attempt=4)
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
