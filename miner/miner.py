from miner.utils import combine_to_one_graph, check_equal_nx
from miner.Subdue import nx_subdue
import networkx as nx
import os
import argparse
import json
import pickle as pkl

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default= "bfs")
    parser.add_argument("--signature", "--sig", type=str, required=True)
    args = parser.parse_args()
    setattr(args, 'save_dir', f'experiments/data/{args.signature}')
    setattr(args, 'result_dir', f'experiments/result/{args.signature}')
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    if not os.path.exists(os.path.join(args.save_dir,'args.txt')):
        json.dump({}, open(os.path.join(args.save_dir,'args.txt'), 'w'))
    with open(os.path.join(args.save_dir,'args.txt'), 'r') as f:
        args.__dict__.update(json.load(f))
    print(args)
    return args

def save_pattern(nx_g, save_path):
    relevant_nodes = []
    target_node = None
    for node in nx_g.nodes(data=True):
        node_idx, attrs = node
        if attrs['is_target'] == 1:
            target_node = node_idx
        else:
            relevant_nodes.append(node_idx)
        nx_g.nodes[node_idx].pop('is_target',None)
    for edge_idx in nx_g.edges:
        nx_g.edges[edge_idx].pop('eweights',None)
        if 'is_target' in nx_g.edges[edge_idx]:
            nx_g.edges[edge_idx].pop('is_target',None)

    mapping = dict((n, i) for i, n in enumerate(sorted(relevant_nodes)))
    mapping[target_node] = 'y'
    pattern = nx.relabel_nodes(nx_g, mapping)
    print(
        list(pattern.nodes(data=True)),
        list(pattern.edges(data=True)))
    pkl.dump(pattern, open(save_path, 'wb'))

def check_in_set(graphset, nx_g, v_attrs=[], e_attrs=[]):
    for ele in graphset:
        print(ele)
        if ele == nx_g or check_equal_nx(ele, nx_g, v_attrs, e_attrs):
            return True
    return False


def mine(args, kept_props_v=[], kept_props_e=[]):
    save_dir = f"{args.result_dir}/influential_substructure"
    os.makedirs(save_dir, exist_ok=True)
    combined_graph = combine_to_one_graph(
        f"{args.save_dir}/explain/substructure",
        kept_props_v=kept_props_v, kept_props_e=kept_props_e)
    print("Patterns for {args.signature}:\n")
    patterns = nx_subdue(combined_graph, v_attribs=kept_props_v, e_attribs=[])
    nx_out_list = []
    for i, pattern in enumerate(patterns):
        nx_gs = []
        for inst in pattern:
            nx_g = nx.edge_subgraph(combined_graph, [(u, v, 0) for u, v in inst['edges']]).copy()
            nx_gs.append(nx_g)
        if not check_in_set(nx_out_list, nx_gs[0], v_attrs=kept_props_v):
            nx_out_list.append(nx_gs[0])
    for pattern_idx, nx_g in enumerate(nx_out_list):
        print(f"Pattern {pattern_idx}:")
        save_path = os.path.join(save_dir, f"{pattern_idx}.pkl")
        save_pattern(nx_g, save_path)

    return nx_out_list



if __name__ == '__main__':
    args = get_args()
    if args.task == "bfs":
        mine(args, ['is_target'], [])
    elif args.task == "blmfd":
        mine(args, ['is_target'], ['is_target'])
    elif args.task == 'sat':
        mine(args, ['is_target'], ['label'])
    elif args.task == 'tsp':
        mine(args, ['is_target'], [])
    elif args.task == 'dfs':
        mine(args, ['is_target'], ['etype'])
    elif args.task == 'cpp':
        mine(args, ['is_target', 'ntype'], ['etype'])
    else:
        raise NotImplementedError("Not implemented for this task")
