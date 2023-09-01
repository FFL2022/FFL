import networkx as nx
from typing import Tuple, List
import glob
from numerize_graph.numerize_graph import (
        get_meta_data,
        get_node_type_mapping,
        get_edge_type_mapping,
        convert_single_graph_attrs_to_int,
        convert_graph_attrs_to_int,
        convert_graphs_int_to_attr_single,
        convert_graphs_int_to_attr
        )
import pickle as pkl
import argparse
import os
import tqdm


def load_graphs_and_labels(
        load_dir: str) -> Tuple[List[nx.MultiDiGraph], List[int]]:
    # load all gpickle in the folder
    graphs = []
    labels = []
    label_mapping = {'pos': 1, 'neg': 0, 'uncertain': -1}
    counts = {1: 0, 0: 0, -1: 0}
    # threshold: 4000 graphs each
    for fp in glob.glob(load_dir + "/*.gpickle"):
        graph = nx.read_gpickle(fp)
        label = label_mapping[fp.split('/')[-1].split('_')[0]]
        if counts[label] < 4000:
            graphs.append(graph)
            labels.append(label)
            counts[label] += 1
    return graphs, labels


def remove_self_loops(graphs: List[nx.MultiDiGraph],
                      verbose=False) -> List[nx.MultiDiGraph]:
    bar = graphs
    if verbose:
        bar = tqdm.tqdm(graphs)
        bar.set_description("Removing self loops")
    for graph in bar:
        for edge in list(graph.edges)[:]:
            if edge[0] == edge[1]:
                graph.remove_edge(edge[0], edge[1])
    return graphs


def remove_inverse_edge(graphs: List[nx.MultiDiGraph],
                        verbose=False) -> List[nx.MultiDiGraph]:
    bar = graphs
    if verbose:
        bar = tqdm.tqdm(graphs)
        bar.set_description("Removing inverse edges")
    for graph in bar:
        for edge in list(graph.edges)[:]:
            if '_reverse' in graph.edges[edge]['etype']:
                graph.remove_edge(edge[0], edge[1])
    return graphs


def to_gspan_format(converted_graphs: List[nx.MultiDiGraph],
                    labels: List[int]) -> str:
    # the format is
    # t # 0 label
    # v 0 vlabel
    # v 1 vlabel
    # ...
    # e 0 1 elabel
    # e 1 2 elabel
    # ...
    out_str = ""
    for i, graph in enumerate(converted_graphs):
        out_str += f"t # {i} {labels[i]}\n"
        for node in graph.nodes:
            out_str += f"v {node} {graph.nodes[node]['label']}\n"
        for edge in graph.edges:
            out_str += f"e {edge[0]} {edge[1]} {graph.edges[edge]['label']}\n"
    return out_str


def from_gspan_format(gspan_str: str,
                      supports: List[int] = None) -> List[nx.MultiDiGraph]:
    lines = gspan_str.split('\n')
    graphs = []
    graph = None
    for line in lines:
        if line.startswith('t'):
            if graph is not None:
                graphs.append(graph)
            graph = nx.MultiDiGraph()
            if supports is not None:
                supports.append(int(line.split(' ')[-1]))
        elif line.startswith('v'):
            node, label = line.split(' ')[1:]
            graph.add_node(int(node), label=int(label))
        elif line.startswith('e'):
            src, dst, label = line.split(' ')[1:]
            graph.add_edge(int(src), int(dst), label=int(label))
    return graphs


def nx_to_gspan(graphs: List[nx.MultiDiGraph], labels: List[int],
                node_attr_names, edge_attr_names, node_types,
                edge_types) -> str:
    converted_graphs = convert_graph_attrs_to_int(
        graphs,
        node_attr_names=node_attr_names,
        edge_attr_names=edge_attr_names,
        node_types=node_types,
        edge_types=edge_types)
    converted_graphs = remove_self_loops(converted_graphs)
    gspan_str = to_gspan_format(converted_graphs, labels)
    return gspan_str


def gspan_to_nx(gspan_str: str, node_attr_names, edge_attr_names, node_types,
                edge_types) -> List[nx.MultiDiGraph]:
    graphs = from_gspan_format(gspan_str)
    graphs = convert_graphs_int_to_attr(graphs, node_attr_names,
                                        edge_attr_names, node_types,
                                        edge_types)
    return graphs


def main_nx_to_gspan(args):
    # 1. read all the graphs
    graphs, labels = load_graphs_and_labels(
        'ego_pyg_codeflaws_pyc_cfl_stmt_level')
    graphs = remove_self_loops(graphs)
    graphs = remove_inverse_edge(graphs)
    # 2. convert to gSpan format
    node_attr_names, edge_attr_names, node_types, edge_types = get_meta_data(
        graphs, ["graph", "ntype", "is_target"], ["etype"])
    pkl.dump((node_attr_names, edge_attr_names, node_types, edge_types),
             open('ego_pyg_codeflaws_pyc_cfl_stmt_level/meta_data.pkl', 'wb'))
    gspan_str = to_gspan_format(
        convert_graph_attrs_to_int(graphs,
                                   node_attr_names=node_attr_names,
                                   edge_attr_names=edge_attr_names,
                                   node_types=node_types,
                                   edge_types=edge_types), labels)
    print(gspan_str)


def main_gspan_to_nx(args):
    # 1. read all the graphs
    gspan_str = open(
        'ego_pyg_codeflaws_pyc_cfl_stmt_level/ego_pyg_codeflaws_pyc_cfl_stmt_level.gspan'
    ).read()
    node_attr_names, edge_attr_names, node_types, edge_types = pkl.load(
        open('ego_pyg_codeflaws_pyc_cfl_stmt_level/meta_data.pkl', 'rb'))
    graphs = convert_graphs_int_to_attr(from_gspan_format(gspan_str),
                                        node_attr_names=node_attr_names,
                                        edge_attr_names=edge_attr_names,
                                        node_types=node_types,
                                        edge_types=edge_types)
    # dump all the graph to disk
    os.makedirs('ego_pyg_codeflaws_pyc_cfl_stmt_level/mined', exist_ok=True)
    for i, graph in enumerate(graphs):
        nx.write_gpickle(
            graph, f'ego_pyg_codeflaws_pyc_cfl_stmt_level/mined/{i}.gpickle')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--meta_data', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    if args.mode == 'nx_to_gspan':
        # 1. read all the graphs
        graphs, labels = load_graphs_and_labels(args.input)
        graphs = remove_self_loops(graphs)
        graphs = remove_inverse_edge(graphs)
        # 2. convert to gSpan format
        node_attr_names, edge_attr_names, node_types, edge_types = get_meta_data(
            graphs, ["graph", "ntype", "is_target"], ["etype"])
        pkl.dump((node_attr_names, edge_attr_names, node_types, edge_types),
                 open(args.meta_data, 'wb'))
        gspan_str = to_gspan_format(
            convert_graph_attrs_to_int(graphs,
                                       node_attr_names=node_attr_names,
                                       edge_attr_names=edge_attr_names,
                                       node_types=node_types,
                                       edge_types=edge_types), labels)
        # write to args.output
        with open(args.output, 'w') as f:
            f.write(gspan_str)
    elif args.mode == 'gspan_to_nx':
        # 1. read all the graphs
        gspan_str = open(args.input).read()
        node_attr_names, edge_attr_names, node_types, edge_types = pkl.load(
            open(args.meta_data, 'rb'))
        graphs = convert_graphs_int_to_attr(from_gspan_format(gspan_str),
                                            node_attr_names=node_attr_names,
                                            edge_attr_names=edge_attr_names,
                                            node_types=node_types,
                                            edge_types=edge_types)
        # dump all the graph to disk
        os.makedirs(args.output, exist_ok=True)
        for i, graph in enumerate(graphs):
            nx.write_gpickle(graph, f'{args.output}/{i}.gpickle')
