import networkx as nx
from typing import Tuple, List
import glob
from numerize_graph.numerize_graph import get_meta_data, get_node_type_mapping, get_edge_type_mapping
import pickle as pkl


def load_graphs_and_labels(
        load_dir: str) -> Tuple[List[nx.MultiDiGraph], List[int]]:
    # load all gpickle in the folder
    graphs = []
    labels = []
    label_mapping = {'pos': 1, 'neg': 0, 'uncertain': -1}
    for file in glob.glob(load_dir + "/*.gpickle"):
        graph = nx.read_gpickle(file)
        graphs.append(graph)
        label = label_mapping[file.split('/')[-1].split('_')[0]]
        labels.append(label)

    return graphs, labels


def convert_graph_attrs_to_int(graphs: List[nx.MultiDiGraph], *,
                               node_attr_names, edge_attr_names, node_types,
                               edge_types) -> List[nx.MultiDiGraph]:

    ntype_mapping = {node_type: i for i, node_type in enumerate(node_types)}
    etype_mapping = {edge_type: i for i, edge_type in enumerate(edge_types)}
    for graph in graphs:
        converted_graph = nx.MultiDiGraph()
        ntypes_map = get_node_type_mapping(graph, node_attr_names,
                                           ["graph", "ntype", "is_target"])
        etypes_map = get_edge_type_mapping(graph, node_attr_names,
                                           edge_attr_names,
                                           ["graph", "ntype", "is_target"],
                                           ["etype"])
        for node in graph.nodes:
            node_type = ntypes_map[node]
            converted_graph.add_node(node, label=ntype_mapping[node_type])
        for edge in graph.edges:
            edge_type = etypes_map[edge]
            converted_graph.add_edge(edge[0],
                                     edge[1],
                                     label=etype_mapping[edge_type])
        # rename all nodes to 0, 1, 2, 3, ...
        mapping = {node: i for i, node in enumerate(converted_graph.nodes)}
        converted_graph = nx.relabel_nodes(converted_graph, mapping)
        yield converted_graph


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


if __name__ == '__main__':
    # 1. read all the graphs
    graphs, labels = load_graphs_and_labels(
        'ego_pyg_codeflaws_pyc_cfl_stmt_level')
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
