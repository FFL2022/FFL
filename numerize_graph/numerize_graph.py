from numerize_graph.meta_data_extractor import get_node_type_mapping, get_edge_type_mapping, \
        get_type_node_mapping, get_type_edge_mapping, get_meta_data, get_node_type_signature
import numpy as np
from collections import defaultdict
from functools import lru_cache
import torch
import networkx as nx
import itertools


class AttrEncoder(object):

    def __init__(self, encoders):
        self.encoders = encoders

    @lru_cache(maxsize=1000)
    def match(self, sig, attr):
        return True

    def register(self, encoder):
        self.encoders.append(encoder)

    def encode(self, val):
        return np.concatenate([e(val) for e in self.encoders], axis=0)

    def __call__(self, attr):
        return self.encode(attr)


class SigAttrEncoder(object):

    def __init__(self, attr_encoders):
        self.attr_encoders = attr_encoders

    def __contains__(self, item):
        sig, attr = item
        return any(e.match(sig, attr) for e in self.attr_encoders)

    def __getitem__(self, key):
        sig, attr = key
        return AttrEncoder(
            [e for e in self.attr_encoders if e.match(sig, attr)])


def numerize_graph(nx_g,
                   node_attr_names,
                   edge_attr_names,
                   node_attrs=[],
                   edge_attrs=[],
                   node_attr_encoders={},
                   edge_attr_encoders={}):
    # Each node in nx_g will be associated with a feature vector
    # Each edge in nx_g will be associated with a feature vector
    # Each node type in nx_g will be associated with a feature vector
    # Each edge type in nx_g will be associated with a feature vector
    node_type_mapping = get_node_type_mapping(nx_g, node_attr_names,
                                              node_attrs)
    edge_type_mapping = get_edge_type_mapping(nx_g, node_attr_names,
                                              edge_attr_names, node_attrs,
                                              edge_attrs)
    type_node_mapping = get_type_node_mapping(nx_g, node_attr_names,
                                              node_attrs)
    type_edge_mapping = get_type_edge_mapping(nx_g, node_attr_names,
                                              edge_attr_names, node_attrs,
                                              edge_attrs)
    node_type_feature_vectors = defaultdict(list)
    edge_type_feature_vectors = defaultdict(list)
    for node_type, nodes in type_node_mapping.items():
        node_type_feature_vectors[node_type] = []
        for n in nodes:
            encoded_attrs = []
            sig = node_type_mapping[n]
            for attr_name in node_attr_names:
                if attr_name in node_attr_encoders and attr_name in nx_g.nodes[
                        n]:
                    encoded_attrs.append(node_attr_encoders[sig, attr_name](
                        nx_g.nodes[n][attr_name]))
            node_type_feature_vectors[node_type].append(
                np.concatenate(encoded_attrs, axis=0))
    for edge_type, edges in type_edge_mapping.items():
        edge_type_feature_vectors[edge_type] = []
        for e in edges:
            encoded_attrs = []
            sig = edge_type_mapping[e]
            for attr_name in edge_attr_names:
                if attr_name in edge_attr_encoders:
                    encoded_attrs.append(edge_attr_encoders[sig, attr_name](
                        nx_g.edges[e[0], e[1]][attr_name]))
            edge_type_feature_vectors[edge_type].append(
                np.concatenate(encoded_attrs, axis=0))
    return node_type_feature_vectors, edge_type_feature_vectors


def add_reversal_edge_and_self_loop_type(node_types,
                                         edge_types):
    node_types = sorted(list(node_types))
    edge_types = sorted(list(edge_types))
    node_type_to_str = {
        node_type: str(i)
        for i, node_type in enumerate(sorted(node_types))
    }
    
    edge_type_to_str = {
            edge_type: f"{i}_{node_type_to_str[edge_type[1]]}_{node_type_to_str[edge_type[2]]}"
            for i, edge_type in enumerate(sorted(edge_types))
        }

    # For each edge type, add a reversal edge
    max_etype = max([int(e.split('_')[0]) for e in edge_type_to_str.values()])
    str_etypes = list(edge_type_to_str.values())
    for edge_type in edge_type_to_str.values():
        _, src_idx, dst_idx = edge_type.split('_')
        edge_type_str = f"{max_etype + 1}_{dst_idx}_{src_idx}"
        str_etypes.append(edge_type_str)
        max_etype += 1
    for node_type in node_type_to_str.values():
        edge_type_str = f"{max_etype + 1}_{node_type}_{node_type}"
        str_etypes.append(edge_type_str)
        max_etype += 1
    return str_etypes


def add_reversal_edge_and_self_loop(numerized_graph, node_types, edge_types):
    node_type_to_str = {
        node_type: str(i)
        for i, node_type in enumerate(sorted(node_types))
    }
    edge_type_to_str = {
        edge_type: str(i)
        for i, edge_type in enumerate(sorted(edge_types))
    }
    graph = numerized_graph
    # graph is a tuple of node_attrs, edge_attrs, edge_index
    edge_index = graph[-1]  # dict str -> E 2
    edge_attrs = graph[1]  # dict str -> E attr
    node_attrs = graph[0]  # dict str -> N attr

    # For each edge type, add a reversal edge type
    max_etype = max([int(e.split('_')[0]) for e in edge_type_to_str.values()])
    for edge_type in edge_type_to_str.values():
        if edge_type in edge_index:
            _, src_idx, dst_idx = edge_type.split('_')
            edge_type_str = f"{max_etype + 1}_{dst_idx}_{src_idx}"
            max_etype += 1
            edge_index[edge_type_str] = edge_index[edge_type].flip(1).transpose(0, 1)
            edge_attrs[edge_type_str] = edge_attrs[edge_type]
        max_etype += 1

    # For each node type, create a self-loop edge type

    for node_type in node_type_to_str.values():
        netype = f"{max_etype + 1}_{node_type}_{node_type}"
        max_etype += 1
        if node_type in node_attrs and node_attrs[node_type].shape[0] > 0:
            edge_index[netype] = torch.tensor(
                [[i, i] for i in range(node_attrs[node_type].shape[0])]).T
            edge_index[netype] = torch.stack([
                torch.arange(node_attrs[node_type].shape[0]),
                torch.arange(node_attrs[node_type].shape[0])
            ], dim=0).transpose(0, 1)
            edge_attrs[netype] = torch.zeros(
                node_attrs[node_type].shape[0], 1)
    return (node_attrs, edge_attrs, edge_index)


# Write the same but use torch


class TorchAttrEncoder(object):

    def __init__(self, encoders):
        self.encoders = encoders

    @lru_cache(maxsize=1000)
    def match(self, sig, attr):
        return True

    def register(self, encoder):
        self.encoders.append(encoder)

    def encode(self, val):
        return torch.cat([e(val) for e in self.encoders], dim=0)

    def __call__(self, attr):
        return self.encode(attr)


class TorchSigAttrEncoder(object):

    def __init__(self, attr_encoders):
        self.attr_encoders = attr_encoders

    def __contains__(self, item):
        sig, attr = item
        return any(e.match(sig, attr) for e in self.attr_encoders)

    def __getitem__(self, key):
        sig, attr = key
        return TorchAttrEncoder(
            [e for e in self.attr_encoders if e.match(sig, attr)])


def torch_numerize_graph(nx_g,
                         node_attr_names,
                         edge_attr_names,
                         node_attrs=[],
                         edge_attrs=[],
                         node_attr_encoders={},
                         edge_attr_encoders={},
                         node_types=[],
                         edge_types=[]):
    # Each node in nx_g will be associated with a feature vector
    # Each edge in nx_g will be associated with a feature vector
    # Each node type in nx_g will be associated with a feature vector
    # Each edge type in nx_g will be associated with a feature vector
    node_type_mapping = get_node_type_mapping(nx_g, node_attr_names,
                                              node_attrs)
    edge_type_mapping = get_edge_type_mapping(nx_g, node_attr_names,
                                              edge_attr_names, node_attrs,
                                              edge_attrs)
    type_node_mapping = get_type_node_mapping(nx_g, node_attr_names,
                                              node_attrs)
    type_edge_mapping = get_type_edge_mapping(nx_g, node_attr_names,
                                              edge_attr_names, node_attrs,
                                              edge_attrs)
    # Construct dict converting every node type to a int
    node_type_to_str = {
        node_type: str(i)
        for i, node_type in enumerate(node_types)
    }
    edge_type_to_str = {
        edge_type: f"{i}_{node_type_to_str[edge_type[1]]}_" +
        f"{node_type_to_str[edge_type[2]]}"
        for i, edge_type in enumerate(edge_types)
    }
    node_type_feature_vectors = defaultdict(list)
    edge_type_feature_vectors = defaultdict(list)
    node2type_idx = dict(
        itertools.chain.from_iterable(
            [(n, i) for i, n in enumerate(type_node_mapping[t])]
            for t in type_node_mapping))
    for node_type, nodes in type_node_mapping.items():
        for n in nodes:
            encoded_attrs = []
            sig = node_type_mapping[n]
            for attr_name in node_attr_names:
                if (sig, attr_name) in node_attr_encoders and\
                        attr_name in nx_g.nodes[n]:
                    encoded_attrs.append(node_attr_encoders[sig, attr_name](
                        nx_g.nodes[n][attr_name]))
            if encoded_attrs:
                node_type_feature_vectors[node_type_to_str[node_type]].append(
                    torch.cat(encoded_attrs, dim=0))
            else:
                node_type_feature_vectors[node_type_to_str[node_type]].append(
                    torch.zeros(0))
        node_type_feature_vectors[node_type_to_str[node_type]] =\
            torch.stack(node_type_feature_vectors[node_type_to_str[node_type]])
    eidxs = defaultdict(list)
    for edge_type, edges in type_edge_mapping.items():
        for e in edges:
            encoded_attrs = []
            sig = edge_type_mapping[e]
            for attr_name in edge_attr_names:
                if (sig, attr_name) in edge_attr_encoders:
                    encoded_attrs.append(edge_attr_encoders[sig, attr_name](
                        nx_g.edges[e[0], e[1]][attr_name]))
            if encoded_attrs:
                edge_type_feature_vectors[edge_type_to_str[edge_type]].append(
                    torch.cat(encoded_attrs, dim=0))
            else:
                edge_type_feature_vectors[edge_type_to_str[edge_type]].append(
                    torch.zeros(0))

            eidxs[edge_type_to_str[edge_type]].append(
                torch.tensor([node2type_idx[e[0]], node2type_idx[e[1]]]))
        if edge_type_feature_vectors[edge_type_to_str[edge_type]]:
            edge_type_feature_vectors[edge_type_to_str[edge_type]] =\
                torch.stack(edge_type_feature_vectors[edge_type_to_str[edge_type]])

    for str_etype in eidxs:
        eidxs[str_etype] = torch.stack(eidxs[str_etype])
    return dict(node_type_feature_vectors), dict(
        edge_type_feature_vectors), dict(eidxs)


##### TEST ####
def test_numerize_graph():
    # Create a graph
    g1 = nx.Graph()
    g1.add_node(0, attr1=1, attr2=2)
    g1.add_node(1, attr1=3, attr2=4)
    g1.add_node(2, attr1=5, attr2=6)
    g1.add_edge(0, 1, attr1=7, attr2=8)
    g1.add_edge(1, 2, attr1=9, attr2=10)

    g2 = nx.Graph()
    g2.add_node(0, attr1=1, attr2=2)
    g2.add_node(1, attr1=3, attr2=4)
    g2.add_node(2, attr1=5, attr2=6)
    g2.add_edge(0, 1, attr1=7, attr2=8)
    g2.add_edge(1, 2, attr1=9, attr2=10)

    # Get metadata
    node_attr_names, edge_attr_names, \
        node_types, edge_types = get_meta_data([g1, g2])
    # Pretty print the metadata
    print("Node attribute names: ", node_attr_names)
    print("Edge attribute names: ", edge_attr_names)
    print("Node types: ", node_types)
    print("Edge types: ", edge_types)

    def to_one_hot(int_labels, num_classes):
        return torch.eye(num_classes)[int_labels]

    # Create the encoders
    class NodeAttr1Encoder(TorchAttrEncoder):

        def match(self, sig, attr):
            return attr == "attr1"

    class NodeAttr2Encoder(TorchAttrEncoder):

        def match(self, sig, attr):
            return attr == "attr2"

    class OneHotEncoder():

        def __init__(self, num_classes, vals):
            self.num_classes = num_classes
            self.vals = vals
            self.vals2int = {v: i for i, v in enumerate(vals)}

        def __call__(self, val):
            return to_one_hot(self.vals2int[val], self.num_classes)

    node_attr1_encoder = NodeAttr1Encoder([OneHotEncoder(3, [1, 3, 5])])
    node_attr2_encoder = NodeAttr2Encoder(
        encoders=[OneHotEncoder(3, [2, 4, 6])])
    node_attr_encoders = TorchSigAttrEncoder(
        [node_attr1_encoder, node_attr2_encoder])

    class EdgeAttr1Encoder(TorchAttrEncoder):

        def match(self, sig, attr):
            return attr == "attr1"

    class EdgeAttr2Encoder(TorchAttrEncoder):

        def match(self, sig, attr):
            return attr == "attr2"

    edge_attr1_encoder = EdgeAttr1Encoder([OneHotEncoder(3, [7, 9])])
    edge_attr2_encoder = EdgeAttr2Encoder(encoders=[OneHotEncoder(3, [8, 10])])
    edge_attr_encoders = TorchSigAttrEncoder(
        [edge_attr1_encoder, edge_attr2_encoder])

    node_type_feature_vectors, edge_type_feature_vectors, edge_idxs =\
        torch_numerize_graph(g1,
                             node_attr_names,
                             edge_attr_names,
                             [], [],
                             node_attr_encoders=node_attr_encoders,
                             edge_attr_encoders=edge_attr_encoders,
                             node_types=node_types,
                             edge_types=edge_types)
    # 3 for attr1, 3 for attr2
    assert node_type_feature_vectors['0'].shape == torch.Size([3, 6])
    assert edge_type_feature_vectors['0_0_0'].shape == torch.Size([2, 6])


if __name__ == '__main__':
    test_numerize_graph()
