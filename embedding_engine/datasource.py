from __future__ import print_function, unicode_literals
import torch
import torch.nn as nn
import json
import numpy as np
from execution_engine import dataset
from graph_algos.graph_transforms import make_graph_fully_connected, \
    remove_no_edge
from graph_algos.graph_sampling import SubDocGraphSampler
from graph_algos.spanning_tree_conversion import sample_bfs_from_graph


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DataSource:
    def gen_batch(batch_tgt, batch_ntgt, batch_nquery, train):
        raise NotImplementedError


class OTFDocumentGraphDataSource(DataSource):
    ''' On-the-fly generated document data for training the subgraph model
    At every iteration, new batch of graphs (positive and negative) are
    generated with a pre-defined generator
    '''

    def __init__(self, max_size=200, min_size=5,
                 n_workers=4, max_queue_size=256,
                 transform_to_full_graph=True,
                 node_anchored=False):
        self.closed = False
        self.min_size = min_size
        self.max_size = max_size
        classes = json.load(open('data/classes.json', 'r'))
        corpus = open('data/corpus.txt', 'r').read()
        self.document_graph_dataset = dataset.NeuralExecutionDataset(
            classes=classes, corpus=corpus, mode='train',
            raw_dir='./data/train'
        )
        self.transform_to_full_graph = transform_to_full_graph

    def gen_data_loaders(self, size, batch_size, train=True,
                         use_distributed_sampling=False):
        loaders = []

    def gen_batch(self, idx, batch_size, max_depth=15, anchored=True):
        ''' Generate a batch of positive query and positive graph,
        negative query and negative graph
        Parameters
        ----------
        idx: int
              index of the original document graph to be sampled against
        batch_size: int
        max_depth: default=15
            The maximum depth of the binary tree
        Returns
        ----------
        anchors: list[str]
            Anchor for each pair
        positive_pairs: tuple[graph, graph]
            positive pair of (query graph, target graph)
        negative_pairs: tuple[graph, graph]
            negative pair of (query graph, target graph)
        dgl_pos_pairs: tuple[dgl.Graph, dgl.Graph]
        dgl_neg_pairs: tuple[dgl.Graph, dgl.Graph]
        '''
        dgl_G, doc_graph, _ = self.document_graph_dataset[idx]
        # Transform to full graph regardless
        doc_graph = make_graph_fully_connected(doc_graph)
        node_dicts, centers = sample_bfs_from_graph(doc_graph, batch_size,
                                                    max_depth)
        sub_graphs = SubDocGraphSampler.sample_graph_from_node_dicts(
            doc_graph,
            node_dicts,
            centers)
        if not self.transform_to_full_graph:
            doc_graph = remove_no_edge(doc_graph)
            sub_graph = [remove_no_edge(sub_graph) for sub_graph in sub_graphs]

        # make it fully connected one more time
        o_subgraphs = []
        o_newsubgraphs = []
        o_dgl_subgraphs = []
        o_dgl_newsubgraphs = []
        if self.transform_to_full_graph:
            choices = np.random.binomial(1, 0.5, batch_size)
        else:
            choices = np.random.binomial(2, 0.5, batch_size)

        mutually_exclusive_edges = {
            'tb': 'bt',
            'lr': 'rl',
            'bt': 'tb',
            'rl': 'lr',
            'no_edge': ['lr', 'rl', 'bt', 'tb']
        }
        # Choice on which kind of expansion
        for i in range(batch_size):
            sub_graph = sub_graphs[i]
            parent_subgraph = sub_graph.copy()
            new_subgraph = None
            if choices[i] == 0:
                # 1. Add new nodes from original graph: Sampling a bigger
                # subgraph
                sub_nodes = sub_graph.nodes()
                all_nodes = doc_graph.nodes()
                expandable_nodes = list(
                    set(all_nodes).difference(set(sub_nodes)))
                node_choices = np.random.binomial(
                    1, 0.5, len(expandable_nodes))
                new_nodes = [expandable_nodes[j]
                             for j, choice in enumerate(node_choices)
                             if node_choices[j] > 0]
                new_subgraph = doc_graph.subgraph(list(sub_nodes)
                                                  + list(new_nodes)).copy()
            elif choices[i] == 1:
                # 2. Add edges between ones that are already connected
                all_edges = list(sub_graph.edges())
                all_labels = list(set([e['label'] for _, _, _, e in
                                       sub_graph.edges(keys=True, data=True)]))
                new_subgraph = sub_graph.copy()
                for u, v in all_edges:
                    labels = []
                    for key, data in sub_graph.get_edge_data(u, v).items():
                        labels.append(data['label'])
                    me_labels = []
                    for label in labels:
                        if isinstance(mutually_exclusive_edges[label], str):
                            me_labels.append(mutually_exclusive_edges[label])
                        else:
                            me_labels.extend(mutually_exclusive_edges[label])
                    labels.extend(me_labels)
                    extendable_labels = list(set(all_labels).difference(labels))
                    # binomial choice
                    echoice = np.random.binomial(
                        1, 0.5, len(extendable_labels))
                    for l, label in enumerate(extendable_labels):
                        if echoice[l] == 1:
                            new_subgraph.add_edge(u, v, label=label)
            else:
                # Note: this option does not exist when we use 'no_edge'
                # If we were to use this attribute, it is simply not possible
                # to add a new edge out of the old
                # Because if that case is a match, the matching condition of
                # 'no_edge' would be violated
                list_nodes = list(sub_graph.nodes())
                node_maps = dict([(node, j)
                                 for j, node in enumerate(list_nodes)])
                A_no_edges = np.ones(
                    (len(list_nodes), len(list_nodes)),
                    dtype=np.int)
                for u, v in sub_graph.edges():
                    A_no_edges[node_maps[u], node_maps[v]] = 0

                all_no_edges = np.argwhere(A_no_edges > 1)
                no_edge_choices = np.random.binomial(1, 0.5, all_no_edges.shape[0]) > 0
                all_no_edges = all_no_edges[no_edge_choices, :]
                new_subgraph = sub_graph.copy()
                all_labels = list(set([e['label'] for _, _, _, e in
                                       sub_graph.edges(keys=True, data=True)]))
                label_choices = np.random.binomial(len(all_labels)-1,
                                                   0.5, all_no_edges.shape[0])
                for j in range(all_no_edges.shape[0]):
                    edge = all_no_edges[i]
                    new_subgraph.add_edge(
                        list_nodes[edge[0]], list_nodes[edge[1]],
                        label=all_labels[label_choices[i]]
                    )
            o_subgraphs.append(sub_graph)
            o_newsubgraphs.append(new_subgraph)
            center = None
            if anchored:
                center = centers[i]
            dgl_new_subgraph = self.document_graph_dataset.convert_single_graph_to_dgl(
                new_subgraph, center=center)
            dgl_subgraph = self.document_graph_dataset.convert_single_graph_to_dgl(
                sub_graph, center=center)
            o_dgl_subgraphs.append(dgl_subgraph)
            o_dgl_newsubgraphs.append(dgl_new_subgraph)

        return o_subgraphs, o_newsubgraphs, o_dgl_subgraphs, o_dgl_newsubgraphs

    def __len__(self):
        return len(self.document_graph_dataset)
