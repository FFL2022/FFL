from codeflaws.dataloader_cfl import CodeflawsCFLNxStatementDataset
from pyg_version.dataloader_cfl_pyg import PyGStatementDataset, AstGraphMetadata
from pyg_version.model import MPNNModel_A_T_L
from pyg_version.explainer.explainer import Explainer, InflExtractor
from pyg_version.explainer.common import entropy_loss_mask, size_loss
from graph_algos.nx_shortcuts import where_node, where_edge
# Todo: use edge weight and feature weight balanced
import torch
import networkx as nx
import os
import argparse
from pyg_version.codeflaws.explain_nx_a_nc_cfl_pyg import from_data_to_nx
from typing import Tuple, Dict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Todo: sample ego graph from the whole graph, the ratio of positive/negative
# nodes is determined by a ratio
# If a node belong to both positive and negative, it is considered as unknown


class TopKTripletStatementIterator(object):

    def __init__(self, model, dataset: PyGStatementDataset, k,
                 device):
        self.model = model
        self.dataset = dataset
        self.k = k
        self.len_data = self.calc_len()
        self.device = device
        self.model = self.model

    def calc_len(self):
        # loop through the dataset and check how many statements
        # are in the data
        # return the number of statements
        tot = 0
        for data, stmt_nodes in self.dataset:
            tot += min(len(stmt_nodes), self.k)
        return tot

    def __iter__(self):
        self.model.eval()
        with torch.no_grad():
            for data, stmt_nodes in self.dataset:
                data = data.to(device)
                output = self.model(data.xs, data.ess)[0]
                output = output[stmt_nodes, 1].cpu()
                k = min(len(stmt_nodes), self.k)
                totlen = len(stmt_nodes)
                topk = output.topk(totlen, dim=0)[1]
                k_pos = list(range(k - max(k-totlen//2, 0)))
                k_uncertain = list(range(k - max(k-totlen//2, 0), min(k + max(k-totlen//2, 0), 0)))
                k_neg = list(range(totlen - max((k - max(k-totlen//2, 0)), 0), totlen))
                yield (data, stmt_nodes.to(device)), output, topk[k_pos], topk[k_uncertain], topk[k_neg]

    def __len__(self):
        return self.len_data


class EgoGraphExtractor(object):

    def __init__(self, model, dataset: PyGStatementDataset, k, hops,
                 meta_data, device):
        self.triplet_iter = TopKTripletStatementIterator(model, dataset, k, device)
        self.meta_data = meta_data
        self.hops = hops

    def extract(self) -> nx.MultiDiGraph:
        for (data, stmt_nodes), output, k_pos, k_uncertain, k_neg in self.triplet_iter:
            stmt_nodes = stmt_nodes.to("cpu")
            data = data.to("cpu")
            # 1. convert data to nx
            graph = from_data_to_nx(data, None, self.meta_data)
            # 2. extract the ego graph from the nx
            # 2.1 extract the top k positive nodes
            pos_ego_graphs = []
            for pos_node in stmt_nodes[k_pos]:
                ego_graph = nx.ego_graph(graph, f"ast_{pos_node}", self.hops, undirected=True).copy()
                ego_graph.nodes[f"ast_{pos_node}"]['is_target'] = 1
                ego_graph.nodes[f"ast_{pos_node}"]['color'] = 'red'
                ego_graph.nodes[f"ast_{pos_node}"]['style'] = 'filled'
                pos_ego_graphs.append(ego_graph)
            # 2.2 extract the top k negative nodes
            neg_ego_graphs = []
            for neg_node in stmt_nodes[k_neg]:
                ego_graph = nx.ego_graph(graph, f"ast_{neg_node}", self.hops, undirected=True).copy()
                ego_graph.nodes[f"ast_{neg_node}"]['is_target'] = 1
                ego_graph.nodes[f"ast_{neg_node}"]['color'] = 'red'
                ego_graph.nodes[f"ast_{neg_node}"]['style'] = 'filled'
                neg_ego_graphs.append(ego_graph)
            # 2.3 extract the top k uncertain nodes
            uncertain_ego_graphs = []
            for uncertain_node in stmt_nodes[k_uncertain]:
                ego_graph = nx.ego_graph(graph, f"ast_{uncertain_node}", self.hops, undirected=True).copy()
                ego_graph.nodes[f"ast_{uncertain_node}"]['is_target'] = 1
                ego_graph.nodes[f"ast_{uncertain_node}"]['color'] = 'red'
                ego_graph.nodes[f"ast_{uncertain_node}"]['style'] = 'filled'
                uncertain_ego_graphs.append(ego_graph)
            # 3. yield the ego graph
            yield pos_ego_graphs, neg_ego_graphs, uncertain_ego_graphs



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path",
                        type=str,
                        default="ego_pyg_codeflaws_pyc_cfl_stmt_level")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--hops", type=int, default=5)
    # parser.add_argument("--loss_func", type=str,
    #                     default="total_loss_size_stmt_entropy")
    parser.add_argument("--device", type=str, default='cuda')
    return parser.parse_args()



def main():
    t2id = {'ast': 0, 'test': 1}
    args = get_args()
    print("Loading data...")
    nx_dataset = CodeflawsCFLNxStatementDataset()
    #### TEST ####
    os.makedirs("tmp", exist_ok=True)
    for i, (nx_g, stmt_nodes) in enumerate(nx_dataset):
        nx.write_gpickle(nx_g, f"tmp/{i}.gpickle")
        # remove the name attribute if there is from nodes and edges
        for node in nx_g.nodes:
            nx_g.nodes[node].pop("name", None)
        for edge in nx_g.edges:
            nx_g.edges[edge].pop("name", None)
        nx.drawing.nx_pydot.write_dot(nx_g, f"tmp/{i}.dot")
    #### END TEST ####
    meta_data = AstGraphMetadata(nx_dataset)
    pyg_dataset = PyGStatementDataset(
            dataloader=nx_dataset,
            meta_data=meta_data,
            ast_enc=None,
            save_dir="preprocessed/codeflaws/",
            name='train_pyg_cfl_stmt')
    print("Loading model...")
    model = MPNNModel_A_T_L(dim_h=64,
                            netypes=len(meta_data.meta_graph),
                            t_srcs=[t2id[e[0]] for e in meta_data.meta_graph],
                            t_tgts=[t2id[e[2]] for e in meta_data.meta_graph],
                            n_al=len(meta_data.t_asts),
                            n_layers=5,
                            n_classes=2).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    print("Extracting ego graph")
    ego_graph_extractor = EgoGraphExtractor(model, pyg_dataset, args.k, args.hops,
                                            meta_data, device)
    for i, (pos_ego_graphs, neg_ego_graphs, uncertain_ego_graphs) in enumerate(ego_graph_extractor.extract()):
        for j, ego_graph in enumerate(pos_ego_graphs):
            nx.write_gpickle(ego_graph, os.path.join(save_path, f"pos_{i}_{j}.gpickle"))
            nx.drawing.nx_pydot.write_dot(ego_graph, os.path.join(save_path, f'pos_{i}_{j}.dot'))
        for j, ego_graph in enumerate(neg_ego_graphs):
            nx.write_gpickle(ego_graph, os.path.join(save_path, f"neg_{i}_{j}.gpickle"))
            nx.drawing.nx_pydot.write_dot(ego_graph, os.path.join(save_path, f'neg_{i}_{j}.dot'))
        for j, ego_graph in enumerate(uncertain_ego_graphs):
            nx.write_gpickle(ego_graph, os.path.join(save_path, f"uncertain_{i}_{j}.gpickle"))
            nx.drawing.nx_pydot.write_dot(ego_graph, os.path.join(save_path, f'uncertain_{i}_{j}.dot'))


if __name__ == "__main__":
    main()
