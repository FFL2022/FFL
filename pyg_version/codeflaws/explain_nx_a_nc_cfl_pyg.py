from codeflaws.dataloader_cfl import CodeflawsCFLNxStatementDataset, \
    CodeflawsCFLStatementGraphMetadata
from pyg_version.codeflaws.dataloader_cfl_pyg import CodeflawsCFLPyGStatementDataset, CodeflawsCFLStatementGraphMetadata
from pyg_version.model import MPNNModel_A_T_L
from pyg_version.explainer.explainer import Explainer, InflExtractor
from pyg_version.explainer.common import EdgeWeights, NodeWeights
# Todo: use edge weight and feature weight balanced
import torch
import networkx as nx
import os

def target_node_loss(perturbed_pred, orig_pred, _, instance):
    # Perturbed pred is of size N x class
    # orig_pred is of size N x class
    # instance is a tuple of (data, target)
    # data is Data object, where xs is of size N x F, and ess is of size [L x 2 x E], L is the number of edge type, determined via the graph metadata
    # target is a node in the ast


    # Get the target node
    target_node = instance[1]
    perturbed_pred = perturbed_pred[target_node]
    orig_pred = orig_pred[target_node]
    return torch.nn.functional.binary_cross_entropy_with_logits(perturbed_pred, orig_pred)


def target_statement_loss(perturbed_pred, orig_pred, _, instance):
    # Perturbed pred is of size N x class
    # orig_pred is of size N x class
    # instance is a tuple of (data, target)
    # data is Data object, where xs is of size N x F, and ess is of size [L x 2 x E], L is the number of edge type, determined via the graph metadata
    # target is an integer of line number
    (data, stmt_nodes), target = instance
    # target is the target statement
    orig_pred_stmt = orig_pred[stmt_nodes, 1].detach().cpu()
    perturbed_pred_stmt = perturbed_pred[stmt_nodes, 1].detach().cpu()
    return torch.nn.functional.binary_cross_entropy_with_logits(perturbed_pred_stmt, orig_pred_stmt)



class TopKStatementIterator(object):
    def __init__(self, model, dataset: CodeflawsCFLPyGStatementDataset, k, device):
        self.model = model
        self.dataset = dataset
        self.k = k
        self.len_data = self.calc_len()
        self.device = device

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
                data = data.to(self.device)
                output = self.model(data)
                output = output.cpu()
                topk = output.topk(self.k, dim=0)
                for i in topk.indices:
                    yield data, i

    def __len__(self):
        return self.len_data


class StatementGraphPerturber(torch.nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.xs_weights = list([torch.nn.Parameter(torch.ones(x.shape[0], x.shape[1])) for x in graph.xs])
        self.ess_weights = list([torch.nn.Parameter(torch.ones(e.shape[0], e.shape[1])) for e in graph.ess])

    def forward(self, data):
        return data, self.xs_weights, self.ess_weights


class TopKStatmentExplainer(Explainer):
    def __init__(self, model, loss_func, dataset: CodeflawsCFLPyGStatementDataset, k, device):
        super(TopKStatmentExplainer, self).__init__(model, loss_func, dataset, device)
        self.iterator = TopKStatementIterator(model, dataset, k, device)

    def get_data(self, instance):
        return instance[0]

    def get_perturber(self, data) -> torch.nn.Module:
        return StatementGraphPerturber(data)

    def explain(self):
        return super().explain(self.iterator)


def from_data_to_nx(data, perturber: StatementGraphPerturber, metadata: CodeflawsCFLStatementGraphMetadata):
    g = nx.MultiDiGraph()
    for i, x in enumerate(data.xs):
        x = x.reshape(-1)
        if i == 0:
            # Then the node graph is ast
            for j, node in enumerate(x):
                g.add_node(f"ast_{j}", ntype=metadata.t_asts[int(x[j].item())], explain_weight=perturber.xs_weights[i][j].item())
        elif i == 1:
            # Then the node graph is test
            for j, node in enumerate(x):
                g.add_node(f"test_{j}", ntype=metadata.t_tests[int(x[j].item())], explain_weight=perturber.xs_weights[i][j].item())
        # each row of x is a data of a node
    # Translate from each edge type to the corresponding edge
    for i, es in enumerate(data.ess):
        # i is the ith etype
        src_type, etype, dst_type = metadata.meta_graph
        for j in range(es.shape[1]):
            src_node = f"{src_type}_{es[0, j].item()}"
            dst_node = f"{dst_type}_{es[1, j].item()}"
            g.add_edge(src_node, dst_node, etype=etype[i], explain_weight=perturber.ess_weights[i][j].item())
    return g

if __name__ == '__main__':
    nx_dataset = CodeflawsCFLNxStatementDataset()
    pyg_dataset = CodeflawsCFLPyGStatementDataset()
    graph_metadata = CodeflawsCFLStatementGraphMetadata()
    model = MPNNModel_A_T_L(graph_metadata, 2)
    explainer = TopKStatmentExplainer(model, target_statement_loss, 'test', pyg_dataset, 5, 'cpu')
    save_dir = "explain_pyg_codeflaws_pyc_cfl_stmt_level"
    os.makedirs(save_dir, exist_ok=True)
    for perturber in explainer.explain():
        data = perturber.get_data()
        g = from_data_to_nx(data, perturber)
        # Dump this g
        nx.write_gpickle(g, os.path.join(save_dir, f"{data.id}.gpickle"))
        # Visualize with dot
        nx.drawing.nx_pydot.write_dot(g, os.path.join(save_dir, f"{data.id}.dot"))
