from codeflaws.dataloader_cfl import CodeflawsCFLNxStatementDataset, \
    CodeflawsCFLStatementGraphMetadata
from pyg_version.codeflaws.dataloader_cfl_pyg import CodeflawsCFLPyGStatementDataset, CodeflawsCFLStatementGraphMetadata
from pyg_version.model import MPNNModel_A_T_L
from pyg_version.explainer.explainer import Explainer, InflExtractor
from pyg_version.explainer.common import entropy_loss_mask, size_loss
from graph_algos.nx_shortcuts import where_node, where_edge
# Todo: use edge weight and feature weight balanced
import torch
import networkx as nx
import os
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def target_node_loss(perturbed_pred, orig_pred, _, instance):
    # Perturbed pred is of size N x class
    # orig_pred is of size N x class
    # instance is a tuple of (data, target)
    # data is Data object, where xs is of size N x F, and ess is of size [L x 2 x E], L is the number of edge type, determined via the graph metadata
    # target is a node in the ast

    # Get the target node
    target_node = instance[1]
    perturbed_pred = perturbed_pred[1][target_node]
    orig_pred = orig_pred[1][target_node]
    return torch.nn.functional.binary_cross_entropy_with_logits(
        perturbed_pred, orig_pred)


def target_statement_loss(perturbed_pred, orig_pred, _, instance):
    # Perturbed pred is of size N x class
    # orig_pred is of size N x class
    # instance is a tuple of (data, target)
    # data is Data object, where xs is of size N x F, and ess is of size [L x 2 x E], L is the number of edge type, determined via the graph metadata
    # target is an integer of line number
    (data, stmt_nodes), target = instance
    # target is the target statement
    orig_pred_stmt = orig_pred[1][stmt_nodes, 1]
    # get the max index
    perturbed_pred_stmt = perturbed_pred[1][stmt_nodes, 1]
    return torch.nn.functional.binary_cross_entropy_with_logits(
        perturbed_pred_stmt, orig_pred_stmt)


def total_loss_size_stmt_entropy(perturbed_pred, orig_pred, perturber,
                                 instance):
    # Perturbed pred is of size N x class
    # orig_pred is of size N x class
    # instance is a tuple of (data, target)
    # data is Data object, where xs is of size N x F, and ess is of size [L x 2 x E],
    # L is the number of edge type, determined via the graph metadata
    # target is an integer of line number
    (data, stmt_nodes), target = instance
    # target is the target statement
    stmt_loss = target_statement_loss(perturbed_pred, orig_pred, perturber,
                                      instance)
    size_loss_val = size_loss(torch.sigmoid(perturber.get_node_weights()),
                              torch.sigmoid(perturber.get_edge_weights()),
                              coeff_n=0.002,
                              coeff_e=0.005)
    entropy_loss = entropy_loss_mask(
        torch.sigmoid(perturber.get_node_weights()),
        torch.sigmoid(perturber.get_edge_weights()),
        coeff_n=0.1,
        coeff_e=0.3)
    return stmt_loss + size_loss_val + entropy_loss


def total_loss_size_stmt_entropy_edge_only(perturbed_pred, orig_pred,
                                           perturber, instance):
    # Perturbed pred is of size N x class
    # orig_pred is of size N x class
    # instance is a tuple of (data, target)
    # data is Data object, where xs is of size N x F, and ess is of size [L x 2 x E],
    # L is the number of edge type, determined via the graph metadata
    # target is an integer of line number
    (data, stmt_nodes), target = instance
    # target is the target statement
    stmt_loss = target_statement_loss(perturbed_pred, orig_pred, perturber,
                                      instance)
    size_loss_val = size_loss(torch.sigmoid(perturber.get_edge_weights()),
                              torch.sigmoid(perturber.get_edge_weights()),
                              coeff_n=0.002,
                              coeff_e=0.005)
    entropy_loss = entropy_loss_mask(
        torch.sigmoid(perturber.get_edge_weights()),
        torch.sigmoid(perturber.get_edge_weights()),
        coeff_n=0.1,
        coeff_e=0.3)
    return stmt_loss + size_loss_val + entropy_loss


class TopKStatementIterator(object):

    def __init__(self, model, dataset: CodeflawsCFLPyGStatementDataset, k,
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
                topk = output.topk(k, dim=0)[1]
                for i in topk:
                    yield (data, stmt_nodes.to(device)), i

    def __len__(self):
        return self.len_data


class StatementGraphPerturber(torch.nn.Module):

    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.xs_weights = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones(x.shape[0], 1, requires_grad=True))
            for x in graph.xs
        ])
        self.ess_weights = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones(e.shape[1], 1, requires_grad=True))
            for e in graph.ess
        ])

    def get_node_weights(self):
        # stack all self weights
        node_weights = [weight for weight in self.xs_weights]
        return torch.cat(node_weights, dim=0)

    def get_edge_weights(self):
        edge_weights = [weight for weight in self.ess_weights]
        return torch.cat(edge_weights, dim=0)

    def forward(self, data):
        return data.xs, data.ess, self.xs_weights, self.ess_weights


class StatementGraphPerturberEdgeOnly(torch.nn.Module):

    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.xs_weights = None
        self.ess_weights = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones(e.shape[1], 1, requires_grad=True))
            for e in graph.ess
        ])
        self.nnodes = sum([x.shape[0] for x in graph.xs])

    def get_node_weights(self):
        # Return 0
        return torch.zeros(self.nnodes, 1)

    def get_edge_weights(self):
        edge_weights = [weight for weight in self.ess_weights]
        return torch.cat(edge_weights, dim=0)

    def forward(self, data):
        return data.xs, data.ess, None, self.ess_weights


class TopKStatmentExplainer(Explainer):

    def __init__(self, model, loss_func,
                 dataset: CodeflawsCFLPyGStatementDataset, k, device):
        super(TopKStatmentExplainer, self).__init__(model, loss_func, 3000)
        self.iterator = TopKStatementIterator(model, dataset, k, 3000)

    def get_data(self, instance):
        return instance[0]

    def data_to_model(self, data):
        return data[0].xs, data[0].ess

    def get_perturber(self, data) -> torch.nn.Module:
        perturber = StatementGraphPerturber(data[0]).to(device)
        perturber.train()
        return perturber

    def explain(self):
        return super().explain(self.iterator)

    def data_to_perturber(self, data):
        return data[0]


class TopKStatmentExplainerEdge(Explainer):

    def __init__(self, model, loss_func,
                 dataset: CodeflawsCFLPyGStatementDataset, k, device):
        super(TopKStatmentExplainerEdge, self).__init__(model, loss_func, 3000)
        self.iterator = TopKStatementIterator(model, dataset, k, 3000)

    def get_data(self, instance):
        return instance[0]

    def data_to_model(self, data):
        return data[0].xs, data[0].ess

    def get_perturber(self, data) -> torch.nn.Module:
        perturber = StatementGraphPerturberEdgeOnly(data[0]).to(device)
        perturber.train()
        return perturber

    def explain(self):
        return super().explain(self.iterator)

    def data_to_perturber(self, data):
        return data[0]


def from_data_to_nx(graph, perturber: StatementGraphPerturber,
                    metadata: CodeflawsCFLStatementGraphMetadata):
    g = nx.MultiDiGraph()
    for i, x in enumerate(graph.xs):
        x = x.reshape(-1)
        if i == 0:
            # Then the node graph is ast
            for j, node in enumerate(x):
                g.add_node(f"ast_{j}",
                           ntype=metadata.id2ntype[int(x[j].item())],
                           label=metadata.id2ntype[int(x[j].item())],
                           explain_weight=perturber.xs_weights[i][j].item() if perturber.xs_weights else 0)
        elif i == 1:
            # Then the node graph is test
            for j, node in enumerate(x):
                g.add_node(f"test_{j}",
                           ntype=metadata.t_tests[int(x[j].item())],
                           label=metadata.id2ntype[int(x[j].item())],
                           explain_weight=perturber.xs_weights[i][j].item() if perturber.xs_weights else 0)
        # each row of x is a data of a node
    # Translate from each edge type to the corresponding edge
    for i, es in enumerate(graph.ess):
        # i is the ith etype
        src_type, etype, dst_type = metadata.meta_graph[i]
        for j in range(es.shape[1]):
            src_node = f"{src_type}_{es[0, j].item()}"
            dst_node = f"{dst_type}_{es[1, j].item()}"
            g.add_edge(src_node,
                       dst_node,
                       etype=etype,
                       label=etype,
                       explain_weight=perturber.ess_weights[i][j].item())
    return g


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path",
                        type=str,
                        default="explain_pyg_codeflaws_pyc_cfl_stmt_level")
    parser.add_argument("--k", type=int, default=10)
    # parser.add_argument("--loss_func",
    #                     type=str,
    #                     default="total_loss_size_stmt_entropy")

    parser.add_argument("--loss_func",
                        type=str,
                        default="total_loss_size_stmt_entropy_edge_only")
    parser.add_argument("--device", type=str, default='cuda')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    nx_dataset = CodeflawsCFLNxStatementDataset()
    meta_data = CodeflawsCFLStatementGraphMetadata(nx_dataset)
    pyg_dataset = CodeflawsCFLPyGStatementDataset(dataloader=nx_dataset,
                                                  meta_data=meta_data,
                                                  ast_enc=None,
                                                  name='train_pyg_cfl_stmt')
    t2id = {'ast': 0, 'test': 1}
    exec(f"loss_func = {args.loss_func}", globals(), locals())
    infl_extractor = InflExtractor(where_node(),
                                   where_edge(where_node(), where_node()), 0.3,
                                   0.1)
    model = MPNNModel_A_T_L(dim_h=64,
                            netypes=len(meta_data.meta_graph),
                            t_srcs=[t2id[e[0]] for e in meta_data.meta_graph],
                            t_tgts=[t2id[e[2]] for e in meta_data.meta_graph],
                            n_al=len(meta_data.t_asts),
                            n_layers=5,
                            n_classes=2).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    explainer = TopKStatmentExplainerEdge(model, loss_func, pyg_dataset,
                                          args.k, args.device)
    save_dir = args.save_path
    os.makedirs(save_dir, exist_ok=True)
    for i, (((graph, stmt_nodes), target_stmt_idx),
            perturber) in enumerate(explainer.explain()):
        target_node = f"ast_{stmt_nodes[target_stmt_idx].item()}"
        g = from_data_to_nx(graph, perturber, meta_data)
        infls = infl_extractor.extract_infl_structure(g, target_node)
        # Dump this g
        nx.write_gpickle(infls, os.path.join(save_dir, f"{i}.gpickle"))
        # Visualize with dot
        nx.drawing.nx_pydot.write_dot(infls,
                                      os.path.join(save_dir, f"{i}.dot"))
