import dgl
import torch
import torch.nn as nn

__author__ = "Marc: thanhdatn@student.unimelb.edu.au"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MPNNBlockSingleEdge(torch.nn.Module):
    def __init__(self, hidden_feats, edge_feats, out_feats,
                 device=device,
                 out_edge_feats=None,
                 aggregator_type='max', bias=True, norm=None,
                 activation=None):
        super(MPNNBlockSingleEdge, self).__init__()
        # Create weight for each edge
        self.Ms_u = torch.nn.Linear(hidden_feats, out_feats)
        self.Ms_v = torch.nn.Linear(hidden_feats, out_feats)
        # Modulation by edge feats ?
        self.Ms_ue = torch.nn.Linear(edge_feats, hidden_feats)
        self.Ms_ve = torch.nn.Linear(edge_feats, hidden_feats)
        self.M = torch.nn.Linear(hidden_feats, out_feats)

        self.activation = torch.nn.ReLU()
        if aggregator_type == 'max':
            self.aggregator = dgl.function.max
        elif aggregator_type == 'sum':
            self.aggregator = dgl.function.sum

        elif aggregator_type == 'mean':
            self.aggregator = dgl.function.mean
        self.to(device)
        self.device = device
        self.edge_feats = edge_feats
        self.hidden_feats = hidden_feats
        self.out_feats = out_feats
        # self.U = torch.nn.Linear(hidden_feats*2, hidden_feats)
        # self.decoder = torch.nn.Linear(hidden_feats * 2, out_feats)
        # Termination is only for other
        # TODO: Add edge type "None" for case of no-edges ?

    def construct_module_list(self, in_dim, out_dim):
        out_list = []
        for etype in self.edge_types:
            out_list.append(torch.nn.Linear(in_dim, out_dim))
        return out_list

    def compute_send_messages(self, edges):
        x_src = edges.src['h']
        x_dst = edges.dst['h']
        e_emb = edges.data['h']

        # Scenario 1: +
        # Scenario 2: Correlation
        edge_src_weight = self.Ms_ue(e_emb)     # -> |E| x f_e -> |E| x h_f
        edge_dst_weight = self.Ms_ve(e_emb)     # -> |E| x f_e -> |E| x h_f
        out = self.Ms_u(edge_src_weight + x_src) + self.Ms_v(edge_dst_weight
                                                             + x_dst)
        # Filter each type of edge
        return {'msg': out}

    def activate_node(self, nodes, name_in, name_out):
        return {name_out: self.activation(nodes.data[name_in])}

    def apply_nodes(self, nodes):
        return {'h1': self.M(nodes.data['h'])}

    def sum_h_h1(self, nodes):
        return {'h': nodes.data['h'] + nodes.data['h1']}

    def forward(self, g: dgl.DGLGraph):
        g.apply_nodes(self.apply_nodes)
        if g.number_of_edges() > 0:
            # g.apply_edges(self.compute_send_messages)
            # Hypo 1: aggregate MLP + max + Relu
            g.update_all(
                self.compute_send_messages,
                self.aggregator('msg', 'h'),
                )
            g.apply_nodes(self.sum_h_h1)
            g.apply_nodes(
                lambda nodes: self.activate_node(nodes, 'h', 'h'))
        else:
            g.apply_nodes(
                lambda nodes: self.activate_node(nodes, 'h1', 'h'))

        return g


class MPNNBlockMultSingleEtype(torch.nn.Module):
    def __init__(self, hidden_dim, out_dim, activation=nn.ReLU()):
        super(MPNNBlockMultSingleEtype, self).__init__()
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.edge_transform = nn.Linear(hidden_dim, out_dim,
                                        bias=False)
        # self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        self.aggregator = dgl.function.mean
        self.out_dim = out_dim
        self.self_loop = nn.Linear(hidden_dim, out_dim)
        # TODO: add self loop to: each type of edge, add linear, etc
        # TODO: sigmoid weight

    def compute_send_messages(self, edges):
        x_src = edges.src['h']  # N_n, hidden_dim
        msg = self.edge_transform(x_src)
        return {'msg': msg}

    def activate_node(self, nodes, name_in, name_out):
        return {name_out: self.activation(nodes.data[name_in])}

    def add_self_loop(self, nodes):
        if 'h1' in nodes.data:
            return {'h': self.self_loop(nodes.data['h']) + nodes.data['h1']}
        return {'h': self.self_loop(nodes.data['h'])}

    def forward(self, g: dgl.DGLGraph):
        if g.number_of_edges() > 0:
            g.update_all(lambda edges: self.compute_send_messages(edges),
                         self.aggregator('msg', 'h1'),
                         )
        g.apply_nodes(self.add_self_loop)
        g.apply_nodes(lambda nodes: self.activate_node(nodes, 'h', 'h'))
        return g



class HeteroMPNNBlockSimp(torch.nn.Module):
    ''' Propagate infor through each type of edge'''
    def __init__(self, meta_graph,
                 hidden_dim, hidden_edim,
                 out_dim,
                 device=device):
        super(HeteroMPNNBlockSimp, self).__init__()
        # 1. Get all edges via meta graph
        self.meta_graph = meta_graph
        per_type_linear = {}
        self.funcs = {}
        for c_etype in self.meta_graph:
            # etype is a tuple of node type, etype, dst type
            t_src, t_e, t_dst = c_etype
            # 2. for each meta graph, create a mpnn block
            per_type_linear[t_e] = MPNNBlockMultSingleEtype(hidden_dim,
                                                 hidden_edim, out_dim)
            self.funcs[t_e] = (per_type_linear[t_e].compute_send_messages,
                               per_type_linear[t_e].aggregator('msg', 'h'))

        self.per_type_linear = torch.nn.ModuleDict(per_type_linear)

    def forward(self, h_g):
        # 4. Passing message through each of these sub graph onces each
        # TODO: Beware of gradient explodes
        temp_func = {}
        for c_etype in self.meta_graph:
            if h_g.number_of_edges(c_etype[1]) > 0:
                temp_func[c_etype[1]] = self.funcs[c_etype[1]]
            else:
                temp_func[c_etype[1]] = (lambda x: {}, lambda x: {})
        h_g.multi_update_all(temp_func, 'sum')
        return h_g