from __future__ import print_function, unicode_literals
import torch.nn.functional as F
import torch.nn as nn
import torch
import dgl
import dgl.function as fn
from layers.sparsemax import Sparsemax
from layers.layers_mult import MPNNEncBlockNodeOnly, MPNNBlockLabelOnly, \
    MPNNDecBlockNode


def dense_to_sparse(adj):
    r"""Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes. - Copied from pytorch geometric

    Args:
        adj (Tensor): The dense adjacency matrix.
     :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)

    index = adj.nonzero(as_tuple=True)
    edge_attr = adj[index]

    if len(index) == 3:
        batch = index[0] * adj.size(-1)
        index = (batch + index[1], batch + index[2])

    return torch.stack(index, dim=0), edge_attr


class NodeInformationScore(nn.Module):
    def __init__(self, cached=False, **kwargs):
        super(NodeInformationScore, self).__init__()
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None
        self.aggregator = dgl.function.mean

    def compute_send_message(self, edges):
        x_src = edges.src['h']  # N_e, hidden_dim
        eweights = edges.data['weights'].unsqueeze(-1)    # N_e, 1
        return {'msg': x_src * eweights}

    def cal_node_info(self, nodes):
        ninfo = nodes.data['info']
        return {'info': torch.abs(torch.sum(ninfo, -1))}

    def forward(self, g):
        # Normal mean, but
        g.update_all(self.compute_send_message,
                     self.aggregator('msg', 'h1'))
        g.apply_nodes(lambda nodes: {'info': nodes.data['h'] - nodes.data['h1']})
        g.apply_nodes(self.cal_node_info)


class HGPSLPool(nn.Module):

    def __init__(self, dim, num_etypes,
                 ratio=0.8, sl=True, lamb=1.0, negative_slop=0.2):
        '''
        :param sl: structure learning
        '''
        super(HGPSLPool, self).__init__()
        self.ratio = ratio
        self.negative_slop = negative_slop
        self.lamb = torch.Tensor([lamb])
        if isinstance(lamb, float):
            self.lamb = self.lamb.repeat(num_etypes)
            self.lamb = self.lamb.unsqueeze(-1)
            self.lamb = self.lamb.unsqueeze(-1)
        else:
            self.lamb = torch.unsqueeze(self.lamb, -1)
        self.calc_node_info = NodeInformationScore()
        self.att = nn.Parameter(torch.Tensor(1, dim * 2))
        self.num_etypes = num_etypes
        self.sparse_attention = Sparsemax(-1)

    def forward(self, g):
        '''
        Return kept indices
        '''
        # First, calculcate the node information
        self.calc_node_info(g)
        # Sort nodes top down according to info
        _, idxs = torch.sort(g.ndata['info'], descending=True)
        # Take top k nodes
        # To retrieve old node names: 'dgl.NID' and 'dgl.EID'
        sub_g = g.subgraph(idxs[:int(self.ratio*g.number_of_nodes())],
                             store_ids=True)
        last_eid = g.number_of_edges()

        # Structure learning
        # Construct a new full connected graph
        num_nodes = sub_g.number_of_nodes()
        new_adj = torch.ones((num_nodes, num_nodes), device=sub_g.device)
        new_edge_index, _ = dense_to_sparse(new_adj)

        row, col = new_edge_index
        weights = (torch.cat([sub_g.ndata['h'][row],
                              sub_g.ndata['h'][col]], dim=1) * self.att).sum(dim=-1)
        weights = F.leaky_relu(weights, self.negative_slop)
        new_adj[row, col] = weights

        # Copy out each edge type
        all_adjs = torch.stack([torch.zeros(num_nodes, num_nodes).to(
            sub_g.device) for etype in range(self.num_etypes)],
            dim=0
        ).to(g.device)  # etype, A
        for etype in range(self.num_etypes):
            mask = sub_g.edata['etype'] == etype
            sub_rows, sub_cols = sub_g.all_edges()
            sub_rows = sub_rows[mask]
            sub_cols = sub_cols[mask]
            all_adjs[etype, sub_rows, sub_cols] = sub_g.edata['weights'][mask]

        self.lamb = self.lamb.to(g.device)

        new_adj = new_adj + torch.sum(all_adjs * self.lamb, 0)
        # Sparsemax
        '''
        new_edge_attr = self.sparse_attention(new_adj, row)
        new_adj[row, col] = new_edge_attr
        new_edge_index, new_edge_attr = dense_to_sparse(new_adj)
        '''
        new_edge_attr = self.sparse_attention(new_adj)    # N N
        new_edge_index, new_edge_attr = dense_to_sparse(new_adj)
        row, col = new_edge_index
        # release gpu mem
        del new_adj
        torch.cuda.empty_cache()

        # Add these new edges
        new_eid = torch.arange(last_eid, last_eid + row.shape[0], 1).long().to(sub_g.device)
        new_etype = torch.ones(row.shape[0]).long().to(sub_g.device)* self.num_etypes
        sub_g.add_edges(row, col, data={'_ID': new_eid,
                                        'weights': new_edge_attr,
                                        'etype': new_etype})
        return sub_g


class NodeUnpool(torch.nn.Module):
    def __init__(self, dim, num_etypes, activation=nn.ReLU()):
        super(NodeUnpool, self).__init__()
        self.num_etypes = num_etypes
        self.merger1 = nn.Linear(dim, dim)
        self.merger2 = nn.Linear(dim, dim)
        self.activation = activation

    def forward(self, g, sub_g):
        # Find corresponding nodes on sub_g
        old_idxs = sub_g.ndata['_ID']
        old_data = g.ndata['h'][old_idxs]
        new_data = sub_g.ndata['h']
        g.ndata['h'][old_idxs] = self.merger1(old_data) + self.merger2(new_data)
        return g


class ResGCNHgpPool(torch.nn.Module):
    def __init__(self, dim, num_etypes):
        super(ResGCNHgpPool, self).__init__()
        self.mpnn1 = MPNNBlockLabelOnly(dim, dim, num_etypes)
        self.mpnn2 = MPNNBlockLabelOnly(dim, dim, num_etypes)
        self.pool = HGPSLPool(dim, num_etypes)

    def forward(self, g):
        data = g.ndata['h']
        g = self.mpnn1(g)
        # Might be better if we leave this here, since in the original
        # architecture, they need one GCN layer before pool
        g.ndata['h'] = data + g.ndata['h']
        g = self.mpnn2(g)
        sub_g = self.pool(g)
        return g, sub_g


class GCNResUnpool(torch.nn.Module):
    def __init__(self, dim, num_etypes):
        super(GCNResUnpool, self).__init__()
        self.mpnn1 = MPNNBlockLabelOnly(dim, dim, num_etypes+ 1)
        self.mpnn2 = MPNNBlockLabelOnly(dim, dim, num_etypes)
        self.unpool = NodeUnpool(dim, num_etypes)

    def forward(self, g, sub_g):
        g = self.unpool(g, sub_g)
        sub_g = self.mpnn1(sub_g)
        data = g.ndata['h']
        g = self.mpnn2(g)
        g.ndata['h'] = data + g.ndata['h']
        return g


class HGP_SL_GraphUNet(torch.nn.Module):
    def __init__(self, num_feats, num_classes, g_hidden_dim, num_etypes,
                 add_default_eweight=True, add_default_nweight=True):
        super(HGP_SL_GraphUNet, self).__init__()
        self.encoder = MPNNEncBlockNodeOnly(
            num_feats, g_hidden_dim,
            add_default_eweight=add_default_eweight,
            add_default_nweight=add_default_nweight)
        self.gcn_pool1 = ResGCNHgpPool(g_hidden_dim, num_etypes)
        '''
        self.gcn_pool2 = ResGCNHgpPool(g_hidden_dim, num_etypes + 1)
        self.gcn_unpool2 = GCNResUnpool(g_hidden_dim, num_etypes + 1)
        '''
        self.mpnn = MPNNBlockLabelOnly(g_hidden_dim, g_hidden_dim, num_etypes)
        self.gcn_unpool1 = GCNResUnpool(g_hidden_dim, num_etypes)
        self.decoder = MPNNDecBlockNode(g_hidden_dim, num_classes)

    def forward(self, g):
        g = self.encoder(g)
        g, sub_g1 = self.gcn_pool1(g)
        # Some how: 2 pool and 2 unpool wont work
        # sub_g1, sub_g2 = self.gcn_pool2(sub_g1)
        sub_g1 = self.mpnn(sub_g1)
        # unsub_g2 = self.gcn_unpool2(sub_g1, sub_g2)
        unsub_g1 = self.gcn_unpool1(g, sub_g1)
        return self.decoder(g)
