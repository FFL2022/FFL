import torch
import torch_scatter
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch_geometric.utils

# device = torch.device('cpu')
class GCNLayer(MessagePassing):
    def __init__(self, in_feat, out_feat, aggr='mean'):
        super().__init__(aggr=aggr)
        self.linear = nn.Linear(in_feat, out_feat)
        self.relu = nn.ReLU()

    def forward(self, x, edge_idx, weights=None):
        x1 = self.linear(x)
        x = self.propagate(edge_idx, x=x1, weights=weights)
        return self.relu(x)

    def message(self, x_j, weights=None):
        if weights is not None:
            return  x_j * weights
        return x_j


class MPNNLayerNode(MessagePassing):
    def __init__(self, n_in_feat, n_out_feat, aggr='max'):
        super().__init__(aggr=aggr)
        self.linear_src = nn.Linear(n_in_feat, n_out_feat)
        self.linear_dst = nn.Linear(n_in_feat, n_out_feat)
        self.n_out_feat = n_out_feat
        self.linear_u = nn.Linear(n_in_feat+n_out_feat, n_out_feat)
        self.relu = nn.ReLU()

    def forward(self, x, edge_idx, weights=None):
        x_j = self.linear_src(x)
        x_i = self.linear_dst(x)
        x1 = torch.cat([x_i, x_j], dim=-1)
        x1 = self.propagate(edge_idx, x=x1, weights=weights)
        x = torch.cat([x1, x], dim=-1)
        return self.linear_u(x)

    def message(self, x_i, x_j, weights):
        m = (x_i[:, self.n_out_feat:] + x_j[:, :self.n_out_feat])
        if weights is not None:
            return m * weights
        return m


class MPNNLayerNodeMulti(MessagePassing):
    def __init__(self, n_in_feat, n_out_feat,
                 n_layer_message=1,
                 n_layer_update=1,
                 aggr='max'):
        super().__init__(aggr=aggr)
        self.linear_src = nn.Linear(n_in_feat, n_out_feat)
        self.linear_dst = nn.Linear(n_in_feat, n_out_feat)
        self.n_layer_message = n_layer_message
        self.linear_message = nn.ModuleList(
            [
                nn.Linear(n_out_feat, n_out_feat)
                for _ in range(n_layer_message)
            ]
        )
        self.n_out_feat = n_out_feat
        self.n_layer_update = n_layer_update
        self.linear_u = nn.ModuleList([
            nn.Linear(n_out_feat, n_out_feat)
            for _ in range(n_layer_update)])
        self.relu = nn.ReLU()

    def forward(self, x, edge_idx, weights=None):
        x1 = self.propagate(edge_idx, x=x, weights=weights)
        for i in range(self.n_layer_update):
            x1 = self.linear_u[i](self.relu(x1))
        return x1

    def message(self, x_i, x_j, weights):
        m = self.linear_src(x_i) + self.linear_dst(x_j)
        if weights is not None:
            m = weights * m
        for i in range(self.n_layer_message):
            m = self.relu(m)
            m = self.linear_message[i](m)
            if weights is not None:
                m = weights * m
        return m


class MPNNLayer(MessagePassing):
    def __init__(self, n_in_feat, e_feat, n_out_feat, aggr='max'):
        super().__init__(aggr=aggr)
        self.linear_src = nn.Linear(n_in_feat, n_out_feat)
        self.linear_dst = nn.Linear(n_in_feat, n_out_feat)
        self.linear_edge = nn.Linear(e_feat, n_out_feat)
        self.n_out_feat = n_out_feat
        self.linear_u = nn.Linear(n_in_feat+n_out_feat, n_out_feat)
        self.relu = nn.ReLU()

    def forward(self, x, edge_idx, e_feat=None, weights=None):
        e_feat_updated = self.linear_edge(e_feat)
        x_j = self.linear_src(x)
        x_i = self.linear_dst(x)
        x1 = torch.cat([x_i, x_j], dim=-1)
        x1 = self.propagate(edge_idx, x=x1, e_feat=e_feat_updated,
            weights=weights)
        x = torch.cat([x1, x], dim=-1)
        return self.linear_u(x)


    def message(self, x_i, x_j, e_feat, weights):
        m = (x_i[:, self.n_out_feat:] + x_j[:, :self.n_out_feat] + e_feat)
        if weights is not None:
            return m * weights
        return m


class MPNNLayerBipartite(MessagePassing):
    def __init__(self, hfeats, efeats, ofeats, aggr='max'):
        pass
