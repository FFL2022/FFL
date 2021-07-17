from __future__ import print_function, unicode_literals
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.attention import SparsemaxNodeAttention, SoftmaxNodeAttention
from layers.hgp_sp_with_unpool import ResGCNHgpPool, GCNResUnpool
from layers.layers_mult import GCNBlock


class HGP_SL_GraphUNet(torch.nn.Module):
    def __init__(self, g_hidden_dim, num_etypes, n_layers=1):
        super(HGP_SL_GraphUNet, self).__init__()
        self.n_layers = n_layers
        self.gcn_pools = torch.nn.ModuleList(
            [ResGCNHgpPool(g_hidden_dim, num_etypes + i)
             for i in range(n_layers)])
        self.gcn_unpools = torch.nn.ModuleList(
            [GCNResUnpool(g_hidden_dim, num_etypes + i)
             for i in range(n_layers)])
        self.pool_gcns = torch.nn.ModuleList([
            GCNBlock(g_hidden_dim, g_hidden_dim, num_etypes + i + 1)
            for i in range(n_layers)])

    def forward(self, g):
        parents = []
        pools = []
        for i in range(self.n_layers):
            g, sub_g1 = self.gcn_pools[i](g)
            sub_g1 = self.pool_gcns[i](sub_g1)
            pools.append(sub_g1)
            parents.append(g)
        for i in range(self.n_layers)[::-1]:
            g = self.gcn_unpools[i](parents[i], pools[i])
        return g


class HGP_SL_Dual_GraphUNet(torch.nn.Module):
    def __init__(self, g_hidden_dim, doc_num_etypes, vc_num_etypes, n_layers=1):
        super(HGP_SL_Dual_GraphUNet, self).__init__()
        self.n_layers = n_layers

        self.doc_gcn_pools = torch.nn.ModuleList([ResGCNHgpPool(g_hidden_dim, doc_num_etypes + i) for i in range(n_layers)])
        self.doc_gcn_unpools = torch.nn.ModuleList([GCNResUnpool(g_hidden_dim, doc_num_etypes + i) for i in range(n_layers)])
        self.doc_pool_gcns = torch.nn.ModuleList([GCNBlock(g_hidden_dim, g_hidden_dim, doc_num_etypes + i + 1) for i in range(n_layers)])

        self.vc_gcn_pools = torch.nn.ModuleList([ResGCNHgpPool(g_hidden_dim, vc_num_etypes + i) for i in range(n_layers)])
        self.vc_gcn_unpools = torch.nn.ModuleList([GCNResUnpool(g_hidden_dim, vc_num_etypes + i) for i in range(n_layers)])
        self.vc_pool_gcns = torch.nn.ModuleList([GCNBlock(g_hidden_dim, g_hidden_dim, vc_num_etypes + i + 1) for i in range(n_layers)])

        self.doc_node_attn = torch.nn.ModuleList([SparsemaxNodeAttention(g_hidden_dim, g_hidden_dim, qkv_bias=False) for i in range(n_layers)])
        self.vc_node_attn = torch.nn.ModuleList([SparsemaxNodeAttention(g_hidden_dim, g_hidden_dim, qkv_bias=False) for i in range(n_layers)])

        self.doc_cross_attn = torch.nn.ModuleList([nn.MultiheadAttention(embed_dim=g_hidden_dim, num_heads=8) for i in range(n_layers)])
        self.vc_cross_attn = torch.nn.ModuleList([nn.MultiheadAttention(embed_dim=g_hidden_dim, num_heads=8) for i in range(n_layers)])

    def forward(self, doc_g, vc_g):
        doc_parents = []
        doc_pools = []

        vc_parents = []
        vc_pools = []

        for i in range(self.n_layers):
            doc_g, doc_sub_g1 = self.doc_gcn_pools[i](doc_g)
            doc_sub_g1 = self.doc_pool_gcns[i](doc_sub_g1)

            vc_g, vc_sub_g1 = self.vc_gcn_pools[i](vc_g)
            vc_sub_g1 = self.vc_pool_gcns[i](vc_sub_g1)

            # cross attention
            doc_attn_output, _ = self.doc_cross_attn[i](doc_sub_g1.ndata['h'].unsqueeze(1), vc_sub_g1.ndata['h'].unsqueeze(1), vc_sub_g1.ndata['h'].unsqueeze(1))
            vc_attn_output, _ = self.vc_cross_attn[i](vc_sub_g1.ndata['h'].unsqueeze(1), doc_sub_g1.ndata['h'].unsqueeze(1), doc_sub_g1.ndata['h'].unsqueeze(1))

            # node attention
            doc_sub_g1.ndata['h'] = self.doc_node_attn[i](doc_attn_output.squeeze(1))
            vc_sub_g1.ndata['h'] = self.vc_node_attn[i](vc_attn_output.squeeze(1))

            doc_pools.append(doc_sub_g1)
            doc_parents.append(doc_g)

            vc_pools.append(vc_sub_g1)
            vc_parents.append(vc_g)

        for i in range(self.n_layers)[::-1]:
            doc_g = self.doc_gcn_unpools[i](doc_parents[i], doc_pools[i])
            vc_g = self.vc_gcn_unpools[i](vc_parents[i], vc_pools[i])

        return doc_g, vc_g



class ResGCNProcess(torch.nn.Module):
    def __init__(self, g_hidden_dim, num_etypes):
        super().__init__()
        self.process_block1 = GCNBlock(g_hidden_dim, g_hidden_dim,
                                       num_etypes)
        self.process_block2 = GCNBlock(g_hidden_dim, g_hidden_dim, num_etypes)

        self.process_block3 = GCNBlock(g_hidden_dim*2, g_hidden_dim*2,
                                       num_etypes)

        self.process_block4 = GCNBlock(g_hidden_dim*2,
                                       g_hidden_dim*2,
                                       num_etypes)
        self.gamma = nn.Parameter(torch.FloatTensor(g_hidden_dim*3))
        self.attention = SparsemaxNodeAttention(g_hidden_dim*3,
                                                g_hidden_dim,
                                                qkv_bias=False)

        self.process_block5 = GCNBlock(g_hidden_dim*3, g_hidden_dim,
                                       num_etypes)

    def forward(self, g):
        enc_data = g.ndata['h']
        self.process_block1(g)
        self.process_block2(g)

        enc_data2 = g.ndata['h']
        g.ndata['h'] = torch.cat((enc_data, g.ndata['h']), -1)
        self.process_block3(g)
        self.process_block4(g)

        g.ndata['h'] = torch.cat((enc_data2, g.ndata['h']), -1)

        self.process_block5(g)
        return g
