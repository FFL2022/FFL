import torch
import torch.nn as nn
from common import device


__author__ = "Marc: thanhdatn@student.unimelb.edu.au"
edge_types = ['tb', 'lr', 'bt', 'child', 'parent', 'master']
e2idmap = [(e, i) for i, e in enumerate(edge_types)]


class MPNNModel(nn.Module):
    def __init__(self, cl_dim, cc_dim,
                 hdim, edim, netypes, ncls,
                 al_dim, ac_dim, n_layers=5, device=device):
        super().__init__()
        self.cl_enc = nn.Linear(cl_dim, hdim)
        self.cc_enc = nn.Linear(cc_dim, hdim)

        self.al_enc = nn.Linear(al_dim, hdim)
        self.ac_enc = nn.Linear(ac_dim, hdim)

        nn.init.xavier_normal_(self.al_enc.weight)
        nn.init.normal_(self.al_enc.bias)
        nn.init.xavier_normal_(self.al_enc.weight)
        nn.init.normal_(self.ac_enc.bias)

        self.pt_emb = nn.Parameter(torch.FloatTensor(hdim))
        nn.init.normal_(self.pt_emb)
        self.ft_emb = nn.Parameter(torch.FloatTensor(hdim))
        nn.init.normal_(self.ft_emb)

        self.netypes = netypes
        self.n_layers = n_layers
        self.relu = nn.ReLU()
        self.mpnn_cc = nn.ModuleList(
            [nn.Sequential(nn.Linear(hdim, hdim), nn.ReLU())
             for _ in range(n_layers)])
        self.mpnn_tc = nn.ModuleList(
            [nn.Sequential(nn.Linear(hdim, hdim), nn.ReLU())
             for _ in range(n_layers)])
        self.mpnn_ac = nn.ModuleList(
            [nn.Sequential(nn.Linear(hdim, hdim), nn.ReLU())
             for _ in range(n_layers)])
        self.mpnn_aa = nn.ModuleList(
            [nn.Sequential(nn.Linear(hdim, hdim), nn.ReLU())
             for _ in range(n_layers)])
        self.mpnn_ca = nn.ModuleList(
            [nn.Sequential(nn.Linear(hdim, hdim), nn.ReLU())
             for _ in range(n_layers)])
        self.mpnn_ta = nn.ModuleList(
            [nn.Sequential(nn.Linear(hdim, hdim), nn.ReLU())
             for _ in range(n_layers)])
        self.mpnn_ct = nn.ModuleList(
            [nn.Sequential(nn.Linear(hdim, hdim), nn.ReLU())
             for _ in range(n_layers)])
        self.mpnn_at = nn.ModuleList(
            [nn.Sequential(nn.Linear(hdim, hdim), nn.ReLU())
             for _ in range(n_layers)])

    def forward(self, x_cl, x_cc, x_al, x_ac, x_pt, x_ft,
                e_cc_f, e_ca_f, e_ct_f, e_ac_f, e_aa_f, e_at_f, e_tc_f, e_ta_f,
                e_cc, e_ca, e_ct, e_ac, e_aa, e_at, e_tc, e_ta):
        x_c = self.relu(self.cc_enc(x_cc) + self.cl_enc(x_cl))
        x_a = self.relu(self.ac_enc(x_ac) + self.al_enc(x_al))
        for _ in range(self.n_layers):
            # 1. message passing to update x_cl
            # 2. message passing to update x_cc
            # 3. message passing to update x_al
            # 4. message passing to update x_ac
            # 5. message passing to update x_pt
            # 6. message passing to update x_ft
            pass
        pass

