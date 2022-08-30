import torch
import torch.nn as nn
from common import device


__author__ = "Marc: thanhdatn@student.unimelb.edu.au"
edge_types = ['tb', 'lr', 'bt', 'child', 'parent', 'master']
e2idmap = [(e, i) for i, e in enumerate(edge_types)]


class MPNNModel(nn.Module):
    def __init__(self, ncfg_lfts, ncfg_cfts,
                 hfts, efts, netypes, ncls,
                 nast_lfts, nast_cfts, n_layers=5, device=device):
        super().__init__()
        self.cfgl_enc = nn.Linear(ncfg_lfts, hfts)
        self.cfgc_enc = nn.Linear(ncfg_cfts, hfts)

        self.ast_lenc = nn.Linear(nast_lfts, hfts)
        self.ast_cenc = nn.Linear(nast_cfts, hfts)

        nn.init.xavier_normal_(self.ast_lenc.weight)
        nn.init.normal_(self.ast_lenc.bias)
        nn.init.xavier_normal_(self.ast_cenc.weight)
        nn.init.normal_(self.ast_cenc.bias)

        self.pt_emb = nn.Parameter(torch.FloatTensor(hfts))
        nn.init.normal_(self.pt_emb)
        self.ft_emb = nn.Parameter(torch.FloatTensor(hfts))
        nn.init.normal_(self.ft_emb)

        self.netypes = netypes
        self.n_layers = n_layers
        self.mpnn_cc = [

    def forward(self, x_cl, x_cc, x_al, x_ac, x_pt, x_ft,
                e_cc, e_ca, e_ct, e_ac, e_aa, e_at, e_tc, e_ta):
        # 1. message passing to update x_cl
        # 2. message passing to update x_cc
        # 3. message passing to update x_al
        # 4. message passing to update x_ac
        # 5. message passing to update x_pt
        # 6. message passing to update x_ft
        pass
