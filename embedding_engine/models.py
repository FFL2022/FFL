from __future__ import print_function, unicode_literals
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from common.model_process import HGP_SL_GraphUNet, ResGCNProcess
from layers.layers_mult import DocEncBlockMult

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GlobalAddPool(nn.Module):
    ''' Global Add Pool
        few notes: Why add? In embedding settings, larger graph
        tend to have more node feature, making this a good pooling
    '''

    def __init__(self):
        ''' Init global add pool module '''
        super().__init__()

    def forward(self, g):
        ''' Perform global add pooling on g'''
        all_feats = g.ndata['h']
        return torch.sum(all_feats, dim=0, keepdim=False)


class OrderEmbedder(nn.Module):
    '''Embed the graph into a space where subgraph isomorphism is
    interpreted as order'''

    def __init__(self, input_dim, hidden_dim, num_etypes, margin=0.5,
                 dropout=0.8):
        ''' Init
        Parameters
        ----------
        input_dim:
            input dimmension of 'x' property in dgl graph
        hidden_dim:
              Short description
        args:
              Short description
        Returns
        ----------
        param_name: type
              Short description
        '''
        super().__init__()
        self.enc_model = DocEncBlockMult(input_dim, hidden_dim)
        self.emb_model = ResGCNProcess(hidden_dim, num_etypes)
        # self.emb_model = HGP_SL_GraphUNet(hidden_dim, num_etypes, 3)
        self.pool_layer = GlobalAddPool()
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))

        ''' mem : 11 gb
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        '''
        self.margin = margin
        self.clf_model = nn.Sequential(nn.Linear(1, 2), nn.LogSoftmax(dim=-1))

    def forward(self, g):
        return self.post_mp(self.pool_layer(self.emb_model(self.enc_model(g))))

    def predict(self, pred):
        """Predict if b is a subgraph of a (batched), where emb_as, emb_bs = pred.

        pred: list (emb_as, emb_bs) of embeddings of graph pairs

        Returns: list of bools (whether a is subgraph of b in the pair)
        """
        emb_as, emb_bs = pred

        e = torch.sum(torch.max(torch.zeros_like(emb_as,
                                                 device=emb_as.device), emb_bs - emb_as)**2, dim=1)
        return e

    def criterion(self, pred, labels):
        ''' Loss function for order emb
        The e term is amount of violation( if b is a subgraph of a)
        For positive examples, the e term is minimized (close to 0)
        For negative examples, the e term is trained to be at least greater than
        self.margin
        pred: List of embeddings output by forward
        labels: subgraph labels for each entry in pred
        '''
        emb_as, emb_bs = pred
        e = torch.sum(torch.max(
            torch.zeros_like(emb_as,
                             device=device), emb_bs - emb_as)**2, dim=1)

        margin = self.margin
        e[labels == 0] = torch.max(
            torch.tensor(0.0,
                         device=device), margin - e)[labels == 0]

        relation_loss = torch.sum(e)

        return relation_loss
