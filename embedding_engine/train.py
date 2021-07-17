from __future__ import print_function, unicode_literals
import torch
import torch.nn as nn
import torch.nn.functional as F
from embedding_engine.models import OrderEmbedder
from graph_algos.cfl_match_general import build_cpi
from graph_algos.spanning_tree_conversion import sample_bfs_from_graph
import tqdm
import random
from embedding_engine.test_cfl_match_general import check_text_contain_char
from execution_engine.legacy_data import default_corpus
from embedding_engine.datasource import OTFDocumentGraphDataSource

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_data_source(args):
    pass


def train(model, datasource, batch_size, epochs=100):
    ''' Train order embedder model
    Parameters
    ----------
    model:
        The model to be trained
    datasource:
        See dataset class in the same folder
    batch_size: int
    epochs: default=100
        number of training epochs
    Returns
    ----------
    param_name: type
          Short description
    '''
    loss = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(opt,
                                                step_size=1000,
                                                gamma=0.1)
    model.train()
    clf_opt = torch.optim.Adam(model.clf_model.parameters(), lr=0.01)
    model.to(device)
    for i in range(epochs):
        print(len(datasource))
        bar = tqdm.tqdm(range(len(datasource)))
        bar.set_description(f'epoch={i}')
        for j in bar:
            # TODO: Sampling random graph then check their
            dgl_doc_graph, doc_graph, _ = datasource.document_graph_dataset[j]
            max_depth = random.randint(1, 6)
            _, _, g_qs, Gs = datasource.gen_batch(j, batch_size, max_depth,
                                                  anchored=True)
            opt.zero_grad()
            emb_qs = []
            emb_Gs = []
            for g_q in g_qs:
                emb_qs.append(model(g_q.to(device)))
            for G in Gs:
                emb_Gs.append(model(G.to(device)))

            emb_qs = torch.stack(emb_qs, dim=0)
            emb_Gs = torch.stack(emb_Gs, dim=0)
            emb_pos = torch.cat((emb_qs, emb_Gs), dim=0)
            emb_neg = torch.cat((emb_Gs, emb_qs), dim=0)
            labels = torch.tensor([1]*emb_qs.shape[0] +
                                  [0]*emb_Gs.shape[0]).to(device)
            loss = model.criterion((emb_pos, emb_neg), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            with torch.no_grad():
                pred = model.predict((emb_pos, emb_neg))
            model.clf_model.zero_grad()
            pred = model.clf_model(pred.unsqueeze(1))
            criterion = nn.NLLLoss()
            clf_loss = criterion(pred, labels)
            clf_loss.backward()
            clf_opt.step()
            pred = pred.argmax(dim=-1)
            acc = torch.mean((pred == labels).type(torch.float))
            bar.set_postfix(loss=loss.item(), clf_loss=clf_loss.item(),
                            acc=acc.item())


def main():

    model = OrderEmbedder(len(default_corpus) + 1, 256, 4, margin=0.5,
                          dropout=0.8)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    ''' # To be use 'no_edge'
    model = OrderEmbedder(len(default_corpus) + 1, 256, 5, margin=0.5,
                          dropout=0.8)
    '''
    datasource = OTFDocumentGraphDataSource(transform_to_full_graph=False)
    train(model, datasource, 3, 100)


if __name__ == '__main__':
    main()
