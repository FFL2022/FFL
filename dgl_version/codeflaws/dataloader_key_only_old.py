from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
import dgl
from utils.utils import ConfigClass
import os
from codeflaws.dataloader_cfl import CodeflawsCFLNxStatementDataset, \
        CodeflawsAstGraphMetadata
from utils.nx_graph_builder import augment_with_reverse_edge_cat
from graph_algos.nx_shortcuts import nodes_where
import pickle as pkl
import random
import torch
import tqdm
from dgl_version.data_utils import numerize_graph, CodeDGLDataset

embedding_model = fasttext.load_model(ConfigClass.pretrained_fastext)

class CodeflawsASTDGLDataset(CodeDGLDataset):
    def __init__(self, dataloader: NxDataloader,
                 meta_data: CodeflawsAstGraphMetadata,
                 mode, 
                 save_dir=ConfigClass.preprocess_dir_codeflaws):
        self.name = f"{mode}_codeflaws_dgl_ast"
        super().__init__(dataloader, meta_data, self.name, save_dir,
                convert_arg_func=lambda x: embedding_model, *x)
        self.cfg_content_dim = self.gs[0].nodes['cfg'].data['content'].shape[-1]
        self.ast_content_dim = self.gs[0].nodes['ast'].data['content'].shape[-1]


    def convert_from_nx_to_dgl(self, embedding_model, nx_g, ast_lb_d,
                               ast_lb_i, cfg_lb):
        # Create a node mapping for ast and test nodes
        nx_g = augment_with_reverse_edge_cat(nx_g, self.meta_graph.t_e_asts, [])
        g, n2id = numerize_graph(nx_g, ['ast', 'test'])
        ast2id = n2id['ast']
        g = dgl.heterograph(all_canon_etypes)
        g.nodes['ast'].data['label'] = torch.tensor([
            self.meta_graph.t_e_asts.index(nx_g.nodes[node]['ntype'])
            for node in n_asts], dtype=torch.long
        )

        g.nodes['ast'].data['content'] = torch.stack([
            torch.from_numpy(embedding_model.get_sentence_vector(
                nx_g.nodes[n]['token'].replace('\n', '')))
            for n in n_asts], dim=0)

        g = dgl.add_self_loop(g, etype=('ast', 'a_self_loop', 'ast'))
        # g = dgl.add_self_loop(g, etype=('cfg', 'c_self_loop', 'cfg'))
        ast_tgts = torch.zeros(len(n_asts), dtype=torch.long)
        ast_tgts[list(map(lambda x: ast2id[x], ast_lb_d))] = 1
        ast_tgts[list(map(lambda x: ast2id[x], ast_lb_i))] = 2
        g.nodes['ast'].data['tgt'] = ast_tgts

        return g


class CodeflawsFullDGLDataset(CodeDGLDataset):
    def __init__(self, dataloader, meta_data, mode, save_dir=ConfigClass.preprocess_dir_codeflaws):
        self.name = f'{mode}_cf_full_dgl'
        super().__init__(dataloader, meta_data, self.name, save_dir,
                convert_arg_func=lambda x: embedding_model, *x)
        self.cfg_content_dim = self.gs[0].nodes['cfg'].data['content'].shape[-1]
        self.ast_content_dim = self.gs[0].nodes['ast'].data['content'].shape[-1]

    def convert_from_nx_to_dgl(self, embedding_model, nx_g, ast_lb_d,
                               ast_lb_i, cfg_lb):
        nx_g = augment_with_reverse_edge_cat(
            nx_g, self.meta_data.t_e_asts,
            self.meta_data.t_e_cfgs)
        g, n2id = numerize_graph(nx_g, ['cfg', 'ast', 'test'])
        cfg2id, ast2id = n2id['cfg'], n2id['ast']
        
        g.nodes['cfg'].data['label'] = torch.tensor(
            [ConfigClass.cfg_label_corpus.index(nx_g.nodes[n]['ntype'])
             for n in cfg2id], dtype=torch.long)
        g.nodes['cfg'].data['content'] = torch.stack([
            torch.from_numpy(embedding_model.get_sentence_vector(
                nx_g.nodes[n]['text'].replace('\n', '')))
            for n in cfg2id], dim=0)
        g.nodes['ast'].data['label'] = torch.tensor([
            self.meta_data.t_asts.index(nx_g.nodes[node]['ntype'])
            for node in ast2id], dtype=torch.long
        )
        g.nodes['ast'].data['content'] = torch.stack([
            torch.from_numpy(embedding_model.get_sentence_vector(
                nx_g.nodes[n]['token'].replace('\n', '')))
            for n in ast2id], dim=0)
        g = dgl.add_self_loop(g, etype=('ast', 'a_self_loop', 'ast'))
        g = dgl.add_self_loop(g, etype=('cfg', 'c_self_loop', 'cfg'))
        tgts = torch.zeros(len(n_cfgs), dtype=torch.long)
        ast_tgts = torch.zeros(len(n_asts), dtype=torch.long)
        ast_tgts[list(map(lambda x: ast2id[x], ast_lb_d))] = 1
        ast_tgts[list(map(lambda x: ast2id[x], ast_lb_i))] = 2
        g.nodes['ast'].data['tgt'] = ast_tgts
        tgts[list(map(lambda x: cfg2id[x], cfg_lb))] = 1
        g.nodes['cfg'].data['tgt'] = tgts
        return g
