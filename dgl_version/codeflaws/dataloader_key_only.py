from dgl_version.codeflaws.dataloader_key_only_old import *

class CodeflawsASTDGLDataset(CodeDGLDataset):
    def __init__(self, raw_dir=ConfigClass.codeflaws_data_path,
                 save_dir=ConfigClass.preprocess_dir_codeflaws):
        self.name = f'{mode}_cf_ast_only_dgl'
        super().__init__(dataloader, meta_data, self.name, save_dir,
                convert_arg_func=lambda x: embedding_model, *x)


    def convert_from_nx_to_dgl(self, embedding_model, nx_g, ast_lb_d,
                               ast_lb_i, cfg_lb):
        nx_g = augment_with_reverse_edge_cat(
            nx_g, self.nx_dataset.ast_etypes,
            self.nx_dataset.cfg_etypes)
        g, n2id = numerize_graph(nx_g, ['ast'])
        ast2id = n2id['ast']

        g.nodes['ast'].data['label'] = torch.tensor([
            self.nx_dataset.ast_types.index(nx_g.nodes[node]['ntype'])
            for node in n2id], dtype=torch.long
        )

        g.nodes['ast'].data['content'] = torch.stack([
            torch.from_numpy(embedding_model.get_sentence_vector(
                nx_g.nodes[n]['token'].replace('\n', '')))
            for n in n2id], dim=0)
        
        g = dgl.heterograph(all_canon_etypes)
        g = dgl.add_self_loop(g, etype=('ast', 'a_self_loop', 'ast'))
        ast_tgts = torch.zeros(len(n_asts), dtype=torch.long)
        ast_tgts[list(map(lambda n: ast2id[n], ast_lb_d))] = 1
        ast_tgts[list(map(lambda n: ast2id[n], ast_lb_i))] = 2
        g.nodes['ast'].data['tgt'] = ast_tgts

        return g

    def __len__(self):
        return len(self.active_idxs)

    def __getitem__(self, i):
        return self.gs[self.active_idxs[i]]
