from utils.embedding_model import embedding_model
from dgl_version.data_utils import CodeDGLDataset, numerize_graph
from utils.nx_graph_builder import augment_with_reverse_edge_cat


class CodeflawsGumtreeDGLNodeDataset(CodeDGLDataset):
    def __init__(self, dataloader, meta_data, mode, save_dir=ConfigClass.preprocess_dir_codeflaws):
        self.name = f'{mode}_cf_gumtree_dgl_ast_node'
        super().__init__(dataloader, meta_data, self.name, save_dir,
                convert_arg_func=lambda x: embedding_model, *x
                )
        self.cfg_content_dim = 1
        self.ast_content_dim = self.gs[0].nodes['ast'].data['content'].shape[-1]


    def convert_from_nx_to_dgl(self, embedding_model, nx_g):
        nx_g = augment_with_reverse_edge_cat(nx_g, self.meta_graph.t_e_asts, [])
        g, n2id = numerize_graph(nx_g, ['ast', 'test'])
        ast2id = n2id['ast']
        # Create dgl ast node
        g.nodes['ast'].data['label'] = torch.tensor([
            self.nx_dataset.ast_types.index(nx_g.nodes[node]['ntype'])
            for node in ast2id], dtype=torch.long
        )
        g = dgl.add_self_loop(g, etype=('ast', 'a_self_loop', 'ast'))


        # Create dgl ast node
        g.nodes['ast'].data['label'] = torch.tensor([
            self.nx_dataset.ast_types.index(nx_g.nodes[node]['ntype'])
            for node in n_asts], dtype=torch.long
        )

        g.nodes['ast'].data['content'] = torch.stack([
            torch.from_numpy(embedding_model.get_sentence_vector(
                nx_g.nodes[n]['token'].replace('\n', '')))
            for n in n_asts], dim=0)
        # Create dgl test node
        # No need, will be added automatically when we update edges

        ast_tgts = torch.zeros(len(n_asts), dtype=torch.long)
        ast_tgts[list(map(lambda x: ast2id[x], ast2id))] = list(
                nx_g.nodes[x]['status'] for x in ast2id)

        g.nodes['ast'].data['tgt'] = ast_tgts
        return g
