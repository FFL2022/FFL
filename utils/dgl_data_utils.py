from graph_algos.nx_shortcuts import where_node, edges_where
import dgl

def numerize_graph(nx_g, graphs=['ast', 'cfg', 'test']):
    n2id = {}
    for graph in graphs:
        ns = nodes_where(nx_g, graph=graph)
        n2id[graph] = dict([n, i] for i, n in enumerate(n_asts))

    # Create dgl test node
    # No need, will be added automatically when we update edges
    all_canon_etypes = defaultdict(list)
    nx_g = augment_with_reverse_edge_cat(nx_g, self.nx_dataset.ast_etypes, [])

    for u, v, k, e in edges_where(nx_g, where_node(graph=graphs),
                                  where_node(graph=graphs)):
        map_u = n2id[nx_g.nodes[u]['graph']]
        map_v = n2id[nx_g.nodes[v]['graph']]
        all_canon_etypes[
            (nx_g.nodes[u]['graph'], e['label'], nx_g.nodes[v]['graph'])
        ].append([map_u[u], map_v[v]])

    for k in all_canon_etypes:
        type_es = torch.tensor(all_canon_etypes[k], dtype=torch.int32)
        all_canon_etypes[k] = (type_es[:, 0], type_es[:, 1])

    g = dgl.heterograph(all_canon_etypes)
    return g, n2id
