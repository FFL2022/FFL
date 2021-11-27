import networkx as nx
from utils.nx_graph_builder import augment_with_reverse_edge

def map_explain_with_nx(dgl_g, nx_g):
    # check every types of edges
    n_as = [n for n in nx_g.nodes() if nx_g.nodes[n]['graph'] == 'ast']
    n_cs = [n for n in nx_g.nodes() if nx_g.nodes[n]['graph'] == 'cfg']
    n_ts = [n for n in nx_g.nodes() if nx_g.nodes[n]['graph'] == 'test']

    n_alls = {'ast': n_as, 'cfg': n_cs, 'test': n_ts}

    # print(len(n_alls['ast']))
    # print(dgl_g.nodes(ntype='ast').shape)
    # exit()

    # Augment with reverse edge so that the two met
    # ast_etypes = set()
    # for u, v, k, e in list(nx_g.edges(keys=True, data=True)):
    #     if nx_g.nodes[u]['graph'] == 'ast' and\
    #             nx_g.nodes[v]['graph'] == 'ast':
    #         ast_etypes.add(e['label'])
    # cfg_etypes = set()
    # for u, v, k, e in list(nx_g.edges(keys=True, data=True)):
    #     if nx_g.nodes[u]['graph'] == 'cfg' and\
    #             nx_g.nodes[v]['graph'] == 'cfg':
    #         cfg_etypes.add(e['label'])

    # nx_g = augment_with_reverse_edge(nx_g, ast_etypes, cfg_etypes)
    
    all_etypes = set()
    for u, v, k, e in list(nx_g.edges(keys=True, data=True)):
        all_etypes.add((nx_g.nodes[u]['graph'],
                        e['label'],
                        nx_g.nodes[v]['graph']))

    existed_etypes = []
    for etype in dgl_g.etypes:
        if dgl_g.number_of_edges(etype) > 0:
            existed_etypes.append(etype)
    existed_etypes = list(set(existed_etypes))

    # Loop through each type of edges
    for etype in all_etypes:
        if etype not in existed_etypes:
            continue
        # Get edge in from dgl
        # print(etype, type(etype))
        # exit()
        # if dgl_g.number_of_edges(etype) == 0:
        #     continue
        es = dgl_g.edges(etype=etype)
        data = dgl_g.edges[etype].data['weight'] 

        # magic
        data = data * 7

        # print(es)
        # if 'weight' not in es.data:
        #     continue
        us = es[0]
        vs = es[1]
        for i in range(us.shape[0]):
            u = n_alls[etype[0]][us[i].item()]
            v = n_alls[etype[2]][vs[i].item()]
            # print(f'u={u}, v={v}, vs[{i}]={vs[i]}')
            for k in nx_g[u][v]:
                nx_g[u][v][k]['penwidth'] = data[i].item()
    return  nx_g
