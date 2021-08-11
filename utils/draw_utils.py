import networkx as nx
import re

# AST: actual - predicted
# 0: None, white
# 1: Deleted - light red
# 2. Inserted - light green
# 3. 1 - 1  Super Red #FF5252
# 4. 2 - 2  Super Green 67FF4D
# 5. 1 - 2  Orange #FFBD4D
# 6. 2 - 1  Blue #4DC3FF
# 7. 0 - 1  brown #870009
# 8. 0 - 2 purple #9200FF

# CFG: actual - predicted
# 1: Yellow "#FFFFCC"
# 2: 1 - 1 Super Yellow '#FFF386'
# 3: 0 - 1 Red


def set_label_ast(ast, node):

    ast.nodes[node]['label'] = ast.nodes[node]['ntype'] + \
        ' ' + re.sub(r'[^\x00-\x7F]+', '', ast.nodes[node]['token'])
    if 'status' in ast.nodes[node]:
        ast.nodes[node]['style'] = "filled"
        fillcolormap = {
            0: '#FFFFFF',
            2: "#CCFFCC", 1: "#FFCCCC", 3: '#FF5252', 4: '#67FF4D',
            5: '#FFBD4D', 6: '#4DC3FF', 7: '#870009', 8: '#9200FF'
        }
        ast.nodes[node]['fillcolor'] = fillcolormap[ast.nodes[node]['status']]


def set_label_cfg(cfg, node):
    cfg.nodes[node]['label'] = cfg.nodes[node]['ntype'] + \
        cfg.nodes[node]['text']
    if cfg.nodes[node]['ntype'] == 'entry_node':
        cfg.nodes[node]['label'] = cfg.nodes[node]['ntype'] + \
            ' ' + cfg.nodes[node]['funcname'] + ' ' +\
            cfg.nodes[node]['text']
    if 'status' in cfg.nodes[node]:
        cfg.nodes[node]['style'] = "filled"
        fillcolormap = {
            0: '#FFFFFF', 1: '#FFFFCC', 2: "#FFCCCC"}
        cfg.nodes[node]['fillcolor'] = fillcolormap[cfg.nodes[node]['status']]


def cfg_to_agraph(cfg: nx.MultiDiGraph, path: str):
    ''' Cfg to agraph
    AGraph: used by pygraphviz for visualization
    Parameters
    ----------
    cfg:  nx.MultiDiGraph
          cfg graph build from "build_nx_cfg"
    path:  str
    '''
    for node in cfg.nodes():
        set_label_cfg(cfg, node)
    nx.drawing.nx_agraph.to_agraph(cfg).draw(path, prog='dot')


def ast_to_agraph(ast: nx.MultiDiGraph, path: str):
    ''' Nx Ast to agraph
    Parameters
    ----------
    ast:  nx.MultiDiGraph
    path:  str
    '''
    for node in ast.nodes():
        set_label_ast(ast, node)
    nx.drawing.nx_agraph.to_agraph(ast).draw(path, prog='dot')


def cfg_ast_to_agraph(cfg_ast: nx.MultiDiGraph, path: str):
    ''' Cfg ast to agraph
    Parameters
    ----------
    cfg_ast:  nx.MultiDiGraph
    path:  str
    '''
    for node in cfg_ast.nodes():
        if cfg_ast.nodes[node]['graph'] == 'cfg':
            set_label_cfg(cfg_ast, node)
        elif cfg_ast.nodes[node]['graph'] == 'ast':
            set_label_ast(cfg_ast, node)
    nx.drawing.nx_agraph.to_agraph(cfg_ast).draw(path, prog='dot')


def cfg_ast_cov_to_agraph(cfg_ast_cov: nx.MultiDiGraph, path):
    g = cfg_ast_cov
    for node in g.nodes():
        if g.nodes[node]['graph'] == 'cfg':
            set_label_cfg(cfg_ast_cov, node)
        elif g.nodes[node]['graph'] == 'ast':
            set_label_ast(cfg_ast_cov, node)
        else:
            g.nodes[node]['label'] = g.nodes[node]['name']

    nx.drawing.nx_agraph.to_agraph(g).draw(path, prog='dot')
