import itertools
import random
import networkx as nx
from graph_algos.nx_shortcuts import all_neighbors


class TreeNode(object):
    def __init__(self, node_label):
        self.node_label = node_label
        self.children = []
        self.parent_edge_type = 0
        self.parent = None

    def add_child(self, child, edge_type=0):
        self.children.append(child)
        child.parent = self
        child.parent_edge_type = edge_type

    def get_leaf(self):
        if len(self.children) != 0:
            return list(set(list(
                itertools.chain.from_iterable(
                    child.get_leaf()
                    for child in self.children))))

    def get_path_to_root(self, prepend=None):
        if prepend is None:
            prepend = [self.node_label]
        else:
            prepend = [self.node_label] + prepend
        return self.parent.get_path_to_root(prepend)

    def get_all_paths(self):
        # get all path from children
        if len(self.children) == 0:
            return [[self.node_label]]
        paths = []
        for child in self.children:
            child_paths = child.get_all_paths()
            for child_path in child_paths:
                paths.append(
                    [[self.node_label, child.parent_edge_type]] + child_path)
            paths.append([[self.node_label, 'end']])
        return paths


def level_order_traverse(root):
    '''perform level-order traverse'''
    nodes = []
    if root is None:
        return nodes
    queue = []
    queue.append(root)
    while (len(queue) > 0):
        root = queue.pop(0)
        nodes.append(root)
        if len(root.children) > 0:
            for child in root.children:
                queue.append(child)
    return nodes


def get_elbl(epair, q, key=None):
    '''quickly get edge label
    Since nx.MultiDiGraph might have multiple edges for a pair of node,
    key might be required. In case None specified, take the first key
    '''
    if key is None:
        key = list(q.get_edge_data(*epair).keys())[0]
    return q.get_edge_data(*epair)[key]['label']


def graph2spanning_tree(q: nx.MultiDiGraph, n_s, node_dict, max_depth=-1):
    '''
    param :q: input query graph
    param :n_s: starting node
    param :node_dict: dict to store mapping from node label to TreeNode
    param :max_depth: maximum sampling depth
    '''

    unvisited_sets = list(
        [node for node in q.nodes() if not q.nodes[node]['visited']])
    updated_set = [n_s]
    # First, update with multiple spanning trees
    if max_depth == -1:
        max_depth = 9999
    while(len(updated_set) > 0):
        updated_set = []
        for node in unvisited_sets[:]:
            # Check if any of this node label is visited
            # Get all node neighbor
            if q.nodes[node]['visited']:
                continue
            n_is, n_os, n_as = all_neighbors(node, q)
            n_as = list(set(n_as).intersection(set(node_dict.keys())))
            best_n = None
            for n_n in n_as:
                next_dist = q.nodes[n_n]['distance'] + 1
                if next_dist < q.nodes[node]['distance'] and\
                        next_dist <= max_depth:
                    best_n = n_n
                    q.nodes[node]['distance'] = next_dist
            if best_n is not None:
                epair = (best_n, node) if best_n in n_is else (node, best_n)
                node_dict[node] = TreeNode(node)
                node_dict[best_n].add_child(
                    node_dict[node], edge_type=get_elbl(epair, q))
                updated_set.append(node)
                unvisited_sets.remove(node)
    return unvisited_sets


def sample_bfs_from_graph(g, num_sample: int, max_depth: int):
    ''' Sampling BFS from node
    :param :g: graph
    :param :num_sample: number of subgraph to be sampled
    :param :max_depth: maximum depth of the BFS tree
    return mapping from chosen nodes to tree
    Examples output:
        {'n0': TreeNode(n0)}
    '''
    # Choose a node
    max_node = g.number_of_nodes()
    centers = random.sample(range(0, max_node), num_sample)
    node_dicts = []
    for c_idx in centers:
        # Reset visited state before each sampling
        for node in g.nodes():
            g.nodes[node]['visited'] = False
            g.nodes[node]['distance'] = 9999
        node_id = 'n' + str(c_idx)
        g.nodes[node_id]['visited'] = True
        g.nodes[node_id]['distance'] = 0
        node_dict = {node_id: TreeNode(node_id)}
        graph2spanning_tree(g, c_idx, node_dict, max_depth)
        node_dicts.append(node_dict)
        ''' Either that, or just call networkx ego graph'''
    return node_dicts, centers


def graph2spanning_trees(q: nx.MultiDiGraph, n_s):
    '''
    param :q: input query graph
    param :n_s: starting node
    '''
    roots = [TreeNode(n_s)]
    node_dicts = [{n_s: roots[0]}]
    for node in q.nodes():
        if node not in node_dicts[0]:
            q.nodes[node]['visited'] = False
            q.nodes[node]['distance'] = 99999

    q.nodes[n_s]['visited'] = True
    q.nodes[n_s]['distance'] = 0
    unvisited_sets = graph2spanning_tree(q, n_s, node_dicts[0])
    while (len(unvisited_sets) > 0):
        node = unvisited_sets[0]
        roots.append(TreeNode(node))
        node_dicts.append({node: roots[-1]})
        q.nodes[node]['visited'] = True
        q.nodes[node]['distance'] = 0
        unvisited_sets = graph2spanning_tree(q, node, node_dicts[-1])

    # Get set of nodes that have yet to be visited
    for node in q.nodes():
        q.nodes[node]['visited'] = False
    return roots, node_dicts
