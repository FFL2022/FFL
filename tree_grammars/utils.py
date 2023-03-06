import networkx as nx
from typing import List, Dict, Tuple
from collections import defaultdict
from graph_algos.nx_shortcuts import neighbors_out


# implement the following function
# 1. extract all possible extension rules from parent to child
# 2. extract all possible parent-context based breadth-first rules
# 3. extract all possible parent-context based random rules


def extract_parent_child_extension_rules(nx_gs: List[nx.MultiDiGraph]) -> Dict[Tuple[int, int, int], int]:
    """
    Extract all possible extension rules from parent to child
    :param nx_gs: list of networkx graphs
    :return: dictionary of extension rules
    """
    out_dict = defaultdict(int)
    for nx_g in nx_gs:
        for u, v, data in nx_g.edges(data=True):
            out_dict[(nx_g.nodes[u]['label'], nx_g.nodes[v]['label'], data['label'])] += 1
    return out_dict


def extract_breadth_first_rules(nx_gs: List[nx.MultiDiGraph]) -> Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], int]:
    """
    Extract all possible parent-context based breadth-first rules
    :param nx_gs: list of networkx graphs
    :return: dictionary of breadth-first rules
    """
    out_dict = defaultdict(int)
    for nx_g in nx_gs:
        # assumption: previous edges added is the same as index in the list
        for u, v, data in nx_g.edges(data=True):
            prev_siblings = list(n for n in neighbors_out(nx_g.nodes[u], nx_g) if n < v)
            if not prev_siblings:
                out_dict[((nx_g.nodes[u]['label'], -1, -1), (nx_g.nodes[u]['label'], nx_g.nodes[v]['label'], data['label']))] += 1
            else:
                prev_sibling = max(prev_siblings)
                out_dict[((nx_g.nodes[u]['label'], nx_g.nodes[prev_sibling]['label'], nx_g[prev_sibling][u]['label']), (nx_g.nodes[u]['label'], nx_g.nodes[v]['label'], data['label']))] += 1
    return out_dict


def extract_random_rules(nx_gs: List[nx.MultiDiGraph]) -> Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]], int]:
    """
    Extract all possible parent-context based random rules
    :param nx_gs: list of networkx graphs
    :return: dictionary of random rules
    """
    out_dict = defaultdict(int)
    for nx_g in nx_gs:
        for u, v, data in nx_g.edges(data=True):
            prev_siblings = list(n for n in neighbors_out(nx_g.nodes[u], nx_g) if n < v)
            post_siblings = list(n for n in neighbors_out(nx_g.nodes[u], nx_g) if n > v)
            if not prev_siblings and not post_siblings:
                out_dict[((nx_g.nodes[u]['label'], -1, -1), (nx_g.nodes[u]['label'], nx_g.nodes[v]['label'], data['label']), (nx_g.nodes[u]['label'], -1, -1))] += 1
            elif not prev_siblings:
                post_sibling = min(post_siblings)
                out_dict[((nx_g.nodes[u]['label'], -1, -1), (nx_g.nodes[u]['label'], nx_g.nodes[v]['label'], data['label']), (nx_g.nodes[u]['label'], nx_g.nodes[post_sibling]['label'], nx_g[post_sibling][u]['label']))] += 1
            elif not post_siblings:
                prev_sibling = max(prev_siblings)
                out_dict[((nx_g.nodes[u]['label'], nx_g.nodes[prev_sibling]['label'], nx_g[prev_sibling][u]['label']), (nx_g.nodes[u]['label'], nx_g.nodes[v]['label'], data['label']), (nx_g.nodes[u]['label'], -1, -1))] += 1
            else:
                prev_sibling = max(prev_siblings)
                post_sibling = min(post_siblings)
                out_dict[((nx_g.nodes[u]['label'], nx_g.nodes[prev_sibling]['label'], nx_g[prev_sibling][u]['label']), (nx_g.nodes[u]['label'], nx_g.nodes[v]['label'], data['label']), (nx_g.nodes[u]['label'], nx_g.nodes[post_sibling]['label'], nx_g[post_sibling][u]['label']))] += 1
    return out_dict



### TEST ####
def test_extract_parent_child_extension_rules():
    nx_g = nx.MultiDiGraph()
    nx_g.add_nodes_from([1, 2, 3, 4, 5, 6])
    nx_g.add_edges_from([(1, 2, {'label': 1}), (1, 3, {'label': 2}), (2, 4, {'label': 3}), (2, 5, {'label': 4}), (3, 6, {'label': 5})])
    nx_g.nodes[1]['label'] = 1
    nx_g.nodes[2]['label'] = 2
    nx_g.nodes[3]['label'] = 3
    nx_g.nodes[4]['label'] = 4
    nx_g.nodes[5]['label'] = 5
    nx_g.nodes[6]['label'] = 6
    assert extract_parent_child_extension_rules([nx_g]) == {(1, 2, 1): 1, (1, 3, 2): 1, (2, 4, 3): 1, (2, 5, 4): 1, (3, 6, 5): 1}


def test_extract_breadth_first_rules():
    nx_g = nx.MultiDiGraph()
    nx_g.add_nodes_from([1, 2, 3, 4, 5, 6])
    nx_g.add_edges_from([(1, 2, {'label': 1}), (1, 3, {'label': 2}), (2, 4, {'label': 3}), (2, 5, {'label': 4}), (3, 6, {'label': 5})])
    nx_g.nodes[1]['label'] = 1
    nx_g.nodes[2]['label'] = 2
    nx_g.nodes[3]['label'] = 3
    nx_g.nodes[4]['label'] = 4
    nx_g.nodes[5]['label'] = 5
    nx_g.nodes[6]['label'] = 6
    assert extract_breadth_first_rules([nx_g]) == {
        ((1, -1, -1), (1, 2, 1)): 1,
        ((1, 2, 1), (1, 3, 2)): 1,
        ((1, 3, 2), (2, 4, 3)): 1,
        ((1, 3, 2), (2, 5, 4)): 1,
        ((2, 4, 3), (3, 6, 5)): 1,
        ((2, 5, 4), (3, 6, 5)): 1
    }


def test_extract_random_rules():
    nx_g = nx.MultiDiGraph()
    nx_g.add_nodes_from([1, 2, 3, 4, 5, 6])
    nx_g.add_edges_from([(1, 2, {'label': 1}), (1, 3, {'label': 2}), (2, 4, {'label': 3}), (2, 5, {'label': 4}), (3, 6, {'label': 5})])
    nx_g.nodes[1]['label'] = 1
    nx_g.nodes[2]['label'] = 2
    nx_g.nodes[3]['label'] = 3
    nx_g.nodes[4]['label'] = 4
    nx_g.nodes[5]['label'] = 5
    nx_g.nodes[6]['label'] = 6
    assert extract_random_rules([nx_g]) == {
        ((1, -1, -1), (1, 2, 1), (1, -1, -1)): 1,
        ((1, -1, -1), (1, 3, 2), (1, 2, 1)): 1,
        ((1, 2, 1), (1, 3, 2), (1, 2, 1)): 1,
        ((1, 2, 1), (2, 4, 3), (1, 3, 2)): 1,
        ((1, 2, 1), (2, 5, 4), (1, 3, 2)): 1,
        ((1, 3, 2), (2, 4, 3), (1, 2, 1)): 1,
        ((1, 3, 2), (2, 5, 4), (1, 2, 1)): 1,
        ((1, 3, 2), (3, 6, 5), (1, -1, -1)): 1,
        ((2, 4, 3), (3, 6, 5), (1, 3, 2)): 1,
        ((2, 5, 4), (3, 6, 5), (1, 3, 2)): 1
    }

if __name__ == '__main__':
    test_extract_parent_child_extension_rules()
    test_extract_breadth_first_rules()
    test_extract_random_rules()
    print('All tests passed!')
