import networkx as nx
from typing import List, Dict, Tuple
from collections import defaultdict
from graph_algos.nx_shortcuts import neighbors_out
import random

# implement the following function
# 1. extract all possible extension rules from parent to child
# 2. extract all possible parent-context based breadth-first rules
# 3. extract all possible parent-context based random rules


def extract_parent_child_extension_rules(
        nx_gs: List[nx.MultiDiGraph]) -> Dict[Tuple[int, int, int], int]:
    """
    Extract all possible extension rules from parent to child
    :param nx_gs: list of networkx graphs
    :return: dictionary of extension rules
    """
    out_dict = defaultdict(int)
    for nx_g in nx_gs:
        for u, v, data in nx_g.edges(data=True):
            out_dict[(nx_g.nodes[u]['label'], nx_g.nodes[v]['label'],
                      data['label'])] += 1
    return out_dict


def extract_breadth_first_rules(
    nx_gs: List[nx.MultiDiGraph]
) -> Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], int]:
    """
    Extract all possible parent-context based breadth-first rules
    :param nx_gs: list of networkx graphs
    :return: dictionary of breadth-first rules
    """
    out_dict = defaultdict(int)
    for nx_g in nx_gs:
        # assumption: previous edges added is the same as index in the list
        for u, v, data in nx_g.edges(data=True):
            prev_siblings = list(n for n in neighbors_out(u, nx_g) if n < v)
            if not prev_siblings:
                out_dict[((nx_g.nodes[u]['label'], -1, -1),
                          (nx_g.nodes[u]['label'], nx_g.nodes[v]['label'],
                           data['label']))] += 1
            else:
                prev_sibling = max(prev_siblings)
                out_dict[((nx_g.nodes[u]['label'],
                           nx_g.nodes[prev_sibling]['label'],
                           nx_g.edges[u, prev_sibling, 0]['label']),
                          (nx_g.nodes[u]['label'], nx_g.nodes[v]['label'],
                           data['label']))] += 1
    return out_dict


def extract_random_rules(
    nx_gs: List[nx.MultiDiGraph]
) -> Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int,
                                                                  int]], int]:
    """
    Extract all possible parent-context based random rules
    :param nx_gs: list of networkx graphs
    :return: dictionary of random rules
    """
    out_dict = defaultdict(int)
    for nx_g in nx_gs:
        for u, v, data in nx_g.edges(data=True):
            prev_siblings = list(n for n in neighbors_out(u, nx_g) if n < v)
            post_siblings = list(n for n in neighbors_out(u, nx_g) if n > v)
            if not prev_siblings and not post_siblings:
                out_dict[((nx_g.nodes[u]['label'], -1, -1),
                          (nx_g.nodes[u]['label'], nx_g.nodes[v]['label'],
                           data['label']), (nx_g.nodes[u]['label'], -1,
                                            -1))] += 1
            elif not prev_siblings:
                post_sibling = min(post_siblings)
                out_dict[((nx_g.nodes[u]['label'], -1, -1),
                          (nx_g.nodes[u]['label'], nx_g.nodes[v]['label'],
                           data['label']), (nx_g.nodes[u]['label'],
                                            nx_g.nodes[post_sibling]['label'],
                                            nx_g.edges[u, post_sibling,
                                                       0]['label']))] += 1
            elif not post_siblings:
                prev_sibling = max(prev_siblings)
                out_dict[((nx_g.nodes[u]['label'],
                           nx_g.nodes[prev_sibling]['label'],
                           nx_g.edges[u, prev_sibling, 0]['label']),
                          (nx_g.nodes[u]['label'], nx_g.nodes[v]['label'],
                           data['label']), (nx_g.nodes[u]['label'], -1,
                                            -1))] += 1
            else:
                prev_sibling = max(prev_siblings)
                post_sibling = min(post_siblings)
                out_dict[((nx_g.nodes[u]['label'],
                           nx_g.nodes[prev_sibling]['label'],
                           nx_g.edges[u, prev_sibling, 0]['label']),
                          (nx_g.nodes[u]['label'], nx_g.nodes[v]['label'],
                           data['label']), (nx_g.nodes[u]['label'],
                                            nx_g.nodes[post_sibling]['label'],
                                            nx_g.edges[u, post_sibling,
                                                       0]['label']))] += 1
    return out_dict


class ProbabilisticTreeGrammar:
    """
    Probabilistic Tree grammar class
    """

    def __init__(self, nx_gs: List[nx.MultiDiGraph]):
        """
        :param nx_gs: list of networkx graphs
        """
        self.extension_rules = extract_parent_child_extension_rules(nx_gs)
        self.breadth_first_rules = extract_breadth_first_rules(nx_gs)
        self.random_rules = extract_random_rules(nx_gs)

    def filter_independent_rules(self, **kw_args) -> Dict:
        label_u = kw_args.get('label_u', None)
        label_v = kw_args.get('label_v', None)
        label_e = kw_args.get('label_e', None)
        filtered_extension_rules = {}
        if label_u is not None:
            filtered_extension_rules = {
                k: v
                for k, v in self.extension_rules.items() if k[0][0] == label_u
            }
        if label_v is not None:
            filtered_extension_rules = {
                k: v
                for k, v in filtered_extension_rules.items()
                if k[1][0] == label_v
            }
        if label_e is not None:
            filtered_extension_rules = {
                k: v
                for k, v in filtered_extension_rules.items()
                if k[1][2] == label_e
            }
        return filtered_extension_rules

    def sampling_independent_rules(self,
                                   *,
                                   label_u=None,
                                   label_v=None,
                                   label_e=None,
                                   k=None) -> Tuple[int, int, int] or None:
        filtered_extension_rules = self.filter_independent_rules(
            label_u=label_u, label_v=label_v, label_e=label_e)
        if filtered_extension_rules:
            return random.choices(list(filtered_extension_rules.keys()),
                                  weights=list(
                                      filtered_extension_rules.values()),
                                  k=k)[0]
        else:
            return None

    def sampling(nx_g, u):
        # TODO: build an uniformed sampling based
        # u is the parent node of the new node
        # and return a new edge

        # Build a bayesian condition
        # How to sample a new edge?
        # 1. Starts from sampling random_rules
        # 2. If no random_rules, sample breadth_first_rules
        # 3. If no breadth_first_rules, sample extension_rules
        # if none: return None
        pass


### TEST ####
def test_extract_parent_child_extension_rules():
    nx_g = nx.MultiDiGraph()
    nx_g.add_nodes_from([1, 2, 3, 4, 5, 6])
    nx_g.add_edges_from([(1, 2, {
        'label': 1
    }), (1, 3, {
        'label': 2
    }), (2, 4, {
        'label': 3
    }), (2, 5, {
        'label': 4
    }), (3, 6, {
        'label': 5
    })])
    nx_g.nodes[1]['label'] = 1
    nx_g.nodes[2]['label'] = 2
    nx_g.nodes[3]['label'] = 3
    nx_g.nodes[4]['label'] = 4
    nx_g.nodes[5]['label'] = 5
    nx_g.nodes[6]['label'] = 6
    assert extract_parent_child_extension_rules([nx_g]) == {
        (1, 2, 1): 1,
        (1, 3, 2): 1,
        (2, 4, 3): 1,
        (2, 5, 4): 1,
        (3, 6, 5): 1
    }


def test_extract_breadth_first_rules():
    nx_g = nx.MultiDiGraph()
    nx_g.add_nodes_from([1, 2, 3, 4, 5, 6])
    nx_g.add_edges_from([(1, 2, {
        'label': 1
    }), (1, 3, {
        'label': 2
    }), (2, 4, {
        'label': 3
    }), (2, 5, {
        'label': 4
    }), (3, 6, {
        'label': 5
    })])
    nx_g.nodes[1]['label'] = 1
    nx_g.nodes[2]['label'] = 2
    nx_g.nodes[3]['label'] = 3
    nx_g.nodes[4]['label'] = 4
    nx_g.nodes[5]['label'] = 5
    nx_g.nodes[6]['label'] = 6
    assert extract_breadth_first_rules([nx_g]) == {
        ((1, -1, -1), (1, 2, 1)): 1,
        ((1, 2, 1), (1, 3, 2)): 1,
        ((2, -1, -1), (2, 4, 3)): 1,
        ((2, 4, 3), (2, 5, 4)): 1,
        ((3, -1, -1), (3, 6, 5)): 1,
    }, extract_breadth_first_rules([nx_g])


def test_extract_random_rules():
    nx_g = nx.MultiDiGraph()
    nx_g.add_nodes_from([1, 2, 3, 4, 5, 6])
    nx_g.add_edges_from([(1, 2, {
        'label': 1
    }), (1, 3, {
        'label': 2
    }), (2, 4, {
        'label': 3
    }), (2, 5, {
        'label': 4
    }), (3, 6, {
        'label': 5
    })])
    nx_g.nodes[1]['label'] = 1
    nx_g.nodes[2]['label'] = 2
    nx_g.nodes[3]['label'] = 3
    nx_g.nodes[4]['label'] = 4
    nx_g.nodes[5]['label'] = 5
    nx_g.nodes[6]['label'] = 6
    assert extract_random_rules([nx_g]) == {
        ((1, -1, -1), (1, 2, 1), (1, 3, 2)): 1,
        ((1, 2, 1), (1, 3, 2), (1, -1, -1)): 1,
        ((2, -1, -1), (2, 4, 3), (2, 5, 4)): 1,
        ((2, 4, 3), (2, 5, 4), (2, -1, -1)): 1,
        ((3, -1, -1), (3, 6, 5), (3, -1, -1)): 1,
    }, extract_random_rules([nx_g])


if __name__ == '__main__':
    test_extract_parent_child_extension_rules()
    test_extract_breadth_first_rules()
    test_extract_random_rules()
    print('All tests passed!')
