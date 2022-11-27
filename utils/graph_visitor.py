import networkx as nx
from graph_algos.nx_shortcuts import neighbors_out
from collections import deque
from collections.abc import Iterable

class DirectedVisitor(object):
    def __init__(self, nx, start_elems, edge_filter_func):
        self.nx = nx
        self.start_elems = start_elems
        self.edge_filter_func = edge_filter_func
        assert isinstance(start_elems, Iterable)

    def __iter__(self):
        assert 
        self.queue = deque(self.start_elems)
        return self

    def __next__(self):
        if not self.queue:
            raise StopIteration
        n = self.queue.popleft()
        self.queue.extend(neighbors_out(n, self.nx, self.edge_filter_func))
        return n
