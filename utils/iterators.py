import random
import numpy as np
import pickle as pkl
import glob
import torch

class NodeIterator(object):
    def __init__(self, N):
        self.N = N

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.N:
            raise StopIteration
        x = self.i
        self.i += 1
        return x


class UniformRandomIterator(object):
    def __init__(self, N, a:float, b:float):
        self.N = N
        self.rr = a, b

    def __iter__(self):
        self.i = 0
        self.total = 1
        return self

    def __next__(self):
        if self.i >= self.N:
            raise StopIteration
        prob = random.uniform(*self.rr)
        while self.i < self.N and prob < 1/self.total:
            self.total += 1
            self.i += 1
            prob = random.uniform(*self.rr)
        if self.i >= self.N:
            raise StopIteration
        else:
            x = self.i
            self.i += 1
            return x


class BFSChosenNode(object):
    def __init__(self, g, N=1):
        lb = g.lbl.flatten().detach().cpu().numpy()
        inp = g.visiteds.flatten().detach().cpu().numpy()
        unchange_indices = np.where(lb == inp)[0]
        change_indices = np.where(lb != inp)[0]
        if len(unchange_indices) == 0:
            self.indices = np.random.choice(
                unchange_indices, min(N, unchange_indices.shape[0])).tolist()
        elif len(change_indices) == 0:
            self.indices = np.random.choice(
                change_indices,
                min(N, change_indices.shape[0])).tolist()
        else:
            self.indices = self.indices = np.random.choice(
                unchange_indices, min(N, unchange_indices.shape[0])).tolist() +\
                np.random.choice(
                    change_indices,
                    min(N, change_indices.shape[0])).tolist()
        self.N = len(self.indices)
        # self.rr = 0.3, 1.0

    def __iter__(self):
        self.i = 0
        self.total = 1
        return self

    def __next__(self):
        if self.i >= self.N:
            raise StopIteration
        prob = 1
        # prob = random.uniform(*self.rr)
        while self.i < self.N and prob < 1/self.total:
            self.total += 1
            self.i += 1
            prob = random.random()
        if self.i >= self.N:
            raise StopIteration
        else:
            x = self.i
            self.i += 1
            return (self.indices[x],)

class BlmFDChosenNode(object):
    def __init__(self, g, N=1):
        lb = g.lbl_d.flatten().detach().cpu().numpy()
        inp = g.dists.flatten().detach().cpu().numpy()
        unchange_indices = np.where(lb == inp)[0]
        change_indices = np.where(lb != inp)[0]
        if len(unchange_indices) == 0:
            self.indices = np.random.choice(
                unchange_indices, min(N, unchange_indices.shape[0])).tolist()
        elif len(change_indices) == 0:
            self.indices = np.random.choice(
                change_indices,
                min(N, change_indices.shape[0])).tolist()
        else:
            self.indices = self.indices = np.random.choice(
                unchange_indices, min(N, unchange_indices.shape[0])).tolist() +\
                np.random.choice(
                    change_indices,
                    min(N, change_indices.shape[0])).tolist()
        self.N = len(self.indices)

    def __iter__(self):
        self.i = 0
        self.total = 1
        return self

    def __next__(self):
        if self.i >= self.N:
            raise StopIteration
        prob = random.uniform(*self.rr)
        while self.i < self.N and prob < 1/self.total:
            self.total += 1
            self.i += 1
            prob = random.random()
        if self.i >= self.N:
            raise StopIteration
        else:
            x = self.i
            self.i += 1
            return (self.indices[x],)


class BlmFDChosenEdgeNode(object):
    def __init__(self, g, N=1):
        lb = g.lbl_p.flatten().detach().cpu().numpy()
        self.edge_index = g.edge_index.flatten().detach().cpu().numpy()
        # Unchange indices and change indices now has to be based on
        # lbl_p
        unchange_indices = np.where(lb == 0)[0]
        change_indices = np.where(lb == 1)[0]
        change_indices = np.array([i for i in change_indices
                          if g.lbl_v[self.edge_index[i]] == 1 and
                          g.visiteds[self.edge_index[i]] == 0], dtype=np.int)
        print(g.lbl_d, self.edge_index[change_indices])
        if len(unchange_indices) == 0:
            self.indices = np.random.choice(
                change_indices, min(N, change_indices.shape[0])).tolist()
        elif len(change_indices) == 0:
            self.indices = np.random.choice(
                unchange_indices,
                min(N, unchange_indices.shape[0])).tolist()
        else:
            self.indices = self.indices = np.random.choice(
                unchange_indices, min(N, unchange_indices.shape[0])).tolist() +\
                np.random.choice(
                    change_indices,
                    min(N, change_indices.shape[0])).tolist()
        self.N = len(self.indices)
        self.rr = 0.3, 1.0

    def __iter__(self):
        self.i = 0
        self.total = 1
        return self

    def __next__(self):
        if self.i >= self.N:
            raise StopIteration
        prob = 1
        # prob = random.uniform(*self.rr)
        while self.i < self.N and prob < 1/self.total:
            self.total += 1
            self.i += 1
            prob = random.random()
        if self.i >= self.N:
            raise StopIteration
        else:
            x = self.i
            self.i += 1
            return (self.indices[x],
                    self.edge_index[self.indices[x]])


class DFSChosenNodeTargets(object):
    def __init__(self, g, N=1):
        lb = g.lbl_t.flatten().detach().cpu().numpy()
        unchange_indices = np.where(lb == 0)[0]
        change_indices = np.where(lb == 1)[0]
        if len(unchange_indices) == 0:
            self.indices = np.random.choice(
                change_indices, min(N, change_indices.shape[0])).tolist()
        elif len(change_indices) == 0:
            self.indices = np.random.choice(
                unchange_indices,
                min(N, unchange_indices.shape[0])).tolist()
        else:
            self.indices = np.random.choice(
                unchange_indices, min(N, unchange_indices.shape[0])).tolist() +\
                np.random.choice(
                    change_indices,
                    min(N, change_indices.shape[0])).tolist()
        print(lb, self.indices)
        self.N = len(self.indices)

    def __iter__(self):
        self.i = 0
        self.total = 1
        return self

    def __next__(self):
        if self.i >= self.N:
            raise StopIteration
        # prob = random.random()
        prob = 1
        while self.i < self.N and prob < 1/self.total:
            self.total += 1
            self.i += 1
            prob = random.random()
        if self.i >= self.N:
            raise StopIteration
        else:
            x = self.i
            self.i += 1
            return (self.indices[x],)

class MatchedNXOOSIterator(object):
    def __init__(self, match_dir, struct_idx):
        self.m_inst_fps = glob.glob(
            f"{match_dir}/matched_nxs_oos_{struct_idx}*")

    def __iter__(self):
        self.i = -1
        self.j = 0
        self.N = len(self.m_inst_fps)
        self.curr = []
        return self

    def __next__(self):
        if self.j >= len(self.curr):
            self.i += 1
            self.j = 0
            self.reload_file = True
        if self.i >= self.N:
            raise StopIteration
        if self.reload_file:
            with open(self.m_inst_fps[self.i], 'rb') as f:
                self.curr = pkl.load(f)
                self.reload_file = False
        x = self.j
        self.j += 1
        return self.curr[x]


class MatchedNXOOSSplitIterator(object):
    def __init__(self, match_dir, match_oos_dir, struct_idx,
                 batch_size=100000, oos_batch_size=10000):
        self.match_oos_dir = match_oos_dir
        self.struct_idx = struct_idx
        self.oos_batch_size = oos_batch_size
        self.ios_batch_size = batch_size
        n_files = len(
            glob.glob(f"{match_dir}/matched_nxs_{struct_idx}_*.pkl"))
        self.m_inst_fps = list([f"{match_dir}/matched_nxs_{struct_idx}_{i}.pkl"
                                for i in range(1, n_files+1)])


    def __iter__(self):
        self.i = -1
        self.j = 0
        self.N = len(self.m_inst_fps)
        self.curr = []
        self.curr_cols = []
        return self

    def __next__(self):
        if self.j >= len(self.curr):
            self.i += 1
            self.j = 0
            self.reload_file = True
        if self.i >= self.N:
            raise StopIteration
        if (self.j % self.ios_batch_size) % self.oos_batch_size == 0 and not self.reload_file:
            # reload curr_cols
            l = (self.j % self.ios_batch_size) // self.oos_batch_size
            i = self.i + 1
            self.curr_cols = [
                    pkl.load(open(f"{self.match_oos_dir}/matched_oos_{self.struct_idx}_{i}_{n}_{l}_{self.oos_batch_size}.pkl", 'rb'))
                    for n in range(self.n_nodes)
                ]

        if self.reload_file:
            with open(self.m_inst_fps[self.i], 'rb') as f:
                self.curr = pkl.load(f)
                self.n_nodes = len(self.curr[0].nodes())
                l = (self.j % self.ios_batch_size) // self.oos_batch_size
                i = self.i + 1
                self.curr_cols = [
                    pkl.load(open(f"{self.match_oos_dir}/matched_oos_{self.struct_idx}_{i}_{n}_{l}_{self.oos_batch_size}.pkl", 'rb'))
                    for n in range(self.n_nodes)
                ]
                self.reload_file = False
        oos_per_batch_idx = (self.j - self.i*self.ios_batch_size) % self.oos_batch_size
        x = self.j
        self.j += 1
        return self.curr[x],  [self.curr_cols[n][oos_per_batch_idx]
                                if oos_per_batch_idx in self.curr_cols[n] else []
                               for n in range(self.n_nodes)]


class DFSChosenNodeVisiteds(object):
    def __init__(self, g, N=1):
        lb = g.lbl_v.flatten().detach().cpu().numpy()
        indices = torch.max(g.state, dim=1)[1].cpu().numpy()
        print(indices, lb)
        unchange_indices = np.where(lb == indices)[0]
        change_indices = np.where(lb != indices)[0]
        if len(unchange_indices) == 0:
            self.indices = np.random.choice(
                change_indices, min(N, change_indices.shape[0])).tolist()
        elif len(change_indices) == 0:
            self.indices = np.random.choice(
                unchange_indices,
                min(N, unchange_indices.shape[0])).tolist()
        else:
            self.indices = np.random.choice(
                unchange_indices, min(N, unchange_indices.shape[0])).tolist() +\
                np.random.choice(
                    change_indices,
                    min(N, change_indices.shape[0])).tolist()
        print(lb, self.indices)
        self.N = len(self.indices)

    def __iter__(self):
        self.i = 0
        self.total = 1
        return self

    def __next__(self):
        if self.i >= self.N:
            raise StopIteration
        # prob = random.random()
        prob = 1
        while self.i < self.N and prob < 1/self.total:
            self.total += 1
            self.i += 1
            prob = random.random()
        if self.i >= self.N:
            raise StopIteration
        else:
            x = self.i
            self.i += 1
            return (self.indices[x],)

class MatchedNXOOSIterator(object):
    def __init__(self, match_dir, struct_idx):
        self.m_inst_fps = glob.glob(
            f"{match_dir}/matched_nxs_oos_{struct_idx}*")

    def __iter__(self):
        self.i = -1
        self.j = 0
        self.N = len(self.m_inst_fps)
        self.curr = []
        return self

    def __next__(self):
        if self.j >= len(self.curr):
            self.i += 1
            self.j = 0
            self.reload_file = True
        if self.i >= self.N:
            raise StopIteration
        if self.reload_file:
            with open(self.m_inst_fps[self.i], 'rb') as f:
                self.curr = pkl.load(f)
                self.reload_file = False
        x = self.j
        self.j += 1
        return self.curr[x]


class MatchedNXOOSSplitIterator(object):
    def __init__(self, match_dir, match_oos_dir, struct_idx,
                 batch_size=100000, oos_batch_size=10000):
        self.match_oos_dir = match_oos_dir
        self.struct_idx = struct_idx
        self.oos_batch_size = oos_batch_size
        self.ios_batch_size = batch_size
        n_files = len(
            glob.glob(f"{match_dir}/matched_nxs_{struct_idx}_*.pkl"))
        self.m_inst_fps = list([f"{match_dir}/matched_nxs_{struct_idx}_{i}.pkl"
                                for i in range(1, n_files+1)])


    def __iter__(self):
        self.i = -1
        self.j = 0
        self.N = len(self.m_inst_fps)
        self.curr = []
        self.curr_cols = []
        return self

    def __next__(self):
        if self.j >= len(self.curr):
            self.i += 1
            self.j = 0
            self.reload_file = True
        if self.i >= self.N:
            raise StopIteration
        if (self.j % self.ios_batch_size) % self.oos_batch_size == 0 and not self.reload_file:
            # reload curr_cols
            l = (self.j % self.ios_batch_size) // self.oos_batch_size
            i = self.i + 1
            self.curr_cols = [
                    pkl.load(open(f"{self.match_oos_dir}/matched_oos_{self.struct_idx}_{i}_{n}_{l}_{self.oos_batch_size}.pkl", 'rb'))
                    for n in range(self.n_nodes)
                ]

        if self.reload_file:
            with open(self.m_inst_fps[self.i], 'rb') as f:
                self.curr = pkl.load(f)
                self.n_nodes = len(self.curr[0].nodes())
                l = (self.j % self.ios_batch_size) // self.oos_batch_size
                i = self.i + 1
                self.curr_cols = [
                    pkl.load(open(f"{self.match_oos_dir}/matched_oos_{self.struct_idx}_{i}_{n}_{l}_{self.oos_batch_size}.pkl", 'rb'))
                    for n in range(self.n_nodes)
                ]
                self.reload_file = False
        oos_per_batch_idx = (self.j - self.i*self.ios_batch_size) % self.oos_batch_size
        x = self.j
        self.j += 1
        return self.curr[x],  [self.curr_cols[n][oos_per_batch_idx]
                                if oos_per_batch_idx in self.curr_cols[n] else []
                               for n in range(self.n_nodes)]
