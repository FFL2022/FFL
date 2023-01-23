
# Parameters.py
#
# Written by Larry Holder (holder@wsu.edu).
#
# Copyright (c) 2017-2021. Washington State University.

import os

class Parameters:

    def __init__(self):
        # User-defined parameters
        self.in_fp = ""       # Store name of input file
        self.out_fn = ""      # Same as in_fp, but with .json removed from end if present
        self.k_beam = 4            # Number of patterns to retain after each expansion of previous patterns; based on value.
        self.n_iters = 1           # Iterations of Subdue's discovery process. If more than 1, Subdue compresses graph with best pattern before next run. If 0, then run until no more compression (i.e., set to |E|).
        self.limit = 100 # Number of patterns considered; default (0) is |E|/2.
        self.max_size = 0              # Maximum size (#edges) of a pattern; default (0) is |E|/2.
        self.min_size = 1              # Minimum size (#edges) of a pattern; default is 1.
        self.n_best = 3              # Number of best patterns to report at end; default is 3.
        self.overlap = "none"         # Extent that pattern instances can overlap (none, vertex, edge)
        self.prune = False            # Remove any patterns that are worse than their parent.
        self.valueBased = False       # Retain all patterns with the top beam best values.
        self.writeCompressed = False  # Write compressed graph after iteration i to file out_fn-compressed-i.json
        self.writePattern = False     # Write best pattern at iteration i to file out_fn-pattern-i.json
        self.writeInstances = False   # Write instances of best pattern at iteration i as one graph to file out_fn-instances-i.json
        self.temporal = False         # Discover static (False) or temporal (True) patterns

    def set_params (self, args):
        """Set parameters according to given command-line args list."""
        self.in_fp = args[-1]
        fn, ext = os.path.splitext(self.in_fp)
        if (ext == '.json'):
            self.out_fn = fn
        else:
            self.out_fn = self.in_fp
        index = 1
        numArgs = len(args)
        while index < (numArgs - 1):
            optionName = args[index]
            if optionName == "--beam":
                index += 1
                self.k_beam = int(args[index])
            if optionName == "--iterations":
                index += 1
                self.n_iters = int(args[index])
            if optionName == "--limit":
                index += 1
                self.limit = int(args[index])
            if optionName == "--maxsize":
                index += 1
                self.max_size = int(args[index])
            if optionName == "--minsize":
                index += 1
                self.min_size = int(args[index])
            if optionName == "--numbest":
                index += 1
                self.n_best = int(args[index])
            if optionName == "--overlap":
                index += 1
                overlap_type = args[index]
                if overlap_type in ["none", "vertex", "edge"]:
                    self.overlap = overlap_type
            if optionName == "--prune":
                self.prune = True
            if optionName == "--valuebased":
                self.valueBased = True
            if optionName == "--writecompressed":
                self.writeCompressed = True
            if optionName == "--writepattern":
                self.writePattern = True
            if optionName == "--writeinstances":
                self.writeInstances = True
            if optionName == "--temporal":
                self.temporal = True
            index += 1

    def print(self):
        print("Parameters:")
        print("  Input File Name: " + self.in_fp)
        print("  Output File Name: " + self.out_fn)
        print("  Beam Width: " + str(self.k_beam))
        print("  Iterations: " + str(self.n_iters))
        print("  Limit: " + str(self.limit))
        print("  Max Size: " + str(self.max_size))
        print("  Min Size: " + str(self.min_size))
        print("  Num Best: " + str(self.n_best))
        print("  Overlap: " + self.overlap)
        print("  Prune: " + str(self.prune))
        print("  Value Based: " + str(self.valueBased))
        print("  Write Compressed: " + str(self.writeCompressed))
        print("  Write Pattern: " + str(self.writePattern))
        print("  Write Instances: " + str(self.writeInstances))
        print("  Temporal: " + str(self.temporal) + "\n")

    def set_defaults_for_graph(self, graph):
        if (self.limit == 0):
            self.limit = int(len(graph.es) / 2)
        if (self.max_size == 0):
            self.max_size = int(len(graph.es) / 2)
        if (self.n_iters == 0):
            self.n_iters = len(graph.es)

    def set_params_from_kwargs(self, **kwargs):
        self.__dict__.update(kwargs)
