#!/usr/bin/env python3

import os
import multiprocessing as mp

import numpy as np
import awkward as ak
import uproot

from tools.binnings import Binnings
from tools.config import get_config
from tools.roottree import read_tree
from tools.variables import DerivedVariables


def initialize_tree(file, treename, branch_type_dict):
    file.mktree(treename, branch_type_dict, title=treename)


def make_counter_branch(branch_name):
    def _counter_branch(events):
        return ak.num(events[branch_name])
    return _counter_branch


def handle_file(arg):
    filename, treename, chunk_size, rank, nranks, kwargs = arg

    config = kwargs["config"]
    workdir = kwargs["workdir"]
    resultdir = kwargs["resultdir"]
    outputprefix = kwargs["outputprefix"]
    verbose = kwargs["verbose"]
    nbytes_min = kwargs["nbytes_min"]
    result_treename = kwargs["result_treename"]

    branches_to_write = ["ISSTerrestrialLongitude", "ISSTerrestrialLatitude", "AMSGalacticLongitude", "AMSGalacticLatitude", "TriggerLiveTime", "UTCTime"]

    binnings = Binnings.from_config(config)
    derived_variables = DerivedVariables(config=config, workdir=workdir, energy_estimator="Energy", binnings=binnings)

    branches_to_read, derived_branches = derived_variables.resolve_branches(branches_to_write)

    event_cache = None

    trees_initialized = False

    with uproot.recreate(os.path.join(resultdir, f"{outputprefix}_{rank}.root")) as root_file:
        for events in read_tree(filename, treename, branches=branches_to_read, rank=rank, nranks=nranks, chunk_size=chunk_size, verbose=verbose):
            if len(events) == 0:
                continue
            for branch in derived_branches:
                events.add_field(branch, derived_variables.functions[branch])
            if not trees_initialized:
                for branch in branches_to_write:
                    # delazify the branch
                    events[branch]
                branch_types = events.get_dtypes()
                irregular_branches = [branch for branch in branches_to_write if type(branch_types[branch]).__name__ == "ListType"]
                counter_branches = [f"n{branch_name}" for branch_name in irregular_branches]
                for counter_branch in counter_branches:
                    branches_to_write.append(counter_branch)
                    branch_types[counter_branch] = np.uint16
                var_type_dict = {var: branch_types[var] for var in branches_to_write}
                initialize_tree(root_file, result_treename, var_type_dict)
                trees_initialized = True

            for branch in irregular_branches:
                counter_branch = f"n{branch}"
                events.add_field(counter_branch, make_counter_branch(branch))
                events[counter_branch]

            #sample_events = ak.to_packed(events)
            if len(events) == 0:
                continue
            reduced_array = ak.to_packed(ak.Array({branch_name: events[branch_name] for branch_name in list(branches_to_write) + counter_branches}))
            if event_cache is None:
                event_cache = reduced_array
            else:
                event_cache = np.concatenate((event_cache, reduced_array))
            if event_cache.nbytes >= nbytes_min and len(event_cache) > 0:
                root_file[result_treename].extend(event_cache)
                event_cache = None

        if event_cache is not None:
            root_file[result_treename].extend(event_cache)


def make_args(filename, treename, chunk_size, nranks, parallel, **kwargs):
    parallel_index, parallel_total = parallel
    assert parallel_index >= 0 and parallel_index < parallel_total
    for rank in range(parallel_index * nranks, (parallel_index + 1) * nranks):
        yield (filename, treename, chunk_size, rank, nranks * parallel_total, kwargs)


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--tree", nargs="+", required=True, help="Path to tree file(s).")
    parser.add_argument("--treename", default="IssTree", help="Tree name to read from root files file.")
    parser.add_argument("--result-treename", default="ExposureTree", help="Tree name to read from root files file.")
    parser.add_argument("--config", required=True, nargs=2, help="Path to analysis config file and workdir.")
    parser.add_argument("--outputprefix", default="ExposureTree", help="Prefix for the reduced trees.")
    parser.add_argument("--resultdir", default="results", help="Directory to store the reduced trees in.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of parallel processes.")
    parser.add_argument("--parallel", type=int, nargs=2, default=(0, 1), help="Index of this job and total number of parallel jobs.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Number of events per chunk to read.")
    parser.add_argument("--nbytes-min", type=int, default=int(1e6), help="Minimum number of bytes to write a basket.")
    parser.add_argument("--quiet", "-q", dest="verbose", action="store_false", help="Don't output verbose progress information.")

    args = parser.parse_args()

    config_file, workdir = args.config
    config = get_config(config_file)

    parallel = args.parallel
    parallel_index, parallel_total = parallel

    os.makedirs(args.resultdir, exist_ok=True)

    with mp.get_context("fork" if args.verbose else "spawn").Pool(args.nprocesses) as pool:
        pool_args = make_args(args.tree, args.treename, args.chunk_size, args.nprocesses, parallel, result_treename=args.result_treename, config=config, resultdir=args.resultdir, outputprefix=args.outputprefix, workdir=workdir, nbytes_min=args.nbytes_min, verbose=args.verbose)
        for rank in pool.imap_unordered(handle_file, pool_args):
            pass


if __name__ == "__main__":
    main()
