#!/usr/bin/env python3

import os
import multiprocessing as mp

import numpy as np
import awkward as ak
import uproot

from tools.config import get_config
from tools.roottree import read_tree
from tools.sample import Sample


def initialize_tree(file, treename, branch_type_dict):
    file.mktree(treename, branch_type_dict, title=treename)


def make_counter_branch(branch_name):
    def _counter_branch(events):
        return ak.num(events[branch_name])
    return _counter_branch


def handle_file(arg):
    filename, treename, chunk_size, rank, nranks, kwargs = arg

    additional_variables = kwargs["additional_variables"]
    config = kwargs["config"]
    sample_name = kwargs["sample"]
    workdir = kwargs["workdir"]
    resultdir = kwargs["resultdir"]
    outputprefix = kwargs["outputprefix"]
    energy_range = kwargs["energy_range"]
    time_range = kwargs["time_range"]
    verbose = kwargs["verbose"]
    nbytes_min = kwargs["nbytes_min"]
    result_treename = kwargs["result_treename"]
    is_mc = kwargs["is_mc"]
    debug = kwargs["debug"]
    debug_eventlist = kwargs["debug_eventlist"]

    branches_to_write = {"RunNumber": "RunNumber", "EventNumber": "EventNumber", "McWeight": "McWeight", "TotalWeight": "TotalWeight"}
    if additional_variables is not None:
        for variable in additional_variables:
            branches_to_write[variable] = variable

    sample = Sample.load(config, sample_name, workdir)

    if energy_range is not None:
        sample.set_energy_range(*energy_range)
    if time_range is not None:
        sample.set_time_range(*time_range)

    for estimator_name, estimator_variable in sample.estimators.items():
        branches_to_write[estimator_name] = estimator_variable
    if is_mc:
        for estimator_name, estimator_variable in sample.mc_estimators.items():
            branches_to_write[f"Mc{estimator_name}"] = estimator_variable

    branches_to_read, derived_branches, derived_weight_branches = sample.get_branches(list(branches_to_write.values()))

    event_cache = None

    trees_initialized = False

    with uproot.recreate(os.path.join(resultdir, f"{outputprefix}_{rank}.root")) as root_file:
        for events in sample.read_tree(filename, treename, branches_to_write.values(), rank=rank, nranks=nranks, chunk_size=chunk_size, verbose=verbose, prefix=outputprefix, debug=debug, apply_eventlist=debug_eventlist):
            if len(events) == 0:
                continue
            if not trees_initialized:
                for branch in branches_to_write.values():
                    # delazify the branch
                    events[branch]
                branch_types = events.get_dtypes()
                irregular_branches = [branch for branch in branches_to_write if type(branch_types[branches_to_write[branch]]).__name__ == "ListType"]
                counter_branches = [f"n{branch_name}" for branch_name in irregular_branches]
                for counter_branch in counter_branches:
                    branches_to_write[counter_branch] = counter_branch
                    branch_types[counter_branch] = np.uint16
                var_type_dict = {var: branch_types[branches_to_write[var]] for var in branches_to_write}
                initialize_tree(root_file, result_treename, var_type_dict)
                trees_initialized = True

            for branch in irregular_branches:
                counter_branch = f"n{branch}"
                events.add_field(counter_branch, make_counter_branch(branch))
                events[counter_branch]

            #sample_events = ak.to_packed(events)
            if len(events) == 0:
                continue
            reduced_array = ak.to_packed(ak.Array({branch_name: events[branches_to_write[branch_name]] for branch_name in list(branches_to_write.keys()) + counter_branches}))
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
    parser.add_argument("--treename", default="PhotonTree", help="Tree name to read from root files file.")
    parser.add_argument("--result-treename", default="GammaTree", help="Tree name to read from root files file.")
    parser.add_argument("--config", required=True, nargs=2, help="Path to analysis config file and workdir.")
    parser.add_argument("--sample", required=True, help="Sample name to select and write into tree.")
    parser.add_argument("--energy-range", type=float, nargs=2, help="Energy range [start, end] in GeV.")
    parser.add_argument("--time-range", type=int, nargs=2, help="ISS time range [start, end] as timestamps.")
    parser.add_argument("--outputprefix", default="GammaTree", help="Prefix for the reduced trees.")
    parser.add_argument("--resultdir", default="results", help="Directory to store the reduced trees in.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of parallel processes.")
    parser.add_argument("--parallel", type=int, nargs=2, default=(0, 1), help="Index of this job and total number of parallel jobs.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Number of events per chunk to read.")
    parser.add_argument("--nbytes-min", type=int, default=int(1e6), help="Minimum number of bytes to write a basket.")
    parser.add_argument("--variables", nargs="+", help="Additional variables to store in the reduced trees.")
    parser.add_argument("--is-mc", action="store_true", help="Also write MC estimators to resulting gamma tree.")
    parser.add_argument("--quiet", "-q", dest="verbose", action="store_false", help="Don't output verbose progress information.")
    parser.add_argument("--debug-eventlist", help="Debug selection on certain events. Expects a list file of run and event numbers.")

    args = parser.parse_args()

    config_file, workdir = args.config
    config = get_config(config_file)

    parallel = args.parallel
    parallel_index, parallel_total = parallel

    debug = False
    debug_eventlist = None
    if args.debug_eventlist is not None:
        debug = True
        debug_eventlist = []
        with open(args.debug_eventlist) as list_file:
            for line in list_file:
                run_number, event_number = map(int, line.strip().split(" "))
                debug_eventlist.append((run_number, event_number))
        debug_eventlist = np.array(debug_eventlist)

    os.makedirs(args.resultdir, exist_ok=True)

    with mp.get_context("fork" if args.verbose else "spawn").Pool(args.nprocesses) as pool:
        pool_args = make_args(args.tree, args.treename, args.chunk_size, args.nprocesses, parallel, result_treename=args.result_treename, additional_variables=args.variables, config=config, sample=args.sample, resultdir=args.resultdir, energy_range=args.energy_range, time_range=args.time_range, is_mc=args.is_mc, outputprefix=args.outputprefix, workdir=workdir, nbytes_min=args.nbytes_min, verbose=args.verbose, debug=debug, debug_eventlist=debug_eventlist)
        for rank in pool.imap_unordered(handle_file, pool_args):
            pass


if __name__ == "__main__":
    main()
