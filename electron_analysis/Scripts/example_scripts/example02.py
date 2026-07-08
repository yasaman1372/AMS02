#!/usr/bin/env python3

import multiprocessing as mp
import os

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

from tools.roottree import read_tree

# This script will do the same as example01, but with multiprocessing.
# This is necessary to read more and larger root files.
# Each process will read just one part of the files and create a histogram for those.
# All single histograms will be saved to disk.
# Once all processes are done, all histograms have to be loaded from disk and merged.


def handle_file(arg):
    # This function is called once for each process.
    # rank and nranks specifies which process out of how many processes this one is.
    # Based on that, read_tree knows which files to read.

    filename, treename, chunk_size, rank, nranks, kwargs = arg

    resultdir = kwargs["resultdir"]

    charge_binning = np.linspace(0, 4, 101)
    charge_hist = np.zeros(100)

    for events in read_tree(filename, treename, chunk_size=chunk_size, rank=rank, nranks=nranks):
        selection = (events.UpperTofCharge > 1.5) & (events.UpperTofCharge < 2.5) & (events.LowerTofCharge > 1.5) & (events.LowerTofCharge < 2.5)
        events_after_selection = events[selection]

        hist_values, hist_edges = np.histogram(ak.to_numpy(events_after_selection.InnerTrackerCharge), bins=charge_binning)
        charge_hist += hist_values

    # this process is done, save result
    np.savez(os.path.join(resultdir, f"results_{rank}.npz"), charge_binning=charge_binning, charge_hist=charge_hist)

    

def make_args(filename, treename, chunk_size, nranks, **kwargs):
    # this function creates the arguments for handle_file for each process
    for rank in range(nranks):
        yield (filename, treename, chunk_size, rank, nranks, kwargs)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Path to root file to read tree from. Can also be path to a file with a list of root files or a pattern of root files (e.g. results/ExampleAnalysisTree*.root)")
    parser.add_argument("--treename", default="ExampleAnalysisTree", help="Name of the tree in the root file.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")
    parser.add_argument("--nprocesses", type=int, default=os.cpu_count(), help="Number of processes to use in parallel.")
    parser.add_argument("--resultdir", default="results", help="Directory to store plots and result files in.")

    args = parser.parse_args()

    # make sure the directory we will store results in exists
    os.makedirs(args.resultdir, exist_ok=True)
    
    # create pool of worker processes
    with mp.Pool(args.nprocesses) as pool:
        # create arguments for the individual processes
        pool_args = make_args(args.filename, args.treename, args.chunk_size, args.nprocesses, resultdir=args.resultdir)
        # and execute handle_file for each process
        for _ in pool.imap_unordered(handle_file, pool_args):
            pass

    # now all processes are done, load the results and merge them
    charge_binning = None
    charge_hist = None
    for rank in range(args.nprocesses):
        filename = os.path.join(args.resultdir, f"results_{rank}.npz")
        with np.load(filename) as result_file:
            if charge_binning is None:
                charge_binning = result_file["charge_binning"]
            else:
                # make sure all processes used the same binning
                assert np.all(charge_binning == result_file["charge_binning"])
            if charge_hist is None:
                charge_hist = result_file["charge_hist"]
            else:
                charge_hist += result_file["charge_hist"]

    # now save merged result
    np.savez(os.path.join(args.resultdir, "results.npz"), charge_binning=charge_binning, charge_hist=charge_hist)

    # and create a plot of the merged result

    figure = plt.figure(figsize=(12, 6.15))
    plot = figure.subplots(1, 1)
    plot.set_title("Charge histogram")
    plot.set_xlabel("Inner Tracker Charge")
    plot.set_ylabel("Events")
    plot.errorbar((charge_binning[1:] + charge_binning[:-1]) / 2, charge_hist, np.sqrt(charge_hist), fmt=".")
    plot.step(np.concatenate(([charge_binning[0]], charge_binning)), np.concatenate(([0], charge_hist, [0])), where="post")

    figure.savefig("charge_histogram.png", dpi=250)
    plt.close(figure)


if __name__ == "__main__":
    main()
