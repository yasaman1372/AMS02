#!/usr/bin/env python3

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

from tools.roottree import read_tree

# This program will read all events from the root trees,
# select the events with ToF charge around 2,
# create a histogram of the inner tracker charge,
# then plot that histogram and save the plot.


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default= "/Users/yasaman/AMS02/data/MC/electron/ExampleAnalysis_Tree_00000_00047.root", help="Path to root file to read tree from. Can also be path to a file with a list of root files or a pattern of root files (e.g. results/ExampleAnalysisTree*.root)")
    parser.add_argument("--treename", default="ExampleAnalysisTree", help="Name of the tree in the root file.")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Amount of events to read in each step.")

    args = parser.parse_args()

    # create a binning for the charge histogram, with 100 bins (i.e. 101 bin edges)
    charge_binning = np.linspace(0, 2, 101)
    # create an array for the histogram values
    charge_hist = np.zeros(100)
    # loop over chunks of events in the root file(s), since we cannot load all events into memory at the same time
    for events in read_tree(args.filename, args.treename, chunk_size=args.chunk_size):
        print(events)
        # print(events.BetaTof)
        # events is an array of the events in the root tree.
        # Each column is one variable, each row one event.

        # Access the value of one variable for all events like this:
        # events.InnerTrackerCharge
        # or equivalent:
        # events["InnerTrackerCharge"]

        # Access the value of all variables for one event like this:
        # events[0]

        # create selection wht cuts on UpperTofCharge and LowerTofCharge:
        # (events.UpperTofCharge > 1.5) creates an array with one bool value (true/false) for each event.
        # "&" is element-wise logical and, so selection contains one bool value for each event, which is true if
        # all four criteria are true, and false otherwise
        selection = (events.UpperTofCharge > 1.5) & (events.UpperTofCharge < 2.5) & (events.LowerTofCharge > 1.5) & (events.LowerTofCharge < 2.5) 
        print(len(selection))
        # select all events passing these cuts:
        # events[selection] returns a new array with only those events for which the value in selection is true
        events_after_selection = events[selection]

        # create histogram of the charge values of this chunk of events:
        hist_values, hist_edges = np.histogram(ak.to_numpy(events_after_selection.InnerTrackerCharge), bins=charge_binning)
        # add histogram of this chunk to overall histogram:
        charge_hist += hist_values

    # Now the loop over all events is done, plot histogram:

    figure = plt.figure(figsize=(12, 6.15))
    plot = figure.subplots(1, 1)
    plot.set_title("Charge histogram (MC electron)")
    plot.set_xlabel("Inner Tracker Charge")
    plot.set_ylabel("Events")
    # plot as points with errorbars
    plot.errorbar((charge_binning[1:] + charge_binning[:-1]) / 2, charge_hist, np.sqrt(charge_hist), fmt=".")
    # or plot as bars
    plot.step(np.concatenate(([charge_binning[0]], charge_binning)), np.concatenate(([0], charge_hist, [0])), where="post")

    figure.savefig("charge_histogram2.pdf", dpi=250)
    plt.show()


if __name__ == "__main__":
    main()
