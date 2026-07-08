#!/usr/bin/env python3

import os

import numpy as np
import awkward as ak
import uproot


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Path to root file.")
    parser.add_argument("--treename", required=True, help="Name of the tree in the file.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--outputprefix", default="EventList", help="Prefix for result files.")
    parser.add_argument("--variables", nargs="+", default=["RunNumber", "EventNumber"])

    args = parser.parse_args()

    os.makedirs(args.resultdir, exist_ok=True)
    with open(os.path.join(args.resultdir, f"{args.outputprefix}.list"), "w") as list_file:
        for filename in args.files:
            with uproot.open(filename) as root_file:
                tree = root_file[args.treename]
                events = tree.arrays(args.variables)
                for event in events:
                    list_file.write(" ".join([str(event[branch]) for branch in args.variables]) + "\n")


if __name__ == "__main__":
    main()

