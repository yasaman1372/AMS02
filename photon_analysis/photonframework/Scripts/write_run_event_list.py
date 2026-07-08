#!/usr/bin/env python3

import os

from tools.roottree import read_tree


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="+", help="Path to files to read.")
    parser.add_argument("--treename", default="PhotonTree", help="Name of tree to read.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")
    parser.add_argument("--outputprefix", default="TreeLists", help="Prefix for result files.")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Number of events to read per chunk.")
    parser.add_argument("--run-branch", default="RunNumber", help="Name of branch containing the run number.")
    parser.add_argument("--event-branch", default="EventNumber", help="Name of branch containing the event number.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Output progress.")

    args = parser.parse_args()

    os.makedirs(args.resultdir, exist_ok=True)

    branches = [args.run_branch, args.event_branch]

    with open(os.path.join(args.resultdir, f"{args.outputprefix}.list"), "w") as run_event_file:
        for events in read_tree(args.filename, treename=args.treename, chunk_size=args.chunk_size, branches=branches, verbose=args.verbose):
            run_numbers = events[args.run_branch]
            event_numbers = events[args.event_branch]
            for run, event in zip(run_numbers, event_numbers):
                run_event_file.write(f"{run} {event}\n")
                
    

if __name__ == "__main__":
    main()
