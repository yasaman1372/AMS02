#!/usr/bin/env python3

import os

import numpy as np
import awkward as ak
import uproot

from tools.roottree import read_tree, parse_filenames
from tools.utilities import merge_run_and_event_number


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", nargs="+", required=True, help="Path to ROOT files containing selection trees.")
    parser.add_argument("--treename", default="SelectionTree", help="Name of tree in ROOT files.")
    parser.add_argument("--eventlist", required=True, help="List of runs and events to check.")

    args = parser.parse_args()

    with open(args.eventlist) as run_event_file:
        run_numbers = []
        event_numbers = []
        for line in run_event_file:
            run_number, event_number = map(int, line.strip().split(" "))
            run_numbers.append(run_number)
            event_numbers.append(event_number)
    runevent_ref = merge_run_and_event_number(np.array(run_numbers), np.array(event_numbers))

    filenames = parse_filenames(args.tree)
    cut_statistics = {}
    first_file = filenames[0]
    with uproot.open(first_file) as root_file:
        selection_names = str(root_file["Selections"]).split(",")
        cut_labels = {}
        numbers_of_cuts = {}
        for selection_name in selection_names:
            cut_labels[selection_name] = []
            numbers_of_cuts[selection_name] = int(str(root_file[f"NumberOfCuts_{selection_name}"]))
            for cut_index in range(numbers_of_cuts[selection_name]):
                cut_labels[selection_name].append(str(root_file[f"Cut_{selection_name}_{cut_index}"]))
                cut_statistics[(selection_name, cut_index)] = 0


    for events in read_tree(args.tree, args.treename):
        runevent = merge_run_and_event_number(ak.to_numpy(events.RunNumber), ak.to_numpy(events.EventNumber))
        selection = np.isin(runevent, runevent_ref, assume_unique=True)
        events = events[selection]
        for event in events:
            print(f"Event {event.RunNumber} {event.EventNumber}")
            for selection_name in selection_names:
                pattern = event[f"CutPattern_{selection_name}"]
                number_of_cuts = numbers_of_cuts[selection_name]
                if pattern >= 2**number_of_cuts:
                    print(f"Event {event.RunNumber} {event.EventNumber} was not passed to {selection_name}")
                    break
                elif pattern == 2**number_of_cuts - 1:
                    print(f"Event {event.RunNumber} {event.EventNumber} passed {selection_name}")
                    continue
                for cut_index in range(number_of_cuts):
                    passed = (pattern & (2**cut_index)) > 0
                    if not passed:
                        print(f"Event {event.RunNumber} {event.EventNumber} failed {selection_name} cut {cut_labels[selection_name][cut_index]}")
                        cut_statistics[(selection_name, cut_index)] += 1

    print("Summary:")
    for (selection_name, cut_index), failed_events in cut_statistics.items():
        print(f"{cut_labels[selection_name][cut_index]} failed {failed_events} events")


if __name__ == "__main__":
    main()
