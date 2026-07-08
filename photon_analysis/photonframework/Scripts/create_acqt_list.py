#!/usr/bin/env python3

import os

import numpy as np

def first_run(path):
    filename = os.path.basename(path)
    return int(filename.split("_")[0])

def last_run(path):
    filename = os.path.basename(path)
    if filename.endswith(".macqt"):
        return int(filename.split("_")[1])
    return int(filename.split("_")[0])


def list_acqt_files(acqt_datadir, datatype, dataset, min_run=None, max_run=None, run_list=None, max_files=None):
    dir = os.path.join(acqt_datadir, datatype, dataset, "DATA", "all")
    if not os.path.isdir(dir):
        raise FileNotFoundError(f"{dir!r} does not exist.")

    files = [os.path.join(dir, filename) for filename in sorted(os.listdir(dir)) if filename.endswith((".acqt", ".macqt"))]
    if any(filename.endswith(".macqt") for filename in files):
        files = [filename for filename in files if filename.endswith(".macqt")]

    if min_run is not None:
        files = [filename for filename in files if first_run(filename) >= min_run]
    if max_run is not None:
        files = [filename for filename in files if last_run(filename) <= max_run]
    if run_list is not None:
        run_list = np.array(run_list)
        files = [filename for filename in files if np.digitize(first_run(filename), run_list, right=True) != np.digitize(last_run(filename), run_list)]

    if max_files is not None:
        files = files[:max_files]

    return files


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--acqt-datadir", default=os.environ["ACQTDATADIR"])
    parser.add_argument("datatype")
    parser.add_argument("dataset")
    parser.add_argument("-o", "--output", default="acqtfiles.list")
    parser.add_argument("--max-files", type=int) 
    parser.add_argument("--min-run", type=int)
    parser.add_argument("--max-run", type=int)
    parser.add_argument("--run-event-list", help="File containing list of whitespace separated run and event number.")

    args = parser.parse_args()

    run_list = None
    if args.run_event_list is not None:
        with open(args.run_event_list) as run_event_file:
            run_list = sorted(set([int(line.strip().split(" ")[0]) for line in run_event_file]))

    files = list_acqt_files(acqt_datadir=args.acqt_datadir, datatype=args.datatype, dataset=args.dataset, min_run=args.min_run, max_run=args.max_run, run_list=run_list, max_files=args.max_files)

    with open(args.output, "w") as list_file:
        list_file.write("\n".join(files) + "\n")


if __name__ == "__main__":
    main()
