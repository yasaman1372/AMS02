#!/usr/bin/env python3

from glob import glob
import os

import numpy as np

from tools.histograms import Histogram, WeightedHistogram, load_histogram, list_histograms

def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("inputfiles", nargs="+", help="Files to merge.")
    parser.add_argument("--resultdir", default=".", help="Directory to store merged file in.")
    parser.add_argument("--outputprefix", default="Merged", help="Prefix for merged file.")
    parser.add_argument("--keys", nargs="+", help="Names of histograms to merge.")

    args = parser.parse_args()

    keys = args.keys
    if keys is None:
        with np.load(args.inputfiles[0]) as file:
            keys = list_histograms(file)

    histograms = {key: None for key in keys}
    for filename in args.inputfiles:
        with np.load(filename) as input_file:
            for key in keys:
                hist = load_histogram(input_file, key)
                if histograms[key] is None:
                    histograms[key] = hist
                else:
                    histograms[key] += hist

    result = {}
    for key, hist in histograms.items():
        hist.add_to_file(result, key)

    np.savez_compressed(os.path.join(args.resultdir, f"{args.outputprefix}.npz"), **result)


if __name__ == "__main__":
    main()
