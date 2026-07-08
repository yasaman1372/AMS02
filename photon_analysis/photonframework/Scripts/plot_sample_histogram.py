#!/usr/bin/env python3

import os

import numpy as np
import matplotlib.pyplot as plt

from create_sample_histogram import ResultHists


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfiles", nargs="+", help="Files to merge.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store plots in.")

    args = parser.parse_args()

    results = ResultHists.load_files(args.inputfiles)

    os.makedirs(args.plotdir, exist_ok=True)
    results.plot(args.plotdir)
    results.resultdir = args.plotdir
    results.save()


if __name__ == "__main__":
    main()
