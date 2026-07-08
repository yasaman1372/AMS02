#!/usr/bin/env python3

import os

import numpy as np
import matplotlib.pyplot as plt

from tools.config import get_config
from tools.sample import Sample



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfiles", nargs="+", help="Selection files to load and plot.")
    parser.add_argument("--config", nargs=2, required=True, help="Path to config file and workdir.")
    parser.add_argument("--sample", required=True, help="Sample name to plot selection of.")
    parser.add_argument("--plotdir", default="plots", help="Directory to store resulting plots in.")
    parser.add_argument("--outputprefix", default="Selection", help="Prefix for plots.")

    args = parser.parse_args()

    os.makedirs(args.plotdir, exist_ok=True)

    config_filename, workdir = args.config
    config = get_config(config_filename)

    sample = Sample.load(config, args.sample, workdir)
    sample.load_selections(args.inputfiles)
    sample.plot_selections(resultdir=args.plotdir, outputprefix=args.outputprefix)


if __name__ == "__main__":
    main()
