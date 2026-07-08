#!/usr/bin/env python3

import os

from tools.binnings import Binning, Binnings
from tools.config import get_config


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("binning")
    parser.add_argument("--config", nargs=2, required=True)

    args = parser.parse_args()

    config_filename, workdir = args.config
    config = get_config(config_filename)
    binnings = Binnings((config, workdir))

    binning = binnings.variable_binnings.get(args.binning, binnings.special_binnings.get(args.binning, None))
    if binning is None:
        raise ValueError("Cannot find binning {args.binning!r}")

    with open(f"{args.binning}.txt", "w") as binning_file:
        binning_file.write(" ".join(map(str, binning.edges[1:-1])) + "\n")


if __name__ == "__main__":
    main()
