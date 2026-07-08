#!/usr/bin/env python3

import json
import os

from tools.config import get_config
from tools.sample import Sample


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("sample", help="Name of sample to export selections for.")
    parser.add_argument("--config", nargs=2, required=True, help="Path to configuration file and working directory.")
    parser.add_argument("--resultdir", default="results", help="Directory to store results in.")

    args = parser.parse_args()

    config_filename, workdir = args.config
    config = get_config(config_filename)

    sample = Sample.load(config, args.sample, workdir)

    os.makedirs(args.resultdir, exist_ok=True)

    with open(os.path.join(args.resultdir, f"{args.sample}_selection.txt"), "w") as selection_file:
        for selection_name, selection in sample.selections.items():
            selection_file.write(f"{selection_name}\n")
            for cut_variable, cut in selection.cuts.items():
                selection_file.write(f" {cut.label}\n")


if __name__ == "__main__":
    main()
