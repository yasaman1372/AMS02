#!/usr/bin/env python3

import os
import random

import numpy as np

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("inputlist", help="Path to list file to split.")
    parser.add_argument("--component", nargs=2, metavar=("name", "weight"), required=True, dest="components", action="append", help="Name and weight of component to split into.")

    args = parser.parse_args()

    with open(args.inputlist) as input_file:
        elements = input_file.readlines()
    random.shuffle(elements)

    components = {key: float(value) for key, value in args.components}
    total_weight = sum(components.values())
    cumulative_weights = np.cumsum(list(components.values()))

    edges = [None] + [int(weight / total_weight * len(elements)) for weight in cumulative_weights]
    edges[-1] = None
    for component, (begin, end) in zip(components, zip(edges[:-1], edges[1:])):
        component_elements = elements[slice(begin, end)]
        with open(f"{component}.list", "w") as output_file:
            output_file.write("".join(component_elements))


if __name__ == "__main__":
    main()
