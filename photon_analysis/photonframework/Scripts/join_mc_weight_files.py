#!/usr/bin/env python3

import json
import os


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-file", required=True, help="Path to file to store joined results in.")
    parser.add_argument("input_files", nargs="+", help="Path to files to join")

    args = parser.parse_args()

    result_data = {}
    
    for input_file_name in args.input_files:
        with open(input_file_name) as input_file:
            input_data = json.load(input_file)
            for key, value in input_data.items():
                result_data[key] = value

    with open(args.output_file, "w") as output_file:
        json.dump(result_data, output_file)


if __name__ == "__main__":
    main()
