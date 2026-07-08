#!/usr/bin/env python3

import os

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="PhotonTree", help="Filename prefix.")
    parser.add_argument("--infixes", nargs="+", default=[], help="Filename infixes for which to create separate lists.")
    parser.add_argument("--suffix", default=".root", help="Filename ending.")
    parser.add_argument("--directories", nargs="+", help="Directories to search in.")

    args = parser.parse_args()

    tree_files = [
        os.path.join(dir, filename)
        for dir in args.directories
        for filename in os.listdir(dir)
        if filename.startswith(args.prefix) and filename.endswith(args.suffix)
    ]

    with open(f"trees.list", "w") as list_file:
        list_file.write("\n".join(sorted(tree_files)) + "\n")

    for infix in args.infixes:
        matching_files = [filename for filename in tree_files if infix in filename]
        with open(f"{args.prefix}{infix}.list", "w") as list_file:
            list_file.write("\n".join(sorted(matching_files)) + "\n")

if __name__ == "__main__":
    main()
