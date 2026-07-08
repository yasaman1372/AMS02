#!/usr/bin/env python3

import os

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("triggers_files", nargs="+")
    parser.add_argument("--charge", type=int, required=True, help="Charge of the simulated particle.")

    args = parser.parse_args()

    with open("triggers.txt", "w") as result_file:
        for filename in args.triggers_files:
            with open(filename) as file:
                for line in file:
                    p_min, p_max, triggers = map(float, line.split(" "))
                    r_min = p_min / args.charge
                    r_max = p_max / args.charge
                    result_file.write(f"{r_min} {r_max} {triggers}\n")


if __name__ == "__main__":
    main()
