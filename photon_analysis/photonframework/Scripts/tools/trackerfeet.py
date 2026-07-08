#!/usr/bin/env python3

import os
from struct import Struct

import numpy as np

def get_feet_filename():
    assert "ACSOFTDIR" in os.environ
    return os.path.join(os.environ["ACSOFTDIR"], "acqt", "data", "TrackerFootPositions.dat")

def read_tracker_feet_file(filename=None):
    if filename is None:
        filename = get_feet_filename()
    xs = {layer: [] for layer in range(9)}
    ys = {layer: [] for layer in range(9)}
    with open(filename, "rb") as file:
        struct = Struct(">iiiiff")
        for (layer_index, tkid, foot_index, side, x, y) in struct.iter_unpack(file.read()):
            xs[layer_index].append(x)
            ys[layer_index].append(y)
    xs = {layer: np.array(xs[layer]) for layer in range(9)}
    ys = {layer: np.array(ys[layer]) for layer in range(9)}
    return xs, ys


if __name__ == "__main__":
    read_tracker_feet_file()
