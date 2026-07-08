#!/usr/bin/env python3

import os
import re

import numpy as np


def expand_col_name(name):
    if not name.endswith("]"):
        yield name
        return
    prefix = name[:name.index("[")]
    counts = [
        int(match)
        for match in re.findall("\\[(?P<num>[0-9]+)\\]", name)
    ]
    yield from expand_col_suffixes(prefix, counts)

def expand_col_suffixes(prefix, counts):
    if len(counts) == 0:
        yield prefix
    else:
        for index in range(counts[0]):
            yield from expand_col_suffixes(f"{prefix}_{index}", counts[1:])


COLUMNS_55 = ["Time", "Run", "RunTag", "FirstEventIndex", "LastEventIndex", "MicrosecondFirstEvent", "MicrosecondLastEvent", "MeanNumberOfTrdHits", "MeanNumberOfTrackerClusters", "LiveTime", "Cutoff25N", "Cutoff25P", "Cutoff30N", "Cutoff30P", "Cutoff35N", "Cutoff35P", "Cutoff40N", "Cutoff40P", "MPVHeliumRigidity", "GeographicLatitude", "GeographicLongitude", "Altitude", "Zenith", "GalacticLatitude", "GalacticLongitude", "Events", "MissedEvents", "Level1Triggers", "Particles", "CutoffIGRF25N", "CutoffIGRF25P", "CutoffIGRF30N", "CutoffIGRF30P", "CutoffIGRF35N", "CutoffIGRF35P", "CutoffIGRF40N", "CutoffIGRF40P", "LowestCutoffIGRF25N", "LowestCutoffIGRF25P", "LowestCutoffIGRF40N", "LowestCutoffIGRF40P", "NumberOfL1XHits", "NumberOfL1YHits", "NumberOfL)XHits", "NumberOfL9YHits", "AlignmentDeltaPGMAL1X", "AlignmentDeltaPGMAL1Y", "AlignmentDeltaPGMAL1Z", "AlignmentDeltaPGMAL9X", "AlignmentDeltaPGMAL9Y", "AlignmentDeltaPGMAL9Z", "EventsWithHardwareError", "FirstEventUTC", "LastEventUTC", "StatusBits"]
COLUMNS_54 = COLUMNS_55[:2] + COLUMNS_55[3:]

COLUMNS = {55: COLUMNS_55, 54: COLUMNS_54}

def transform_longitude(value):
    while value > 180:
        value -= 360
    return value

def transform_altitude(value):
    return max(((value / 100) - 6378137.0) / 1000, 0)

def identity(value):
    return value

DTYPES = {
    "Time": np.uint32,
    "Run": np.uint32,
    "RunTag": np.uint16,
    "FirstEventIndex": np.uint32,
    "LastEventIndex": np.uint32,
    "MicrosecondFirstEvent": np.uint32,
    "MicrosecondLastEvent": np.uint32,
    "MeanNumberOfTrdHits": np.float32,
    "MeanNumberOfTrackerClusters": np.float32,
    "LiveTime": np.float32,
    "Cutoff25N": np.float32,
    "Cutoff25P": np.float32,
    "Cutoff30N": np.float32,
    "Cutoff30P": np.float32,
    "Cutoff35N": np.float32,
    "Cutoff35P": np.float32,
    "Cutoff40N": np.float32,
    "Cutoff40P": np.float32,
    "MPVHeliumRigidity": np.float32,
    "GeographicLatitude": np.float32,
    "GeographicLongitude": np.float32,
    "Altitude": np.float32,
    "Zenith": np.float32,
    "GalacticLatitude": np.float32,
    "GalacticLongitude": np.float32,
    "Events": np.uint32,
    "MissedEvents": np.uint32,
    "Level1Triggers": np.uint32,
    "Particles": np.uint32,
    "CutoffIGRF25N": np.float32,
    "CutoffIGRF25P": np.float32,
    "CutoffIGRF30N": np.float32,
    "CutoffIGRF30P": np.float32,
    "CutoffIGRF35N": np.float32,
    "CutoffIGRF35P": np.float32,
    "CutoffIGRF40N": np.float32,
    "CutoffIGRF40P": np.float32,
    "LowestCutoffIGRF25N": np.float32,
    "LowestCutoffIGRF25P": np.float32,
    "LowestCutoffIGRF40N": np.float32,
    "LowestCutoffIGRF40P": np.float32,
    "NumberOfL1XHits": np.uint32,
    "NumberOfL1YHits": np.uint32,
    "NumberOfL)XHits": np.uint32,
    "NumberOfL9YHits": np.uint32,
    "AlignmentDeltaPGMAL1X": np.float32,
    "AlignmentDeltaPGMAL1Y": np.float32,
    "AlignmentDeltaPGMAL1Z": np.float32,
    "AlignmentDeltaPGMAL9X": np.float32,
    "AlignmentDeltaPGMAL9Y": np.float32,
    "AlignmentDeltaPGMAL9Z": np.float32,
    "EventsWithHardwareError": np.uint32,
    "FirstEventUTC": np.float64,
    "LastEventUTC": np.float64,
    "StatusBits": np.uint32,
}


TRANSFORMS = {
    "GeographicLongitude": transform_longitude,
    "Altitude": transform_altitude,
}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("rti_dir")
    parser.add_argument("--columns", nargs="+", default=["Time", "GeographicLongitude", "GeographicLatitude", "Altitude"])
    parser.add_argument("--outputprefix", default="LatLon")
    parser.add_argument("--resultdir", default="results")

    args = parser.parse_args()

    results = []

    if os.path.isdir(args.rti_dir):
        filenames = sorted([os.path.join(args.rti_dir, filename) for filename in os.listdir(args.rti_dir)])
    else:
        with open(args.rti_dir) as list_file:
            filenames = [line.strip() for line in list_file]
    for file_index, filename in enumerate(filenames):
        entries = []
        with open(filename) as rti_file:
            next(rti_file)
            for line in rti_file:
                parts = line.strip().split(" ")
                if len(parts) == 2:
                    continue
                raw_values = dict(zip(COLUMNS[len(parts)], parts))
                entries.append([TRANSFORMS.get(col, identity)(DTYPES[col](raw_values[col])) for col in args.columns])
        if entries:
            results.append(np.core.records.fromrecords(entries, names=args.columns, dtype=[(col, DTYPES[col]) for col in args.columns]))
        print(f"{file_index:>5}/{len(filenames):>5} {sum(r.size * r.itemsize / 2**20 for r in results):.1f} MB")

    result = np.concatenate(results)
    os.makedirs(args.resultdir, exist_ok=True)
    np.savez_compressed(os.path.join(args.resultdir, f"{args.outputprefix}.npz"), data=result)


if __name__ == "__main__":
    main()

