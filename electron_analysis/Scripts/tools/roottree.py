#!/usr/bin/env python3

from contextlib import contextmanager
from glob import glob
import multiprocessing as mp

import numpy as np
import uproot
import awkward as ak


class Counter:
    def __init__(self, value=0):
        self.val = mp.Value('L', value)

    def __iadd__(self, value):
        with self.val.get_lock():
            self.val.value += value

    def add(self, value):
        self += value

    def reset(self):
        with self.val.get_lock():
            self.val.values = 0

    @property
    def value(self):
        return self.val.value


event_counter = Counter()
total_event_counter = Counter()


def chunk_iterator(start, stop, chunk_size):
    for index in range(start, stop, chunk_size):
        yield index, min(index + chunk_size, stop)


def read_tree_from_file(filename, treename, chunk_size=1000000, start=None, stop=None, branches=None, variables=None, cut=None, verbose=True):
    with uproot.open(f"{filename}:{treename}") as tree:
        if branches is None:
            branches = [b.name for b in tree.branches]
        if variables is not None:
            branches.extend(list(variables.keys()))
        if start is None or stop is None:
            start = 0
            stop = tree.num_entries
        for chunk_start, chunk_stop in chunk_iterator(start, stop, chunk_size):
            chunk = tree.arrays(expressions=branches, entry_start=chunk_start, entry_stop=chunk_stop, array_cache=None, cut=cut, aliases=variables)
            if verbose:
                event_counter.add(len(chunk))
                current = event_counter.value
                total = total_event_counter.value
                percentage = (current / total * 100) if total > 0 else 0
                print(f"{current:>10} / {total:>10} ({percentage:>5.1f}%)")
            yield chunk


def expand_source_filename(filename):
    if filename.endswith(".list"):
        with open(filename) as list_file:
            yield from expand_filenames((line.strip() for line in list_file))
    else:
        yield from glob(filename)

def expand_filenames(filenames):
    for filename in filenames:
        yield from expand_source_filename(filename)

def parse_filenames(filenames):
    if isinstance(filenames, str):
        filenames = [filenames]
    return sorted(expand_filenames(filenames))
 

def read_tree(filenames, treename, rank=0, nranks=1, chunk_size=1000000, branches=None, variables=None, cut=None, verbose=True):
    assert 0 <= rank < nranks
    filenames = parse_filenames(filenames)
    if verbose:
        total_events = 0
        for filename in filenames[rank::nranks]:
            with uproot.open(filename) as root_file:
                total_events += root_file[treename].num_entries
        total_event_counter.add(total_events)
    if len(filenames) >= nranks:
        for filename in filenames[rank::nranks]:
            yield from read_tree_from_file(filename, treename, chunk_size=chunk_size, branches=branches, variables=variables, cut=cut, verbose=verbose)
    elif len(filenames) >= nranks / 2:
        if rank < len(filenames):
            yield from read_tree_from_file(filenames[rank], treename, chunk_size=chunk_size, branches=branches, variables=variables, cut=cut, verbose=verbose)
    else:
        ranks_per_file = nranks // len(filenames)
        file_index = rank // ranks_per_file
        if file_index >= len(filenames):
            return
        rank_index = rank % ranks_per_file
        filename = filenames[file_index]
        with uproot.open(f"{filename}:{treename}") as tree:
            num_entries = tree.num_entries
            start = int((num_entries / ranks_per_file) * rank_index)
            stop = int((num_entries / ranks_per_file) * (rank_index + 1))
        yield from read_tree_from_file(filename, treename, chunk_size=chunk_size, start=start, stop=stop, branches=branches, variables=variables, cut=cut, verbose=verbose)


def count_total_events(filenames, treename):
    for filename in filenames:
        with uproot.open(filename) as root_file:
            total_event_counter.add(root_file[treename].num_entries)
    return total_event_counter.value


if __name__ == "__main__":
    for chunk in read_tree("AntiMatterTree.root", "AntiMatterTree", branches=["TrkRigidityInner", "TrkCharge", "TofCharge"], variables={"TrkRigidityAsymmetryInnerAll": "TrkRigidityInner-TrkRigidityAll"}, cut="(TrkHasAll)&(abs(TrkRigidityInner)>10)&(TrkCharge>1.5)"):
        print(chunk.fields)
        print(chunk)
