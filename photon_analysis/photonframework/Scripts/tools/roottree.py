#!/usr/bin/env python3

from contextlib import contextmanager
from glob import glob
import multiprocessing as mp
import os
import shutil

import numpy as np
import uproot
import awkward as ak

import packaging

from tools.lazy_arrays import LazyArray

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


PGID = os.getpgid(os.getpid())
CACHE_DIRS = [
    os.path.join(dir, "_".join((os.path.expandvars("${USER}"), str(PGID)))) for dir in
    (os.path.expandvars("${TMP}"), "/tmp")
    if os.path.isdir(dir)
]

@contextmanager
def copy_file(source_path, cache_dirs=CACHE_DIRS, suffix="tmp", safety_factor=2, n_attempts=3, cache_file=True):
    if not cache_file:
        yield source_path
        return
    try:
        file_size = os.stat(source_path).st_size
        for dest_dir in cache_dirs:
            os.makedirs(dest_dir, exist_ok=True)
            vfs_stat = os.statvfs(dest_dir)
            fs_free_size = vfs_stat.f_bsize * vfs_stat.f_bavail
            if fs_free_size < file_size * safety_factor:
                print(f"Not enough space in {dest_dir}: {fs_free_size} < {file_size * safety_factor}")
                continue
            filename, ext = os.path.splitext(os.path.basename(source_path))
            dest_path = os.path.join(dest_dir, f"{filename}_{suffix}{ext}")
            for attempt in range(n_attempts):
                shutil.copy(source_path, dest_path)
                if os.stat(dest_path).st_size == file_size:
                    break
            cache_file_size = os.stat(dest_path).st_size
            if cache_file_size != file_size:
                print(f"Failed to copy {source_path} to {dest_path} ({cache_file_size} != {file_size})")
                os.remove(dest_path)
                continue
            #print(f"Copied {source_path} -> {dest_path}")
            yield dest_path
            os.remove(dest_path)
            return
    except OSError:
        print(f"OS error while copying {source_path!r}")
    print(f"Using original {source_path}")
    yield source_path

def clear_cache(treename):
    for cache_dir in CACHE_DIRS:
        if os.path.isdir(cache_dir):
            for filename in os.listdir(cache_dir):
                if filename.startswith(treename):
                    os.remove(os.path.join(cache_dir, filename))


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
            events = LazyArray.create(chunk)
            if verbose:
                event_counter.add(len(events))
                current = event_counter.value
                total = total_event_counter.value
                percentage = (current / total * 100) if total > 0 else 0
                print(f"{current:>10} / {total:>10} ({percentage:>5.1f}%)", flush=True)
            yield events


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
    result = sorted(expand_filenames(filenames))
    if not result:
        raise FileNotFoundError(filenames)
    return result
 

def read_tree(filenames, treename, rank=0, nranks=1, chunk_size=100000, branches=None, variables=None, cut=None, cache_file=True, verbose=True):
    assert 0 <= rank < nranks
    filenames = parse_filenames(filenames)
    if len(filenames) >= nranks:
        if verbose:
            total_events = 0
            for filename in filenames[rank::nranks]:
                with uproot.open(filename) as root_file:
                    if treename not in root_file:
                        continue
                    total_events += root_file[treename].num_entries
            total_event_counter.add(total_events)
        for filename in filenames[rank::nranks]:
            with uproot.open(filename) as root_file:
                if treename not in root_file:
                    continue
            with copy_file(filename, suffix=str(rank), safety_factor=os.cpu_count() * 2, cache_file=cache_file) as temp_filename:
                yield from read_tree_from_file(temp_filename, treename, chunk_size=chunk_size, branches=branches, variables=variables, cut=cut, verbose=verbose)
    elif len(filenames) >= nranks / 2:
        if rank < len(filenames):
            if verbose:
                with uproot.open(filenames[rank]) as root_file:
                    total_event_counter.add(root_file[treename].num_entries)
            with copy_file(filenames[rank], suffix=str(rank), safety_factor=os.cpu_count() * 2, cache_file=cache_file) as temp_filename:
                yield from read_tree_from_file(temp_filename, treename, chunk_size=chunk_size, branches=branches, variables=variables, cut=cut, verbose=verbose)
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
            if verbose:
                total_event_counter.add(stop - start)
        with copy_file(filename, suffix=str(rank), safety_factor=os.cpu_count() * 2, cache_file=cache_file) as temp_filename:
            yield from read_tree_from_file(temp_filename, treename, chunk_size=chunk_size, start=start, stop=stop, branches=branches, variables=variables, cut=cut, verbose=verbose)


def count_total_events(filenames, treename):
    for filename in filenames:
        with uproot.open(filename) as root_file:
            total_event_counter.add(root_file[treename].num_entries)
    return total_event_counter.value
