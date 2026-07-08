
import numpy as np
import awkward as ak

from tools.binnings import make_int_binning, make_lin_binning, make_log_binning
from tools.variables import returns, depends, annotate_binning as binning

def make_count(branch):
    @returns(np.int16)
    @depends([branch])
    @binning(make_int_binning(10))
    def _count(events):
        return ak.to_numpy(ak.num(events[branch])).astype(np.int16)
    return _count


def load_variables(config, workdir, energy_estimator, binnings):
    yield "NTrkTrackVertices", make_count("TrkVertexMomentum")
