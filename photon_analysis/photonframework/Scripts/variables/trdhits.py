
import numpy as np
import awkward as ak

from tools.binnings import make_int_binning, make_lin_binning, make_log_binning
from tools.variables import depends, returns, annotate_binning as binning



def trd_hit_coordinate(axis):
    @returns(np.float32)
    @depends(["TrdHitCoordinates"])
    @binning("coordinate")
    def _trd_hit_coordinate(events):
        return events.TrdHitCoordinates[:,axis::3]
    return _trd_hit_coordinate


def load_variables(config, workdir, energy_estimator, binnings):

    yield ("TrdHitCoordinatesX", trd_hit_coordinate(0))
    yield ("TrdHitCoordinatesY", trd_hit_coordinate(1))
    yield ("TrdHitCoordinatesZ", trd_hit_coordinate(2))
