
import numpy as np
import awkward as ak

from tools.binnings import make_int_binning, make_lin_binning, make_log_binning
from tools.variables import depends, returns, annotate_binning as binning


@returns(np.float32)
@depends(["AccClustersNumberOfPairs"])
@binning(make_int_binning(8))
def number_of_twosided_acc_clusters(events):
    return np.sum(events.AccClustersNumberOfPairs > 0, axis=1)


def load_variables(config, workdir, energy_estimator, binnings):
    yield ("AccNumberOfTwosidedHits", number_of_twosided_acc_clusters)

