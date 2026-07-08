
import numpy as np
import awkward as ak

from tools.binnings import make_log_binning

from tools.variables import depends, returns, annotate_binning as binning

@returns(np.float32)
@depends(["McWeight", "PrescalingWeight"])
@binning(make_log_binning(1, 10000, 10))
def calculate_total_weight(events):
    return events.McWeight * events.PrescalingWeight


def load_variables(config, workdir, energy_estimator, binnings):
    yield ("TotalWeight", calculate_total_weight)
