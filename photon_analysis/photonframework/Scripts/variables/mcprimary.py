
import numpy as np
import awkward as ak

from tools.variables import returns, depends, annotate_binning as binning, make_theta, make_phi, make_dxydz, make_theta_xy


def make_mc_primary_direction(axis):
    @returns(np.float32)
    @depends(["McPrimaryDirXYZ"])
    @binning("direction")
    def _mc_primary_direction(events):
        return events.McPrimaryDirXYZ[:,axis]
    return _mc_primary_direction


def load_variables(config, workdir, energy_estimator, binnings):
    yield ("McPrimaryDirX", make_mc_primary_direction(0))
    yield ("McPrimaryDirY", make_mc_primary_direction(1))
    yield ("McPrimaryDirZ", make_mc_primary_direction(2))
    yield ("McPrimaryDirTheta", make_theta("McPrimaryDirZ"))
    yield ("McPrimaryDirPhi", make_phi("McPrimaryDirX", "McPrimaryDirY"))
    yield ("McPrimaryDirThetaX", make_theta_xy("McPrimaryDirX", "McPrimaryDirZ"))
    yield ("McPrimaryDirThetaY", make_theta_xy("McPrimaryDirY", "McPrimaryDirZ"))
