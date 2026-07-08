
import numpy as np
import awkward as ak

from tools.variables import depends, returns, annotate_binning as binning


def make_has_trigger_pattern(mask):
    @returns(bool)
    @depends(["TriggerFlags"])
    @binning(bool)
    def _has_trigger_pattern(events):
        return (events.TriggerFlags & mask) > 0
    return _has_trigger_pattern


def load_variables(config, workdir, energy_estimator, binnings):
    yield "HasTrigger", make_has_trigger_pattern(0x5f)
    yield "HasPhysicsTrigger", make_has_trigger_pattern(0x3e)
    yield "HasPhysicsTofTrigger", make_has_trigger_pattern(0x1e)
    yield "HasPhysicsEcalTrigger", make_has_trigger_pattern(0x20)
    yield "HasUnbiasedTrigger", make_has_trigger_pattern(0x41)
    yield "HasUnbiasedTofTrigger", make_has_trigger_pattern(0x01)
    yield "HasUnbiasedEcalTrigger", make_has_trigger_pattern(0x40)
