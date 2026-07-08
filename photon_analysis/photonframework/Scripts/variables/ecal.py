
import numpy as np
import awkward as ak
from matplotlib.path import Path

from tools.binnings import make_int_binning, make_lin_binning, make_log_binning, make_bool_binning
from tools.constants import TRK_LAYER_POSITION_Z, TRD_Z_LOWER, TRD_Z_UPPER, TRD_CONTOUR_BOTTOM, TRD_CONTOUR_TOP
from tools.variables import depends, returns, annotate_binning as binning, make_radius, make_theta, make_phi, make_dxydz, make_theta_xy, extrapolate_to_z, extract_float_axis, make_galactic_latitude_and_longitude, make_angle_in_degrees



def make_highest_energy_shower_index(energy_branch):
    @returns(np.int16)
    @depends([energy_branch])
    @binning(make_int_binning(4))
    def _highest_energy_shower_index(events):
        return ak.argmax(events[energy_branch], axis=1)
    return _highest_energy_shower_index


def make_highest_energy_ecal_shower_position(index_branch, position_branch, n_showers_branch, axis):
    @returns(np.float32)
    @depends([position_branch, index_branch, n_showers_branch])
    @binning("coordinate")
    def _highest_energy_ecal_shower_position(events):
        mask = events[n_showers_branch] > 0
        array_indices = np.arange(len(events))[mask]
        index = ak.to_numpy(events[index_branch][mask], allow_missing=False)
        result = np.zeros(len(events))
        result[mask] = events[position_branch][array_indices,3 * index + axis]
        return result
    return _highest_energy_ecal_shower_position


def make_highest_energy_ecal_shower_direction(index_branch, direction_branch, n_showers_branch, axis, invert=False):
    sign = -1 if invert else 1

    @returns(np.float32)
    @depends([direction_branch, index_branch, n_showers_branch])
    @binning("coordinate")
    def _highest_energy_ecal_shower_direction(events):
        mask = events[n_showers_branch] > 0
        array_indices = np.arange(len(events))[mask]
        index = ak.to_numpy(events[index_branch][mask], allow_missing=False)
        result = np.zeros(len(events))
        result[mask] = sign * events[direction_branch][array_indices,3 * index + axis]
        return result
    return _highest_energy_ecal_shower_direction


def make_highest_energy_ecal_shower_property(index_branch, property_branch, n_showers_branch, binning_hint):
    @returns(np.float32)
    @depends([property_branch, index_branch, n_showers_branch])
    @binning(binning_hint)
    def _highest_energy_ecal_shower_property(events):
        mask = events[n_showers_branch] > 0
        array_indices = np.arange(len(events))[mask]
        index = ak.to_numpy(events[index_branch][mask], allow_missing=False)
        result = np.zeros(len(events))
        result[mask] = events[property_branch][array_indices, index]
        return result
    return _highest_energy_ecal_shower_property


def choose_ecal_2d_shower_direction(energy_estimator, axis_name):
    cog_dir_branch = f"EcalMaxEnergy2DShowerCoGDirection{axis_name}"
    cr_dir_branch = f"EcalMaxEnergy2DShowerCRDirection{axis_name}"

    @returns(np.float32)
    @depends([energy_estimator, cog_dir_branch, cr_dir_branch])
    @binning("direction")
    def _ecal_shower_direction(events):
        energy = events[energy_estimator]
        cog_dir = events[cog_dir_branch]
        cr_dir = events[cr_dir_branch]
        above_20 = energy >= 20
        return cog_dir * (1 - above_20) + cr_dir * above_20
    return _ecal_shower_direction

def make_cos_theta(theta_branch):
    @returns(np.float32)
    @depends([theta_branch])
    @binning(make_lin_binning(0, 1, 100))
    def _cos_theta(events):
        return np.cos(events[theta_branch])
    return _cos_theta

def event_pass_through_trd(trd_edge,extrapolated_shower_x, extrapolated_shower_y):
    @returns(bool)
    @depends([extrapolated_shower_x,extrapolated_shower_y])
    @binning(make_bool_binning())
    def _ecal_best_pair_min_distane_is_in_upper_lower_trd(events):
        result = np.zeros(len(events), dtype = bool)
        if trd_edge == "Lower":
            Conture = Path(TRD_CONTOUR_BOTTOM)
        elif trd_edge == 'Upper':
            Conture = Path(TRD_CONTOUR_TOP)

        x = events[extrapolated_shower_x]
        y = events[extrapolated_shower_y]
        points = np.array([x,y]).T
        result = Conture.contains_points(points)
        return result
    return _ecal_best_pair_min_distane_is_in_upper_lower_trd

def inside_fiducial_volume(showerpositionX, showerpositionY):
    @returns(bool)
    @depends([showerpositionX, showerpositionY])
    @binning(make_bool_binning())
    def _fiducial_volume_cut(events):
        result = np.ones(len(events), dtype=bool)  # Start with all events included
        
        x = events[showerpositionX]
        y = events[showerpositionY]
        
        # Define the boundaries of the fiducial volume
        x_min, x_max = -32, 32
        y_min, y_max = -32, 32
        corner_margin = 2

        # Identify points in the corner regions
        top_left = (x < x_min + corner_margin) & (y > y_max - corner_margin)
        top_right = (x > x_max - corner_margin) & (y > y_max - corner_margin)
        bottom_left = (x < x_min + corner_margin) & (y < y_min + corner_margin)
        bottom_right = (x > x_max - corner_margin) & (y < y_min + corner_margin)

        # Exclude events in the corners
        result[top_left | top_right | bottom_left | bottom_right] = False

        return result
    
    return _fiducial_volume_cut

def load_variables(config, workdir, energy_estimator, binnings):
    binnings.register_binning("chisquare", make_log_binning(0.01, 100, 100))
    binnings.register_binning("ratio", make_lin_binning(0, 1, 100))

    yield ("EcalMaxEnergyShowerIndex", make_highest_energy_shower_index("EcalShowerEnergies"))
    yield ("EcalMaxEnergy2DShowerIndex", make_highest_energy_shower_index("Ecal2DShowerEnergies"))

    yield ("EcalMaxEnergyShowerPositionX", make_highest_energy_ecal_shower_position("EcalMaxEnergyShowerIndex", "EcalShowerPositions", "NEcalShowers", 0))
    yield ("EcalMaxEnergyShowerPositionY", make_highest_energy_ecal_shower_position("EcalMaxEnergyShowerIndex", "EcalShowerPositions", "NEcalShowers", 1))
    yield ("EcalMaxEnergyShowerPositionZ", make_highest_energy_ecal_shower_position("EcalMaxEnergyShowerIndex", "EcalShowerPositions", "NEcalShowers", 2))

    yield ("EcalMaxEnergy2DShowerEntryPositionX", make_highest_energy_ecal_shower_position("EcalMaxEnergy2DShowerIndex", "Ecal2DShowerEntryPositions", "NEcal2DShowers", 0))
    yield ("EcalMaxEnergy2DShowerEntryPositionY", make_highest_energy_ecal_shower_position("EcalMaxEnergy2DShowerIndex", "Ecal2DShowerEntryPositions", "NEcal2DShowers", 1))
    yield ("EcalMaxEnergy2DShowerEntryPositionZ", make_highest_energy_ecal_shower_position("EcalMaxEnergy2DShowerIndex", "Ecal2DShowerEntryPositions", "NEcal2DShowers", 2))

    yield ("EcalMaxEnergyShowerDirectionX", make_highest_energy_ecal_shower_direction("EcalMaxEnergyShowerIndex", "EcalShowerDirectionsXYZ", "NEcalShowers", 0))
    yield ("EcalMaxEnergyShowerDirectionY", make_highest_energy_ecal_shower_direction("EcalMaxEnergyShowerIndex", "EcalShowerDirectionsXYZ", "NEcalShowers", 1))
    yield ("EcalMaxEnergyShowerDirectionZ", make_highest_energy_ecal_shower_direction("EcalMaxEnergyShowerIndex", "EcalShowerDirectionsXYZ", "NEcalShowers", 2))

    yield ("EcalMaxEnergy2DShowerCoGDirectionX", make_highest_energy_ecal_shower_direction("EcalMaxEnergy2DShowerIndex", "Ecal2DShowerDirectionsCoGXYZ", "NEcal2DShowers", 0, invert=True))
    yield ("EcalMaxEnergy2DShowerCoGDirectionY", make_highest_energy_ecal_shower_direction("EcalMaxEnergy2DShowerIndex", "Ecal2DShowerDirectionsCoGXYZ", "NEcal2DShowers", 1, invert=True))
    yield ("EcalMaxEnergy2DShowerCoGDirectionZ", make_highest_energy_ecal_shower_direction("EcalMaxEnergy2DShowerIndex", "Ecal2DShowerDirectionsCoGXYZ", "NEcal2DShowers", 2, invert=True))
    yield ("EcalMaxEnergy2DShowerCRDirectionX", make_highest_energy_ecal_shower_direction("EcalMaxEnergy2DShowerIndex", "Ecal2DShowerDirectionsCRXYZ", "NEcal2DShowers", 0, invert=True))
    yield ("EcalMaxEnergy2DShowerCRDirectionY", make_highest_energy_ecal_shower_direction("EcalMaxEnergy2DShowerIndex", "Ecal2DShowerDirectionsCRXYZ", "NEcal2DShowers", 1, invert=True))
    yield ("EcalMaxEnergy2DShowerCRDirectionZ", make_highest_energy_ecal_shower_direction("EcalMaxEnergy2DShowerIndex", "Ecal2DShowerDirectionsCRXYZ", "NEcal2DShowers", 2, invert=True))
    yield ("EcalMaxEnergy2DShowerEMDirectionX", make_highest_energy_ecal_shower_direction("EcalMaxEnergy2DShowerIndex", "Ecal2DShowerDirectionsEMXYZ", "NEcal2DShowers", 0, invert=True))
    yield ("EcalMaxEnergy2DShowerEMDirectionY", make_highest_energy_ecal_shower_direction("EcalMaxEnergy2DShowerIndex", "Ecal2DShowerDirectionsEMXYZ", "NEcal2DShowers", 1, invert=True))
    yield ("EcalMaxEnergy2DShowerEMDirectionZ", make_highest_energy_ecal_shower_direction("EcalMaxEnergy2DShowerIndex", "Ecal2DShowerDirectionsEMXYZ", "NEcal2DShowers", 2, invert=True))
    yield ("EcalMaxEnergy2DShowerDirectionX", choose_ecal_2d_shower_direction(energy_estimator, "X"))
    yield ("EcalMaxEnergy2DShowerDirectionY", choose_ecal_2d_shower_direction(energy_estimator, "Y"))
    yield ("EcalMaxEnergy2DShowerDirectionZ", choose_ecal_2d_shower_direction(energy_estimator, "Z"))

    yield ("EcalMaxEnergyShowerDirectionTheta", make_theta("EcalMaxEnergyShowerDirectionZ"))
    yield ("EcalMaxEnergyShowerCoGDirectionTheta", make_theta("EcalMaxEnergy2DShowerCoGDirectionZ"))
    yield ("EcalMaxEnergyShowerDirectionPhi", make_phi("EcalMaxEnergyShowerDirectionX", "EcalMaxEnergyShowerDirectionY"))
    yield ("EcalMaxEnergyShowerDirectionThetaX", make_theta_xy("EcalMaxEnergyShowerDirectionX", "EcalMaxEnergyShowerDirectionZ"))
    yield ("EcalMaxEnergyShowerDirectionThetaY", make_theta_xy("EcalMaxEnergyShowerDirectionY", "EcalMaxEnergyShowerDirectionZ"))
    yield ("EcalMaxEnergyShowerDirectionXZ", make_dxydz("EcalMaxEnergyShowerDirectionX", "EcalMaxEnergyShowerDirectionZ"))
    yield ("EcalMaxEnergyShowerDirectionYZ", make_dxydz("EcalMaxEnergyShowerDirectionY", "EcalMaxEnergyShowerDirectionZ"))
    yield ("EcalMaxEnergyShowerCoGDirectionXZ", make_dxydz("EcalMaxEnergy2DShowerCoGDirectionX", "EcalMaxEnergy2DShowerCoGDirectionZ"))
    yield ("EcalMaxEnergyShowerCoGDirectionYZ", make_dxydz("EcalMaxEnergy2DShowerCoGDirectionY", "EcalMaxEnergy2DShowerCoGDirectionZ"))
    yield ("EcalMaxEnergyShowerDirectionCosTheta", make_cos_theta("EcalMaxEnergyShowerDirectionTheta"))
    yield ("EcalMaxEnergyShowerCoGDirectionCosTheta", make_cos_theta("EcalMaxEnergyShowerCoGDirectionTheta"))


    yield ("EcalMaxEnergyShowerDirectionsAngleInDegrees_KX_EM", make_angle_in_degrees("EcalMaxEnergyShowerDirectionX", "EcalMaxEnergyShowerDirectionY", "EcalMaxEnergyShowerDirectionZ", "EcalMaxEnergy2DShowerEMDirectionX", "EcalMaxEnergy2DShowerEMDirectionY", "EcalMaxEnergy2DShowerEMDirectionZ"))
    yield ("EcalMaxEnergyShowerDirectionsAngleInDegrees_EM_CoG", make_angle_in_degrees("EcalMaxEnergy2DShowerEMDirectionX", "EcalMaxEnergy2DShowerEMDirectionY", "EcalMaxEnergy2DShowerEMDirectionZ","EcalMaxEnergy2DShowerCoGDirectionX","EcalMaxEnergy2DShowerCoGDirectionY","EcalMaxEnergy2DShowerCoGDirectionZ"))
    yield ("EcalMaxEnergyShowerDirectionsAngleInDegrees_CR_CoG", make_angle_in_degrees("EcalMaxEnergy2DShowerCRDirectionX", "EcalMaxEnergy2DShowerCRDirectionY", "EcalMaxEnergy2DShowerCRDirectionZ","EcalMaxEnergy2DShowerCoGDirectionX","EcalMaxEnergy2DShowerCoGDirectionY","EcalMaxEnergy2DShowerCoGDirectionZ"))

    yield ("EcalMaxEnergyShowerDirectionsAngleInDegrees", make_angle_in_degrees("EcalMaxEnergyShowerDirectionX", "EcalMaxEnergyShowerDirectionY", "EcalMaxEnergyShowerDirectionZ", "EcalMaxEnergy2DShowerEMDirectionX", "EcalMaxEnergy2DShowerEMDirectionY", "EcalMaxEnergy2DShowerEMDirectionZ"))
    yield ("EcalMaxEnergyShowerDirectionsAngleInDegrees_KX_EM", make_angle_in_degrees("EcalMaxEnergyShowerDirectionX", "EcalMaxEnergyShowerDirectionY", "EcalMaxEnergyShowerDirectionZ", "EcalMaxEnergy2DShowerEMDirectionX", "EcalMaxEnergy2DShowerEMDirectionY", "EcalMaxEnergy2DShowerEMDirectionZ"))     
    yield ("EcalMaxEnergyShowerDirectionsAngleInDegrees_EM_CoG", make_angle_in_degrees("EcalMaxEnergy2DShowerEMDirectionX", "EcalMaxEnergy2DShowerEMDirectionY", "EcalMaxEnergy2DShowerEMDirectionZ","EcalMaxEnergy2DShowerCoGDirectionX","EcalMaxEnergy2DShowerCoGDirectionY","EcalMaxEnergy2DShowerCoGDirectionZ"))     
    yield ("EcalMaxEnergyShowerDirectionsAngleInDegrees_CR_CoG", make_angle_in_degrees("EcalMaxEnergy2DShowerCRDirectionX", "EcalMaxEnergy2DShowerCRDirectionY", "EcalMaxEnergy2DShowerCRDirectionZ","EcalMaxEnergy2DShowerCoGDirectionX","EcalMaxEnergy2DShowerCoGDirectionY","EcalMaxEnergy2DShowerCoGDirectionZ")) 
    
    yield ("EcalMaxEnergyShowerExtrapolatedXAtL1", extrapolate_to_z("EcalMaxEnergyShowerPositionX", "EcalMaxEnergyShowerDirectionXZ", "EcalMaxEnergyShowerPositionZ", TRK_LAYER_POSITION_Z[0]))
    yield ("EcalMaxEnergyShowerExtrapolatedYAtL1", extrapolate_to_z("EcalMaxEnergyShowerPositionY", "EcalMaxEnergyShowerDirectionYZ", "EcalMaxEnergyShowerPositionZ", TRK_LAYER_POSITION_Z[0]))
    yield ("EcalMaxEnergyShowerExtrapolatedRAtL1", make_radius("EcalMaxEnergyShowerExtrapolatedXAtL1", "EcalMaxEnergyShowerExtrapolatedYAtL1"))
    yield ("EcalMaxEnergyShowerExtrapolatedXAtLowerTRD", extrapolate_to_z("EcalMaxEnergyShowerPositionX", "EcalMaxEnergyShowerDirectionXZ", "EcalMaxEnergyShowerPositionZ", TRD_Z_LOWER))
    yield ("EcalMaxEnergyShowerExtrapolatedYAtLowerTRD", extrapolate_to_z("EcalMaxEnergyShowerPositionY", "EcalMaxEnergyShowerDirectionYZ", "EcalMaxEnergyShowerPositionZ", TRD_Z_LOWER))
    yield ("EcalMaxEnergyShowerExtrapolatedRAtLowerTRD", make_radius("EcalMaxEnergyShowerExtrapolatedXAtLowerTRD", "EcalMaxEnergyShowerExtrapolatedYAtLowerTRD"))
    yield ("EcalMaxEnergyShowerExtrapolatedXAtUpperTRD", extrapolate_to_z("EcalMaxEnergyShowerPositionX", "EcalMaxEnergyShowerDirectionXZ", "EcalMaxEnergyShowerPositionZ", TRD_Z_UPPER))
    yield ("EcalMaxEnergyShowerExtrapolatedYAtUpperTRD", extrapolate_to_z("EcalMaxEnergyShowerPositionY", "EcalMaxEnergyShowerDirectionYZ", "EcalMaxEnergyShowerPositionZ", TRD_Z_UPPER))
    yield ("EcalMaxEnergyShowerExtrapolatedRAtUpperTRD", make_radius("EcalMaxEnergyShowerExtrapolatedXAtUpperTRD", "EcalMaxEnergyShowerExtrapolatedYAtUpperTRD"))
    
    yield ("EcalMaxEnergyShowerCoGExtrapolatedXAtLowerTRD", extrapolate_to_z("EcalMaxEnergy2DShowerEntryPositionX", "EcalMaxEnergyShowerCoGDirectionXZ", "EcalMaxEnergy2DShowerEntryPositionZ", TRD_Z_LOWER))
    yield ("EcalMaxEnergyShowerCoGExtrapolatedYAtLowerTRD", extrapolate_to_z("EcalMaxEnergy2DShowerEntryPositionY", "EcalMaxEnergyShowerCoGDirectionYZ", "EcalMaxEnergy2DShowerEntryPositionZ", TRD_Z_LOWER))
    yield ("EcalMaxEnergyShowerCoGExtrapolatedXAtUpperTRD", extrapolate_to_z("EcalMaxEnergy2DShowerEntryPositionX", "EcalMaxEnergyShowerCoGDirectionXZ", "EcalMaxEnergy2DShowerEntryPositionZ", TRD_Z_UPPER))
    yield ("EcalMaxEnergyShowerCoGExtrapolatedYAtUpperTRD", extrapolate_to_z("EcalMaxEnergy2DShowerEntryPositionY", "EcalMaxEnergyShowerCoGDirectionYZ", "EcalMaxEnergy2DShowerEntryPositionZ", TRD_Z_UPPER))
    

    
    yield ("EcalMaxEnergyShowerDirectionInGalacticCoordinates", make_galactic_latitude_and_longitude("EcalMaxEnergyShowerDirectionX", "EcalMaxEnergyShowerDirectionY", "EcalMaxEnergyShowerDirectionZ", "UTCTime"))
    yield ("EcalMaxEnergyShowerDirectionGalacticLatitude", extract_float_axis("EcalMaxEnergyShowerDirectionInGalacticCoordinates", 0, "coordinate"))
    yield ("EcalMaxEnergyShowerDirectionGalacticLongitude", extract_float_axis("EcalMaxEnergyShowerDirectionInGalacticCoordinates", 1, "coordinate"))

    yield ("EcalMaxEnergy2DShowerDepositedEnergy", make_highest_energy_ecal_shower_property("EcalMaxEnergy2DShowerIndex", "Ecal2DShowerDepositedEnergies", "NEcal2DShowers", "energy"))
    yield ("EcalMaxEnergy2DShowerLongitudinalChiSquare", make_highest_energy_ecal_shower_property("EcalMaxEnergy2DShowerIndex", "Ecal2DShowerLongitudinalChiSquares", "NEcal2DShowers", "chisquare"))
    yield ("EcalMaxEnergy2DShowerLateralChiSquare", make_highest_energy_ecal_shower_property("EcalMaxEnergy2DShowerIndex", "Ecal2DShowerLateralChiSquares", "NEcal2DShowers", "chisquare"))
    yield ("EcalMaxEnergy2DShowerRatio1cm", make_highest_energy_ecal_shower_property("EcalMaxEnergy2DShowerIndex", "Ecal2DShowerRatio1cm", "NEcal2DShowers", "ratio"))
    yield ("EcalMaxEnergy2DShowerRatio3cm", make_highest_energy_ecal_shower_property("EcalMaxEnergy2DShowerIndex", "Ecal2DShowerRatio3cm", "NEcal2DShowers", "ratio"))
    yield ("EventPassThroughLowerTRD", event_pass_through_trd('Lower',"EcalMaxEnergyShowerExtrapolatedXAtLowerTRD","EcalMaxEnergyShowerExtrapolatedYAtLowerTRD"))
    yield ("EventPassThroughUpperTRD", event_pass_through_trd('Upper',"EcalMaxEnergyShowerExtrapolatedXAtUpperTRD","EcalMaxEnergyShowerExtrapolatedXAtUpperTRD"))
    yield ("EventPassThroughLowerTRD_CoG", event_pass_through_trd('Lower',"EcalMaxEnergyShowerCoGExtrapolatedXAtLowerTRD","EcalMaxEnergyShowerCoGExtrapolatedYAtLowerTRD"))
    yield ("EventPassThroughUpperTRD_CoG", event_pass_through_trd('Upper',"EcalMaxEnergyShowerCoGExtrapolatedXAtUpperTRD","EcalMaxEnergyShowerCoGExtrapolatedYAtUpperTRD"))
    yield ("EventInsideTheFiducailVolume", inside_fiducial_volume("EcalMaxEnergyShowerPositionX", "EcalMaxEnergyShowerPositionY"))
