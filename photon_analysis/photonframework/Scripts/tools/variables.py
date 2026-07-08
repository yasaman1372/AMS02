
import importlib
import json
import os
import re

import numpy as np
import awkward as ak
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize, newton
from scipy.special import logit
import healpy

from tools.binnings import Binning, make_lin_binning, make_log_binning, make_int_binning, make_healpy_binning
from tools.constants import ACCEPTANCE_CATEGORIES, MC_PARTICLE_CHARGE_ARRAY, MC_PARTICLE_LABELS, MC_PARTICLE_MASS_ARRAY, MC_PROCESS_IDS, MC_PARTICLE_IDS, DEFAULT_RIGIDITY, TRK_LAYER_POSITION_Z, TRK_LAYER_RADIUS, TRK_LAYER_SIZE_X, TRK_LAYER_SIZE_Y, NAF_INDEX, AGL_INDEX, NAF_SIZE, AGL_SIZE, NAF_TILE_SIZE, AGL_TILE_SIZE, RICH_RADIATOR_X, RICH_RADIATOR_Y, RICH_RADIATOR_Z, RICH_RESOLUTION_NAF, RICH_RESOLUTION_AGL, TRK_N_SENSORS, TRK_SENSOR_SIZE_X, TRK_SENSOR_SIZE_Y, TRD_Z_LOWER, TRD_Z_UPPER, TRD_CONTOUR_BOTTOM
from tools.coordinates import convert_ams_direction_to_galactic
from tools.conversions import calc_mass, calc_rig, calc_beta
from tools.statistics import bethe_bloch, bethe_bloch_pm
from tools.utilities import resolve_derived_branches, rigidity_from_beta_resolution
from tools.trackerfeet import read_tracker_feet_file


AC_INNER = ACCEPTANCE_CATEGORIES["inner"]
AC_L1 = ACCEPTANCE_CATEGORIES["l1"]
AC_L9 = ACCEPTANCE_CATEGORIES["l9"]
AC_TRD = ACCEPTANCE_CATEGORIES["trd"]
AC_RICH = ACCEPTANCE_CATEGORIES["rich"]
AC_ECAL = ACCEPTANCE_CATEGORIES["ecal"]

def returns(dtype):
    def _annotate_dtype(func):
        func._dtype = dtype
        return func
    return _annotate_dtype

def depends(variables):
    def _annotate_dependencies(func):
        func._dependencies = variables
        return func
    return _annotate_dependencies

def annotate_binning(binning):
    def _annotate_binning(func):
        func._binning = binning
        return func
    return _annotate_binning


def make_galactic_latitude_and_longitude(direction_branch_x, direction_branch_y, direction_branch_z, time_branch):
    dependencies = ["ISSParameters", time_branch]
    if isinstance(direction_branch_x, str):
        assert isinstance(direction_branch_y, str) and isinstance(direction_branch_z, str)
        dependencies.extend([direction_branch_x, direction_branch_y, direction_branch_z])

    @returns(np.float32)
    @depends(dependencies)
    @annotate_binning(None)
    def _galactic_latitude_and_longitude(events):
        time = ak.to_numpy(events[time_branch])
        iss_parameters = events.ISSParameters
        iss_x = ak.to_numpy(iss_parameters[:,0])
        iss_y = ak.to_numpy(iss_parameters[:,1])
        iss_z = ak.to_numpy(iss_parameters[:,2])
        iss_vx = ak.to_numpy(iss_parameters[:,3])
        iss_vy = ak.to_numpy(iss_parameters[:,4])
        iss_vz = ak.to_numpy(iss_parameters[:,5])
        iss_yaw = ak.to_numpy(iss_parameters[:,6])
        iss_pitch = ak.to_numpy(iss_parameters[:,7])
        iss_roll = ak.to_numpy(iss_parameters[:,8])
        if isinstance(direction_branch_x, str):
            dir_x = events[direction_branch_x]
            dir_y = events[direction_branch_y]
            dir_z = events[direction_branch_z]
        else:
            dir_x = np.full(len(events), direction_branch_x)
            dir_y = np.full(len(events), direction_branch_y)
            dir_z = np.full(len(events), direction_branch_z)
        lat, lon = convert_ams_direction_to_galactic(dir_x, dir_y, dir_z, iss_x, iss_y, iss_z, iss_vx, iss_vy, iss_vz, iss_yaw, iss_pitch, iss_roll, time)
        return np.stack((lat, lon), axis=1)
    return _galactic_latitude_and_longitude


def abs_difference_to_true_value(true_branch, measured_branch):
    @returns(np.float32)
    @depends([true_branch, measured_branch])
    def _abs_difference_to_true_value(chunk):
        return np.abs(chunk[measured_branch] - chunk[true_branch])
    return _abs_difference_to_true_value

def difference_to_true_value(true_branch, measured_branch):
    @returns(np.float32)
    @depends([true_branch, measured_branch])
    def _difference_to_true_value(chunk):
        return chunk[measured_branch] - chunk[true_branch]
    return _difference_to_true_value

def relative_difference_to_true_value(true_branch, measured_branch):
    @returns(np.float32)
    @depends([true_branch, measured_branch])
    def _relative_difference_to_true_value(chunk):
        return (chunk[measured_branch] - chunk[true_branch]) / chunk[true_branch]
    return _relative_difference_to_true_value



def make_logit_bdt(bdt_variable):
    @returns(np.float32)
    @depends(bdt_variable)
    @annotate_binning("logit")
    def _logit_bdt(chunk):
        return logit((chunk[bdt_variable] + 1) / 2)
    return _logit_bdt

def make_difference(var1, var2, expected_scale=1):
    @returns(np.float32)
    @depends([var1, var2])
    @annotate_binning(make_lin_binning(-expected_scale, expected_scale, 100))
    def _difference(chunk):
        return chunk[var1] - chunk[var2]
    return _difference

def make_relative_difference(var1, var2):
    @returns(np.float32)
    @depends([var1, var2])
    @annotate_binning(make_lin_binning(-1, 1, 100))
    def _relative_difference(chunk):
        return (chunk[var1] - chunk[var2]) / chunk[var2]
    return _relative_difference

def make_ratio(var1, var2):
    @returns(np.float32)
    @depends([var1, var2])
    @annotate_binning(make_lin_binning(0, 2, 100))
    def _ratio(chunk):
        return chunk[var1] / chunk[var2]
    return _ratio

def make_sum(var1, var2, dtype, binning_annotation=None):
    @returns(dtype)
    @depends([var1, var2])
    @annotate_binning(binning_annotation)
    def _sum(chunk):
        return chunk[var1] + chunk[var2]
    return _sum


def make_healpy_index(latitude_branch, longitude_branch, nside=128):
    @returns(np.int32)
    @depends([latitude_branch, longitude_branch])
    @annotate_binning(make_healpy_binning(nside))
    def _healpy_index(events):
        latitude = events[latitude_branch]
        longitude = events[longitude_branch]
        mask = (longitude != 1000) & (latitude != 1000) & np.isfinite(longitude) & np.isfinite(latitude)
        result = -np.ones(len(events), dtype=np.int32)
        result[mask] = healpy.ang2pix(nside=nside, theta=longitude[mask], phi=latitude[mask], lonlat=True)
        return result
    return _healpy_index

def make_theta(dir_z_branch):
    @returns(np.float32)
    @depends([dir_z_branch])
    @annotate_binning("theta")
    def _theta(events):
        return np.arccos(events[dir_z_branch])
    return _theta

def make_phi(dir_x_branch, dir_y_branch):
    @returns(np.float32)
    @depends([dir_x_branch, dir_y_branch])
    @annotate_binning("phi")
    def _phi(events):
        return np.arctan2(events[dir_y_branch], events[dir_x_branch])
    return _phi

def make_dxydz(dir_xy_branch, dir_z_branch):
    @returns(np.float32)
    @depends([dir_xy_branch, dir_z_branch])
    @annotate_binning("dxydz")
    def _dxydz(events):
        return events[dir_xy_branch] / events[dir_z_branch]
    return _dxydz

def make_theta_xy(dir_xy_branch, dir_z_branch):
    @returns(np.float32)
    @depends([dir_xy_branch, dir_z_branch])
    @annotate_binning("theta")
    def _theta_x(events):
        return np.arctan(events[dir_xy_branch] / events[dir_z_branch])
    return _theta_x

def make_angle_in_degrees(dir1_x_branch, dir1_y_branch, dir1_z_branch, dir2_x_branch, dir2_y_branch, dir2_z_branch):
    @returns(np.float32)
    @depends([dir1_x_branch, dir1_y_branch, dir1_z_branch, dir2_x_branch, dir2_y_branch, dir2_z_branch])
    @annotate_binning(make_lin_binning(0, 180, 180))
    def _angle(events):
        x1 = events[dir1_x_branch]
        y1 = events[dir1_y_branch]
        z1 = events[dir1_z_branch]
        x2 = events[dir2_x_branch]
        y2 = events[dir2_y_branch]
        z2 = events[dir2_z_branch]
        norm1 = (x1**2 + y1**2 + z1**2)**0.5
        norm2 = (x2**2 + y2**2 + z2**2)**0.5
        return np.rad2deg(np.arccos((x1 * x2 + y1 * y2 + z1 * z2) / (norm1 * norm2)))
    return _angle


def extrapolate_to_z(coordinate_branch, direction_branch, z_coordinate_branch, target_z):
    @returns(np.float32)
    @depends([coordinate_branch, direction_branch, z_coordinate_branch])
    @annotate_binning("coordinate")
    def _extrapolate_to_z(events):
        return events[coordinate_branch] + events[direction_branch] * (target_z - events[z_coordinate_branch])
    return _extrapolate_to_z

def make_radius(coord_x, coord_y):
    @returns(np.float32)
    @depends([coord_x, coord_y])
    @annotate_binning("radius")
    def _radius(events):
        return np.sqrt(events[coord_x]**2 + events[coord_y]**2)
    return _radius


def extract_float_axis(branch, axis, binning):
    @returns(np.float32)
    @depends([branch])
    @annotate_binning(binning)
    def _float_axis(events):
        return events[branch][:,axis]
    return _float_axis



class DerivedVariables:
    def __init__(self, config=None, workdir=None, energy_estimator=None, binnings=None):
        self.functions = None
        self.dependencies = None
        self.energy_estimator = energy_estimator
        self.binnings = binnings
        if config is not None:
            self.initialize(config, workdir, energy_estimator)

    def register(self, variable, function, dependencies=None, dtype=None, binning=None):
        self.functions[variable] = function
        if dependencies is None:
            dependencies = function._dependencies
        self.dependencies[variable] = dependencies
        if dtype is None:
            dtype = function._dtype
        self.dtypes[variable] = dtype
        if binning is None:
            binning = function._binning
        if binning is not None:
            self.binnings.register_variable(variable, binning)

    def initialize(self, config, workdir, energy_estimator):
        self.functions = {}
        self.dependencies = {}
        self.dtypes = {}
        for variable_module_name in config["variables"]:
            variable_module = importlib.import_module(f"variables.{variable_module_name}")
            for args in variable_module.load_variables(config, workdir, energy_estimator, self.binnings):
                self.register(*args)

    def resolve_branches(self, all_branches):
        return resolve_derived_branches(all_branches, self.dependencies, self.functions)
