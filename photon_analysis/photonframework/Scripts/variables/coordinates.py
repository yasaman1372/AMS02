
import numpy as np
import awkward as ak
from matplotlib.path import Path
import healpy as hp

from tools.variables import returns, depends, annotate_binning as binning, make_healpy_index, make_galactic_latitude_and_longitude, extract_float_axis
from tools.coordinates import lvlh_to_itrs
from tools.constants import SAA_OUTLINE


def calculate_longitude(x, y):
    return np.arctan2(y, x)

def calculate_latitude(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    return np.pi / 2 - np.arccos(z / r)


@returns(np.float32)
@depends(["ISSParameters"])
@binning("longitude")
def iss_longitude(events):
    iss_x = ak.to_numpy(events.ISSParameters[:,0])
    iss_y = ak.to_numpy(events.ISSParameters[:,1])
    return np.degrees(calculate_longitude(iss_x, iss_y))

@returns(np.float32)
@depends(["ISSParameters"])
@binning("latitude")
def iss_latitude(events):
    iss_x = ak.to_numpy(events.ISSParameters[:,0])
    iss_y = ak.to_numpy(events.ISSParameters[:,1])
    iss_z = ak.to_numpy(events.ISSParameters[:,2])
    return np.degrees(calculate_latitude(iss_x, iss_y, iss_z))


@returns(np.float32)
@depends(["ISSParameters", "UTCTime"])
@binning(None)
def iss_itrs_coordinates(events):
    up_x = np.zeros(len(events))
    up_y = np.zeros(len(events))
    up_z = -np.ones(len(events))
    time = ak.to_numpy(events.UTCTime)
    parameters = events.ISSParameters
    iss_x = ak.to_numpy(parameters[:,0])
    iss_y = ak.to_numpy(parameters[:,1])
    iss_z = ak.to_numpy(parameters[:,2])
    iss_vx = ak.to_numpy(parameters[:,3])
    iss_vy = ak.to_numpy(parameters[:,4])
    iss_vz = ak.to_numpy(parameters[:,5])
    itrs_longitude, itrs_latitude = lvlh_to_itrs(up_x, up_y, up_z, iss_x, iss_y, iss_z, iss_vx, iss_vy, iss_vz, time)
    return np.stack((itrs_longitude, itrs_latitude), axis=1)


def is_in_polygon(polygon, longitude_branch, latitude_branch, nside=256):
    path = Path(polygon, closed=True)
    ref_lon, ref_lat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    ref_lonlat = np.stack((ref_lon, ref_lat), axis=1)
    inclusion_map = path.contains_points(ref_lonlat)

    @returns(bool)
    @depends([longitude_branch, latitude_branch])
    @binning(bool)
    def _is_in_polygon(events):
        healpy_index = hp.ang2pix(nside=nside, theta=events[longitude_branch], phi=events[latitude_branch], lonlat=True)
        return inclusion_map[healpy_index]

    return _is_in_polygon




def load_variables(config, workdir, energy_estimator, binnings):
    yield ("ISSCelestialLongitude", iss_longitude)
    yield ("ISSCelestialLatitude", iss_latitude)
    yield ("ISSTerrestrialCoordinates", iss_itrs_coordinates)
    yield ("ISSTerrestrialLongitude", extract_float_axis("ISSTerrestrialCoordinates", 0, "longitude"))
    yield ("ISSTerrestrialLatitude", extract_float_axis("ISSTerrestrialCoordinates", 1, "latitude"))
    yield ("ISSCelestialHealpyIndex64", make_healpy_index("ISSCelestialLatitude", "ISSCelestialLongitude", nside=64))
    yield ("ISSCelestialHealpyIndex128", make_healpy_index("ISSCelestialLatitude", "ISSCelestialLongitude", nside=128))
    yield ("ISSCelestialHealpyIndex256", make_healpy_index("ISSCelestialLatitude", "ISSCelestialLongitude", nside=256))
    yield ("ISSTerrestrialHealpyIndex64", make_healpy_index("ISSTerrestrialLatitude", "ISSTerrestrialLongitude", nside=64))
    yield ("ISSTerrestrialHealpyIndex128", make_healpy_index("ISSTerrestrialLatitude", "ISSTerrestrialLongitude", nside=128))
    yield ("ISSTerrestrialHealpyIndex256", make_healpy_index("ISSTerrestrialLatitude", "ISSTerrestrialLongitude", nside=256))
    yield ("AMSGalacticDirection", make_galactic_latitude_and_longitude(0, 0, 1, "UTCTime"))
    yield ("AMSGalacticLatitude", extract_float_axis("AMSGalacticDirection", 0, "latitude"))
    yield ("AMSGalacticLongitude", extract_float_axis("AMSGalacticDirection", 1, "longitude"))
    yield ("IsInSAA", is_in_polygon(SAA_OUTLINE, "ISSTerretrialLongitude", "ISSTerrestrialLatitude"))
