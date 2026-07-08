
from datetime import datetime

import numpy as np

from astropy import units
from astropy.coordinates import SkyCoord, TEME, ITRS, Galactic, CartesianRepresentation, SphericalRepresentation
from astropy.time import Time
import astropy.units as u

from astropy.coordinates.matrix_utilities import rotation_matrix


AMS_TO_BODY_MATRIX = rotation_matrix(-12.0001 * units.deg, axis="y")
BODY_TO_AMS_MATRIX = rotation_matrix(12.0001 * units.deg, axis="y")

def ams_to_body(dx, dy, dz):
    # rotate by AMS inclination
    c = CartesianRepresentation(dx, dy, dz)
    nc = c.transform(AMS_TO_BODY_MATRIX)
    # redefine axes
    return -nc.y.value, -nc.x.value, -nc.z.value

def body_to_ams(dx, dy, dz):
    c = CartesianRepresentation(-dy, -dx, -dz)
    nc = c.transform(BODY_TO_AMS_MATRIX)
    return nc.x.value, nc.y.value, nc.z.value


def body_to_lvlh(dx, dy, dz, iss_yaw, iss_pitch, iss_roll):
    c = CartesianRepresentation(dx, dy, dz)
    c = c.transform(rotation_matrix(-iss_roll * units.rad, axis="x"))
    c = c.transform(rotation_matrix(-iss_pitch * units.rad, axis="y"))
    c = c.transform(rotation_matrix(-iss_yaw * units.rad, axis="z"))
    return c.x.value, c.y.value, c.z.value

def lvlh_to_body(dx, dy, dz, iss_yaw, iss_pitch, iss_roll):
    c = CartesianRepresentation(dx, dy, dz)
    c = c.transform(rotation_matrix(iss_yaw * units.rad, axis="z"))
    c = c.transform(rotation_matrix(iss_pitch * units.rad, axis="y"))
    c = c.transform(rotation_matrix(iss_roll * units.rad, axis="x"))
    return c.x.value, c.y.value, c.z.value


def lvlh_to_teme(dx, dy, dz, iss_pos_x, iss_pos_y, iss_pos_z, iss_vel_x, iss_vel_y, iss_vel_z):
    pos = np.array([dx, dy, dz])
    iss_pos = np.array([iss_pos_x, iss_pos_y, iss_pos_z])
    iss_velocity = np.array([iss_vel_x, iss_vel_y, iss_vel_z])
    r = iss_pos / np.sum(iss_pos**2, axis=0)**0.5
    v = iss_velocity / np.sum(iss_velocity**2, axis=0)**0.5
    rx, ry, rz = r
    vx, vy, vz = v
    Ry = np.array([vy * rz - vz * ry, -vx * rz + vz * rx, vx * ry - vy * rx])
    Ry = Ry / np.sum(Ry**2, axis=0)**0.5
    R = np.array([
        [
            vx * (ry**2 + rz**2) - vy * rx * ry - vz * rx * rz,
            vy * (rx**2 + rz**2) - vx * rx * ry - vz * ry * rz,
            vz * (rx**2 + ry**2) - vx * rx * rz - vy * ry * rz,
        ],
        [
            Ry[0],
            Ry[1],
            Ry[2],
        ],
        [
            -rx,
            -ry,
            -rz,
        ]
    ])
    # move coordinate axis to front, transpose matrix
    R = np.moveaxis(np.moveaxis(R, 2, 0), 1, 2)
    pos = np.moveaxis(pos, 1, 0)[:,:,None]
    result = R @ pos
    result_x, result_y, result_z = np.squeeze(result, axis=2).T
    return result_x, result_y, result_z


def teme_to_lvlh(dx, dy, dz, iss_pos_x, iss_pos_y, iss_pos_z, iss_vel_x, iss_vel_y, iss_vel_z):
    pos = np.array([dx, dy, dz])
    iss_pos = np.array([iss_pos_x, iss_pos_y, iss_pos_z])
    iss_velocity = np.array([iss_vel_x, iss_vel_y, iss_vel_z])
    r = iss_pos / np.sum(iss_pos**2, axis=0)**0.5
    v = iss_velocity / np.sum(iss_velocity**2, axis=0)**0.5
    rx, ry, rz = r
    vx, vy, vz = v
    Ry = np.array([vy * rz - vz * ry, -vx * rz + vz * rx, vx * ry - vy * rx])
    Ry = Ry / np.sum(Ry**2, axis=0)**0.5
    R = np.array([
        [
            vx * (ry**2 + rz**2) - vy * rx * ry - vz * rx * rz,
            vy * (rx**2 + rz**2) - vx * rx * ry - vz * ry * rz,
            vz * (rx**2 + ry**2) - vx * rx * rz - vy * ry * rz,
        ],
        [
            Ry[0],
            Ry[1],
            Ry[2],
        ],
        [
            -rx,
            -ry,
            -rz,
        ]
    ])
    # move coordinate axis to front
    R = np.moveaxis(R, 2, 0)
    pos = np.moveaxis(pos, 1, 0)[:,:,None]
    result = R @ pos
    result_x, result_y, result_z = np.squeeze(result, axis=2).T
    return result_x, result_y, result_z


def teme_to_galactic(dx, dy, dz, timestamp):
    t = Time(timestamp, format="unix", scale="utc")
    coord = SkyCoord(dx, dy, dz, obstime=t, frame=TEME)
    gal = coord.transform_to(Galactic)
    return gal.b.value, gal.l.value

def galactic_to_teme(latitude, longitude, timestamp):
    t = Time(timestamp, format="unix", scale="utc")
    coord = SkyCoord(l=longitude, b=latitude, unit=units.deg, obstime=t, frame=Galactic)
    teme_coord = coord.transform_to(TEME)
    return teme_coord.x.value, teme_coord.y.value, teme_coord.z.value

def teme_to_itrs(dx, dy, dz, timestamp):
    t = Time(timestamp, format="unix", scale="utc")
    coord = SkyCoord(dx, dy, dz, obstime=t, frame=TEME)
    itrs = coord.transform_to(ITRS).represent_as(SphericalRepresentation)
    longitude = np.degrees(itrs.lon.value)
    latitude = np.degrees(itrs.lat.value)
    return longitude, latitude

def lvlh_to_itrs(dx, dy, dz, iss_pos_x, iss_pos_y, iss_pos_z, iss_velocity_x, iss_velocity_y, iss_velocity_z, time):
    return teme_to_itrs(*lvlh_to_teme(dx, dy, dz, iss_pos_x, iss_pos_y, iss_pos_z, iss_velocity_x, iss_velocity_y, iss_velocity_z), time)



def convert_ams_direction_to_galactic(dx, dy, dz, iss_x, iss_y, iss_z, iss_velocity_x, iss_velocity_y, iss_velocity_z, iss_yaw, iss_pitch, iss_roll, time):
    return teme_to_galactic(
        *lvlh_to_teme(
            *body_to_lvlh(
                *ams_to_body(dx, dy, dz),
                iss_yaw, iss_pitch, iss_roll),
            iss_x, iss_y, iss_z, iss_velocity_x, iss_velocity_y, iss_velocity_z),
        time)

def convert_galactic_direction_to_ams(latitude, longitude, iss_x, iss_y, iss_z, iss_velocity_x, iss_velocity_y, iss_velocity_z, iss_yaw, iss_pitch, iss_roll, time):
    teme = galactic_to_teme(latitude, longitude, time)
    lvlh = teme_to_lvlh(*teme, iss_x, iss_y, iss_z, iss_velocity_x, iss_velocity_y, iss_velocity_z)
    body = lvlh_to_body(*lvlh, iss_yaw, iss_pitch, iss_roll)
    ams = body_to_ams(*body)
    return ams


def convert_equatorial_to_galactic_coordinates(ra, dec):
    coord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
    galactic_coord = coord.galactic
    return galactic_coord.l.deg, galactic_coord.b.deg

def convert_galactic_to_equatorial_coordinates(latitude, longitude):
    coord = SkyCoord(l=longitude, b=latitude, unit=units.deg, frame="galactic")
    ra = coord.fk5.ra.value
    dec = coord.fk5.dec.value
    return coord.fk5.ra.value, coord.fk5.dec.value


if __name__ == "__main__":
    # four random example events from run 1437869828
    dx = np.array([-0.386768, -0.671738, -0.233405, 0.240717])
    dy = np.array([-0.463778, 0.00632774, -0.643377, -0.687376])
    dz = np.array([0.79707, 0.740762, 0.7291, 0.685251])
    time = np.array([1437869812, 1437869812, 1437869812, 1437869812])
    iss_yaw = np.array([-0.0707434, -0.0707434, -0.0707434, -0.0707434])
    iss_pitch = np.array([-0.0324442, -0.0324442, -0.0324442, -0.0324443])
    iss_roll = np.array([0.0148007, 0.0148007, 0.0148007, 0.0148005])
    iss_x = np.array([-2211.35, -2211.35, -2211.35, -2211.25])
    iss_y = np.array([-6407.54, -6407.54, -6407.54, -6407.57])
    iss_z = np.array([120.215, 120.215, 120.215, 120.335])
    iss_vx = np.array([4.53558, 4.53558, 4.53558, 4.53563])
    iss_vy = np.array([-1.44619, -1.44619, -1.44619, -1.44602])
    iss_vz = np.array([6.01593, 6.01593, 6.01593, 6.01592])

    longitude = np.array([46.9148, 17.3114, 60.8875, 80.8720])
    latitude = np.array([10.3666, -2.31388, 13.9448, 34.9455])

    print("Converting AMS directions to Galactic, expecting:")
    print(latitude, longitude)
    print("Got:")
    print(*convert_ams_direction_to_galactic(dx, dy, dz, iss_x, iss_y, iss_z, iss_vx, iss_vy, iss_vz, iss_yaw, iss_pitch, iss_roll, time))

    print("Converting Galatic directions to AMS, expecting:")
    print(dx, dy, dz)
    print("Got:")
    print(*convert_galactic_direction_to_ams(latitude, longitude, iss_x, iss_y, iss_z, iss_vx, iss_vy, iss_vz, iss_yaw, iss_pitch, iss_roll, time))

    # result should be close to
    #  46.9148,  10.3666
    #  17.3114, -2.31388
    #  60.8875,  13.9448
    #  80.8720,  34.9455

    print("---")
    print(lvlh_to_itrs(np.zeros(4), np.zeros(4), -np.ones(4), iss_x, iss_y, iss_z, iss_vx, iss_vy, iss_vz, time))

