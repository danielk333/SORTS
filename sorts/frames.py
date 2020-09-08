#!/usr/bin/env python

'''Coordinate frame transformations and related functions

'''

#Python standard import
import pkg_resources

#Third party import
import numpy as np
import scipy.optimize
import pyorb
import pyant
import astropy.coordinates as coord
import astropy.units as units

from pyant.coordinates import cart_to_sph, sph_to_cart, vector_angle
from pyant.coordinates import rot_mat_x, rot_mat_y, rot_mat_z

#Local import
from . import dates
from . import constants


def _convert_to_astropy(state, frame, **kw):
    state_p = coord.CartesianRepresentation(state[:3,...]*units.m)
    state_v = coord.CartesianDifferential(state[3:,...]*units.m/units.s)
    astropy_state = frame(state_p.with_differentials(state_v), **kw)
    return astropy_state




def enu_to_ecef(lat, lon, alt, enu, radians=False):
    '''ENU (east/north/up) to ECEF coordinate system conversion, not including translation. 

    TODO: Docstring
    '''
    if not radians:
        lat, lon = np.radians(lat), np.radians(lon)

    mx = np.array([[-np.sin(lon), -np.sin(lat) * np.cos(lon), np.cos(lat) * np.cos(lon)],
                [np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat) * np.sin(lon)],
                [0, np.cos(lat), np.sin(lat)]])
    
    ecef = np.dot(mx,enu)
    return ecef 


def ned_to_ecef(lat, lon, alt, ned, radians=False):
    '''NED (north/east/down) to ECEF coordinate system conversion, not including translation.

    TODO: Docstring
    '''
    enu = np.empty(ned.size, dtype=ned.dtype)
    enu[0,...] = ned[1,...]
    enu[1,...] = ned[0,...]
    enu[2,...] = -ned[2,...]
    return enu_to_ecef(lat, lon, alt, enu, radians=radians)


def ecef_to_enu(lat, lon, alt, ecef, radians=False):
    '''ECEF coordinate system to local ENU (east,north,up), not including translation.

    TODO: Docstring
    '''
    if not radians:
        lat, lon = np.radians(lat), np.radians(lon)

    mx = np.array([[-np.sin(lon), -np.sin(lat) * np.cos(lon), np.cos(lat) * np.cos(lon)],
                [np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat) * np.sin(lon)],
                [0, np.cos(lat), np.sin(lat)]])
    enu = np.dot(np.linalg.inv(mx),ecef)
    return enu


def azel_to_ecef(lat, lon, alt, az, el, radians=False):
    '''Radar pointing (az,el) to unit vector in ECEF, not including translation.

    TODO: Docstring
    '''
    shape = (3,)
    if isinstance(az,np.ndarray):    
        if len(az) > 1:
            shape = (3,len(az))
            az = az.flatten()
        else:
            az = az[0]
    
    if isinstance(el,np.ndarray):
        if len(el) > 1:
            shape = (3,len(el))
            el = el.flatten()
        else:
            el = el[0]

    sph = np.empty(shape, dtype=np.float64)
    sph[0,...] = az
    sph[1,...] = el
    sph[2,...] = 1.0
    enu = sph_to_cart(sph, radians=radians)
    return enu_to_ecef(lat, lon, alt, enu, radians=radians)

