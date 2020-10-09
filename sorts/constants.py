#!/usr/bin/env python

'''Constants used by SORTS from various sources.

'''

#Third party import
import scipy.constants


class WGS84:
    '''World Geodetic System 1984 constants.
    '''

    a = 6378.137*1e3
    '''float: semi-major axis parameter in meters of the World Geodetic System 1984 (WGS84)
    '''

    b = 6356.7523142*1e3
    '''float: semi-minor axis parameter in meters of the World Geodetic System 1984 (WGS84)
    '''

    esq = 6.69437999014 * 0.001
    '''float: `esq` parameter in meters of the World Geodetic System 1984 (WGS84)
    '''

    e1sq = 6.73949674228 * 0.001
    '''float: `e1sq` parameter in meters of the World Geodetic System 1984 (WGS84)
    '''

    f = 1 / 298.257223563
    '''float: `f` parameter of the World Geodetic System 1984 (WGS84)
    '''


R_earth = 6371.0088e3
'''float: Radius of the Earth using the International Union of Geodesy and Geophysics (IUGG) definition
'''

class WGS72:
    '''World Geodetic System 1972 constants.
    '''

    MU_earth = 398600.8*1e9
    '''float: Standard gravitational parameter of the Earth using the WGS72 convention.
    '''

    M_earth = MU_earth/scipy.constants.G
    '''float: Mass of the Earth using the WGS72 convention.
    '''
