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

from pyant.coordinates import cart_to_sph, sph_to_cart, vector_angle
from pyant.coordinates import rot_mat_x, rot_mat_y, rot_mat_z

#Local import
from . import dates
from . import propagator
from . import constants

_EOP_data = None
_EOP_header = None


def enu_to_ecef(lat, lon, alt, enu, radians=False):
    '''ENU (east/north/up) to ECEF coordinate system conversion. 

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
    '''NED (north/east/down) to ECEF coordinate system conversion.

    TODO: Docstring
    '''
    enu = np.empty(ned.size, dtype=ned.dtype)
    enu[0,...] = ned[1,...]
    enu[1,...] = ned[0,...]
    enu[2,...] = -ned[2,...]
    return enu_to_ecef(lat, lon, alt, enu, radians=radians)


def ecef_to_enu(lat, lon, alt, ecef, radians=False):
    '''ECEF coordinate system to local ENU (east,north,up).

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
    '''Radar pointing (az,el) to unit vector in ECEF.

    TODO: Docstring
    '''
    shape = (3,)
    if isinstance(az,np.ndarray):    
        if len(az.shape) > 1:
            shape = (3,az.shape[1])
    elif isinstance(el,np.ndarray):    
        if len(el.shape) > 1:
            shape = (3,el.shape[1])

    sph = np.empty(shape, dtype=np.float64)
    sph[0,...] = az
    sph[1,...] = el
    sph[2,...] = 1.0
    enu = sph_to_cart(sph, radians=radians)
    return enu_to_ecef(lat, lon, alt, enu, radians=radians)


def geodetic_to_ecef(lat, lon, alt, radians=False):
    '''Convert geodetic coordinates to ECEF using WGS84.
    
    TODO: Docstring
    '''
    if not radians:
        lat, lon = np.radians(lat), np.radians(lon)
    xi = np.sqrt(1 - constants.WGS84.esq * np.sin(lat)**2)
    x = (constants.WGS84.a / xi + alt) * np.cos(lat) * np.cos(lon)
    y = (constants.WGS84.a / xi + alt) * np.cos(lat) * np.sin(lon)
    z = (constants.WGS84.a / xi * (1 - constants.WGS84.esq) + alt) * np.sin(lat)
    return np.array([x, y, z])



def get_IERS_EOP(fname = None):
    '''Loads the IERS EOP data into memory. 

    This is automatically loaded, to load different dataset, call this function again with a different filename.

    Note: Column descriptions are hard-coded in the function and my change if standard IERS format is changed.

    :param str fname: path to input IERS data file.
    :return: tuple of (numpy.ndarray, list of column descriptions)

    '''
    global _EOP_data, _EOP_header

    if fname is None:
        stream = pkg_resources.resource_stream('sorts.data', 'eopc04_IAU2000.62-now')
        data = np.genfromtxt(stream, skip_header=14)
    else:    
        data = np.genfromtxt(fname, skip_header=14)

    header = [
        'Date year (0h UTC)',
        'Date month (0h UTC)',
        'Date day (0h UTC)',
        'MJD',
        'x (arcseconds)',
        'y (arcseconds)',
        'UT1-UTC (s)',
        'LOD (s)',
        'dX (arcseconds)',
        'dY (arcseconds)',
        'x Err (arcseconds)',
        'y Err (arcseconds)',
        'UT1-UTC Err (s)',
        'LOD Err (s)',
        'dX Err (arcseconds)',
        'dY Err (arcseconds)',
    ]
    _EOP_data, _EOP_header = data, header

try:
    #Run to load data
    get_IERS_EOP()
except:
    pass


def _get_jd_rows(jd_ut1):
    mjd = jd_ut1 - 2400000.5
    row = np.argwhere(np.abs(_EOP_data[:,3] - np.floor(mjd)) < 1e-1 )
    if len(row) == 0:
        raise Exception('No EOP data available for JD: {}'.format(jd_ut1))
    row = int(row)
    return _EOP_data[row:(row+2),:]


def _interp_rows(_jd_ut1, cols):
    if not isinstance(_jd_ut1, np.ndarray):
        if not isinstance(_jd_ut1, float):
            raise Exception('Only numpy.ndarray and float input allowed')

        _jd_ut1 = np.array([_jd_ut1], dtype=np.float)
    else:
        if max(_jd_ut1.shape) != _jd_ut1.size:
            raise Exception('Only 1D input arrays allowed')
        else:
            _jd_ut1 = _jd_ut1.copy().flatten()

    ret = np.empty((_jd_ut1.size, len(cols)), dtype=np.float)

    for jdi, jd_ut1 in enumerate(_jd_ut1):
        rows = _get_jd_rows(jd_ut1)
        frac = jd_ut1 - np.floor(jd_ut1)

        for ci, col in enumerate(cols):
            ret[jdi, ci] = rows[0,col]*(1.0 - frac) + rows[1,col]*frac

    return ret


def get_DUT(jd_ut1):
    '''Get the Difference UT between UT1 and UTC, :math:`DUT1 = UT1 - UTC`. This function interpolates between data given by IERS.
    
    :param float/numpy.ndarray jd_ut1: Input Julian date in UT1.
    :return: DUT
    :rtype: numpy.ndarray
    '''
    return _interp_rows(jd_ut1, [6])


def get_polar_motion(jd_ut1):
    '''Get the polar motion coefficients :math:`x_p` and :math:`y_p` used in EOP. This function interpolates between data given by IERS.
    
    :param float/numpy.ndarray jd_ut1: Input Julian date in UT1.
    :return: :math:`x_p` as column 0 and :math:`y_p` as column 1
    :rtype: numpy.ndarray
    '''
    return _interp_rows(jd_ut1, [4,5])*(np.pi/(60.0*60.0*180.0))




def TEME_to_ITRF(TEME, jd_ut1, xp, yp):
    '''Convert TEME position and velocity into standard ITRS coordinates.
    This converts a position and velocity vector in the idiosyncratic
    True Equator Mean Equinox (TEME) frame of reference used by the SGP4
    theory into vectors into the more standard ITRS frame of reference.

    *Reference:* AIAA 2006-6753 Appendix C.

    Original work Copyright (c) 2013-2018 Brandon Rhodes under the MIT license
    Modified work Copyright (c) 2019 Daniel Kastinen

    Since TEME uses the instantaneous North pole and mean direction
    of the Vernal equinox, a simple GMST and polar motion transformation will move to ITRS.

    # TODO: There is some ambiguity about if this is ITRS00 or something else? I dont know.

    :param numpy.ndarray TEME: 6-D state vector in TEME frame given in SI-units.
    :param float jd_ut1: UT1 Julian date.
    :param float xp: Polar motion constant for rotation around x axis
    :param float yp: Polar motion constant for rotation around y axis
    :return: ITRF 6-D state vector given in SI-units.
    :rtype: numpy.ndarray
    '''

    if len(TEME.shape) > 1:
        rTEME = TEME[:3, :]
        vTEME = TEME[3:, :]*3600.0*24.0
    else:
        rTEME = TEME[:3]
        vTEME = TEME[3:]*3600.0*24.0

    theta, theta_dot = dates.GMST1982(jd_ut1)
    zero = theta_dot * 0.0
    angular_velocity = np.array([zero, zero, -theta_dot])
    R = pyant.coordinates.rot_mat_z(-theta)

    if len(rTEME.shape) == 1:
        rPEF = (R).dot(rTEME)
        vPEF = (R).dot(vTEME) + np.cross(angular_velocity, rPEF)
    else:
        rPEF = np.einsum('ij...,j...->i...', R, rTEME)
        vPEF = np.einsum('ij...,j...->i...', R, vTEME) + np.cross(
            angular_velocity, rPEF, 0, 0).T

    if np.abs(xp) < 1e-10 and np.abs(yp) < 1e-10:
        rITRF = rPEF
        vITRF = vPEF
    else:
        W = (pyant.coordinates.rot_mat_x(yp)).dot(pyant.coordinates.rot_mat_y(xp))
        rITRF = (W).dot(rPEF)
        vITRF = (W).dot(vPEF)

    ITRF = np.empty(TEME.shape, dtype=TEME.dtype)
    if len(TEME.shape) > 1:
        ITRF[:3,:] = rITRF
        ITRF[3:,:] = vITRF/(3600.0*24.0)
    else:
        ITRF[:3] = rITRF
        ITRF[3:] = vITRF/(3600.0*24.0)

    return ITRF



def ITRF_to_TEME(ITRF, jd_ut1, xp, yp):
    '''Convert ITRF position and velocity into the idiosyncratic
    True Equator Mean Equinox (TEME) frame of reference used by the SGP4
    theory.

    Original work Copyright (c) 2013-2018 Brandon Rhodes under the MIT license
    Modified work Copyright (c) 2019 Daniel Kastinen

    # TODO: There is some ambiguity about if this is ITRS00 or something else? I dont know.

    :param numpy.ndarray ITRF: 6-D state vector in ITRF frame given in SI-units.
    :param float jd_ut1: UT1 Julian date.
    :param float xp: Polar motion constant for rotation around x axis
    :param float yp: Polar motion constant for rotation around y axis
    :return: ITRF 6-D state vector given in SI-units.
    :rtype: numpy.ndarray
    '''

    if len(ITRF.shape) > 1:
        rITRF = ITRF[:3, :]
        vITRF = ITRF[3:, :]*3600.0*24.0
    else:
        rITRF = ITRF[:3]
        vITRF = ITRF[3:]*3600.0*24.0

    theta, theta_dot = dates.GMST1982(jd_ut1)
    zero = theta_dot * 0.0
    angular_velocity = np.array([zero, zero, -theta_dot])
    R = pyant.coordinates.rot_mat_z(theta)

    if np.abs(xp) < 1e-10 and np.abs(yp) < 1e-10:
        rPEF = rITRF
        vPEF = vITRF
    else:
        W = (pyant.coordinates.rot_mat_y(-xp)).dot(pyant.coordinates.rot_mat_x(-yp))
        rPEF = (W).dot(rITRF)
        vPEF = (W).dot(vITRF)

    if len(rITRF.shape) == 1:
        rTEME = (R).dot(rPEF)
        vTEME = (R).dot(vPEF - np.cross(angular_velocity, rPEF))
    else:
        rTEME = np.einsum('ij...,j...->i...', R, rPEF)
        vTEME = np.einsum('ij...,j...->i...', R, vPEF - np.cross(angular_velocity, rPEF, 0, 0) )

    TEME = np.empty(ITRF.shape, dtype=ITRF.dtype)
    if len(TEME.shape) > 1:
        TEME[:3,:] = rTEME
        TEME[3:,:] = vTEME/(3600.0*24.0)
    else:
        TEME[:3] = rTEME
        TEME[3:] = vTEME/(3600.0*24.0)

    return TEME




def ECEF_to_TEME( t, p, v, mjd0=57084, xp=0.0, yp=0.0, model='80', lod=0.0015563):
    '''This function transforms a vector from ECEF to true equator mean equniox frame 

    The results take into account the effects of sidereal time, and polar motion.

    *References:* Vallado  2007, 219-228.

    Author: David Vallado 719-573-2600, 10 dec 2007.
    Adapted to Python, Daniel Kastinen 2018
    Reverse operation, developed by Daniel Kastinen 2019

    :param numpy.ndarray t: numpy vector row of seconds relative to :code:`mjd0`
    :param numpy.ndarray p: numpy matrix of TEME positions, Cartesian coordinates in km. Columns correspond to times in t, and rows to x, y and z coordinates respectively.
    :param numpy.ndarray v: numpy matrix of TEME velocities, Cartesian coordinates in km/s. Columns correspond to times in t, and rows to x, y and z coordinates respectively.
    :param float mjd0: Modified julian date epoch that t vector is relative to
    :param float xp: x-axis polar motion coefficient in radians
    :param float yp: y-axis polar motion coefficient in radians
    :param str model: The polar motion model used in transformation, options are '80' or '00', see David Vallado documentation for more info.
    :param float lod: Excess length of day in seconds

    :return: State vector of position and velocity in km and km/s.
    :rtype: numpy.ndarray (6-D vector)

    **Uses:**

       * :func:`sorts.dates.gstime`
       * :func:`sorts.frames.polarm`
    '''

    if not isinstance(t, np.ndarray):
        if isinstance(t, float):
            t = np.array([t], dtype=np.float64)
        else:
            raise TypeError('type(t) = {} not supported'.format(type(t)))

    if len(p.shape) == 1:
        p = p.reshape(3,1)
    if len(v.shape) == 1:
         v = v.reshape(3,1)


    if len(t.shape) == 1:
        try:
            t = t.reshape(1, t.size)
        except:
            print('Reshaping failed: t-{}, p-{}, v-{}'.format(
                t.shape, p.shape, v.shape,
            ))
            raise

    if p.shape[0] != 3 or v.shape[0] != 3:
        try:
            p = p.reshape(3, t.size)
            v = v.reshape(3, t.size)
        except:
            print('Reshaping failed: t-{}, p-{}, v-{}'.format(
                t.shape, p.shape, v.shape,
            ))
            raise



    if not isinstance(xp, np.ndarray) or not isinstance(yp, np.ndarray):
        if np.abs(xp) < 1e-10 and np.abs(yp) < 1e-10:
            model = ''
    else:
        if xp.size != t.size or yp.size != t.size:
            raise Exception("Polar motion lengths conflicting with time length, xp {}, yp {} ".format(xp.size, yp.size))
        if xp.shape != t.shape:
            xp = xp.reshape(1, t.size)
            yp = yp.reshape(1, t.size)

    #from MJD to J2000 relative JD to BC-relative JD to julian centuries, 2400000.5 - 2451545.0 = - 51544.5
    #a Julian year (symbol: a) is a unit of measurement of time defined as exactly 365.25 days of 86400 SI seconds each
    ttt = (mjd0 + t/86400.0 - 51544.5)/(365.25*100.0)

    #JDUT1 is actually in days from 4713 bc
    jdut1 = mjd0 + t/86400.0 + 2400000.5

    # ------------------------ find gmst --------------------------
    theta = dates.gstime( jdut1 );

    #find omega from nutation theory
    omega=  125.04452222  + ( -6962890.5390 *ttt + 7.455 *ttt*ttt + 0.008 *ttt*ttt*ttt ) / 3600.0;
    omega= np.radians(np.fmod( omega, 360.0  ));

    # ------------------------ find mean ast ----------------------
    # teme does not include the geometric terms here
    # after 1997, kinematic terms apply
    gmstg = np.where(
        jdut1 > 2450449.5,
        theta + 0.00264*np.pi /(3600*180)*np.sin(omega) + 0.000063*np.pi /(3600*180)*np.sin(2.0 *omega),
        theta,
    )
    gmstg = np.fmod( gmstg, 2.0*np.pi  )

    recef = np.empty(p.shape, dtype=p.dtype)
    vecef = np.empty(v.shape, dtype=v.dtype)

    if len(model) == 0:
        recef = p
        vecef = v
    elif model == '80':
        if isinstance(xp, np.ndarray):
            for tind in range(t.size):
                pm = polar_motion_matrix(xp[0,tind], yp[0,tind], 0.0, model)
                pm = np.linalg.inv(pm)
                recef[:,tind] = np.dot(pm, p[:,tind])
                vecef[:,tind] = np.dot(pm, v[:,tind])
        else:
            pm = polar_motion_matrix(xp, yp, 0.0, model)
            pm = np.linalg.inv(pm)
            recef = np.dot(pm, p)
            vecef = np.dot(pm, v)
    elif model == '00':
        if isinstance(xp, np.ndarray):
            for tind in range(ttt.size):
                pm = polar_motion_matrix(xp[0,tind], yp[0,tind], ttt[0,tind], model)
                pm = np.linalg.inv(pm)
                recef[:,tind] = np.dot(pm, p[:,tind])
                vecef[:,tind] = np.dot(pm, v[:,tind])
        else:
            for tind in range(ttt.size):
                pm = polar_motion_matrix(xp, yp, ttt[0,tind], model)
                pm = np.linalg.inv(pm)
                recef[:,tind] = np.dot(pm, rpef[:,tind])
                vecef[:,tind] = np.dot(pm, vpef[:,tind])


    thetasa    = 7.29211514670698e-05*(1.0  - lod/86400.0)
    omegaearth = np.array([0., 0., -thetasa])

    for ind in range(t.shape[1]):
        vecef[:,ind] -= np.cross(omegaearth, recef[:,ind])

    rpef=np.zeros(p.shape)
    vpef=np.zeros(v.shape)

    # direct calculation of rotation matrix with different rotation for each
    # column of p
    theta = -theta

    rpef[0,:]=np.multiply(np.cos(theta),recef[0,:])+np.multiply(np.sin(theta),recef[1,:])
    rpef[1,:]=-np.multiply(np.sin(theta),recef[0,:])+np.multiply(np.cos(theta),recef[1,:])
    rpef[2,:]=recef[2,:]

    vpef[0,:]=np.multiply(np.cos(theta),vecef[0,:])+np.multiply(np.sin(theta),vecef[1,:])
    vpef[1,:]=-np.multiply(np.sin(theta),vecef[0,:])+np.multiply(np.cos(theta),vecef[1,:])
    vpef[2,:]=vecef[2,:]


    teme = np.empty((6,t.size))
    teme[:3,:] = rpef
    teme[3:,:] = vpef

    return teme

def TEME_to_ECEF( t, p, v, mjd0=57084, xp=0.0, yp=0.0, model='80', lod=0.0015563):
    '''This function tranforms a vector from the true equator mean equinox frame
    (teme), to an earth fixed (ITRF) frame.  the results take into account
    the effects of sidereal time, and polar motion.

    *References:* Vallado  2007, 219-228.

    Author: David Vallado 719-573-2600, 10 dec 2007.
    Adapted to Python, Daniel Kastinen 2018

    :param numpy.ndarray t: numpy vector row of seconds relative to :code:`mjd0`
    :param numpy.ndarray p: numpy matrix of TEME positions, Cartesian coordinates in km. Columns correspond to times in t, and rows to x, y and z coordinates respectively.
    :param numpy.ndarray v: numpy matrix of TEME velocities, Cartesian coordinates in km/s. Columns correspond to times in t, and rows to x, y and z coordinates respectively.
    :param float mjd0: Modified julian date epoch that t vector is relative to
    :param float xp: x-axis polar motion coefficient in radians
    :param float yp: y-axis polar motion coefficient in radians
    :param str model: The polar motion model used in transformation, options are '80' or '00', see David Vallado documentation for more info.
    :param float lod: Excess length of day in seconds

    :return: State vector of position and velocity in km and km/s.
    :rtype: numpy.ndarray (6-D vector)

    **Uses:**

       * :func:`sorts.dates.gstime`
       * :func:`sorts.frames.polarm`

    '''


    if not isinstance(t, np.ndarray):
        if isinstance(t, float):
            t = np.array([t], dtype=np.float64)
        else:
            raise TypeError('type(t) = {} not supported'.format(type(t)))

    if len(p.shape) == 1:
        p = p.reshape(3,1)
    if len(v.shape) == 1:
         v = v.reshape(3,1)


    if t.size != p.shape[1]:
        raise Exception("t and p lengths conflicting, t {}, p {} ".format(t.size, p.shape[1]))


    if len(t.shape) == 1:
        try:
            t = t.reshape(1, t.size)
        except:
            print('Reshaping failed: t-{}, p-{}, v-{}'.format(
                t.shape, p.shape, v.shape,
            ))
            raise

    if p.shape[0] != 3 or v.shape[0] != 3:
        try:
            p = p.reshape(3, p.size)
            v = v.reshape(3, v.size)
        except:
            print('Reshaping failed: t-{}, p-{}, v-{}'.format(
                t.shape, p.shape, v.shape,
            ))
            raise


    if not isinstance(xp, np.ndarray) or not isinstance(yp, np.ndarray):
        if np.abs(xp) < 1e-10 and np.abs(yp) < 1e-10:
            model = ''
    else:
        if xp.size != t.size or yp.size != t.size:
            raise Exception("Polar motion lengths conflicting with time length, xp {}, yp {} ".format(xp.size, yp.size))
        if xp.shape != t.shape:
            xp = xp.reshape(1, t.size)
            yp = yp.reshape(1, t.size)

    #from MJD to J2000 relative JD to BC-relative JD to julian centuries, 2400000.5 - 2451545.0 = - 51544.5
    #a Julian year (symbol: a) is a unit of measurement of time defined as exactly 365.25 days of 86400 SI seconds each
    ttt = (mjd0 + t/86400.0 - 51544.5)/(365.25*100.0)

    #JDUT1 is actually in days from 4713 bc
    jdut1 = mjd0 + t/86400.0 + 2400000.5

    # ------------------------ find gmst --------------------------
    theta = dates.gstime( jdut1 );

    #find omega from nutation theory
    omega=  125.04452222  + ( -6962890.5390 *ttt + 7.455 *ttt*ttt + 0.008 *ttt*ttt*ttt ) / 3600.0;
    omega= np.radians(np.fmod( omega, 360.0  ));

    # ------------------------ find mean ast ----------------------
    # teme does not include the geometric terms here
    # after 1997, kinematic terms apply
    gmstg = np.where(
        jdut1 > 2450449.5,
        theta + 0.00264*np.pi /(3600*180)*np.sin(omega) + 0.000063*np.pi /(3600*180)*np.sin(2.0 *omega),
        theta,
    )
    gmstg = np.fmod( gmstg, 2.0*np.pi  )

    rpef=np.zeros(p.shape)
    vpef=np.zeros(v.shape)

    # direct calculation of rotation matrix with different rotation for each
    # column of p
    rpef[0,:]=np.multiply(np.cos(theta),p[0,:])+np.multiply(np.sin(theta),p[1,:])
    rpef[1,:]=-np.multiply(np.sin(theta),p[0,:])+np.multiply(np.cos(theta),p[1,:])
    rpef[2,:]=p[2,:]

    vpef[0,:]=np.multiply(np.cos(theta),v[0,:])+np.multiply(np.sin(theta),v[1,:])
    vpef[1,:]=-np.multiply(np.sin(theta),v[0,:])+np.multiply(np.cos(theta),v[1,:])
    vpef[2,:]=v[2,:]

    thetasa    = 7.29211514670698e-05*(1.0  - lod/86400.0)
    omegaearth = np.array([0., 0., thetasa])

    for ind in range(t.shape[1]):
        vpef[:,ind] -= np.cross(omegaearth, rpef[:,ind])

    recef = np.empty(rpef.shape, dtype=rpef.dtype)
    vecef = np.empty(vpef.shape, dtype=vpef.dtype)

    if len(model) == 0:
        recef = rpef
        vecef = vpef
    elif model == '80':
        if isinstance(xp, np.ndarray):
            for tind in range(t.size):
                pm = polar_motion_matrix(xp[0,tind], yp[0,tind], 0.0, model)
                recef[:,tind] = np.dot(pm, rpef[:,tind])
                vecef[:,tind] = np.dot(pm, vpef[:,tind])
        else:
            pm = polar_motion_matrix(xp, yp, 0.0, model)
            recef = np.dot(pm, rpef)
            vecef = np.dot(pm, vpef)
    elif model == '00':
        if isinstance(xp, np.ndarray):
            for tind in range(ttt.size):
                pm = polar_motion_matrix(xp[0,tind], yp[0,tind], ttt[0,tind], model)
                recef[:,tind] = np.dot(pm, rpef[:,tind])
                vecef[:,tind] = np.dot(pm, vpef[:,tind])
        else:
            for tind in range(ttt.size):
                pm = polar_motion_matrix(xp, yp, ttt[0,tind], model)
                recef[:,tind] = np.dot(pm, rpef[:,tind])
                vecef[:,tind] = np.dot(pm, vpef[:,tind])

    ecef = np.empty((6,t.size))
    ecef[:3,:] = recef
    ecef[3:,:] = vecef

    return ecef


def _sgp4_elems2cart(kep):
    '''Orbital elements to cartesian coordinates. Wrap pyorb-function to use mean anomaly, km and reversed order on aoe and raan. Output in SI.
    
    Neglecting mass is sufficient for this calculation (the standard gravitational parameter is 24 orders larger then the change).
    '''
    _kep = kep.copy()
    _kep[0] *= 1e3
    tmp = _kep[4]
    _kep[4] = _kep[3]
    _kep[3] = tmp
    _kep[5] = pyorb.mean_to_true(_kep[5], _kep[1], degrees=False)
    cart = pyorb.kep_to_cart(kep, mu=constants.WGS72.M_earth*pyorb.G, degrees=False)
    return cart

def _cart2sgp4_elems(cart, degrees=False):
    '''Cartesian coordinates to orbital elements. Wrap pyorb-function to use mean anomaly, km and reversed order on aoe and raan.
    
    Neglecting mass is sufficient for this calculation (the standard gravitational parameter is 24 orders larger then the change).
    '''
    kep = pyorb.cart_to_kep(cart, mu=constants.WGS72.M_earth*pyorb.G, degrees=False)
    kep[0] *= 1e-3
    tmp = kep[4]
    kep[4] = kep[3]
    kep[3] = tmp
    kep[5] = pyorb.true_to_mean(kep[5], kep[1], degrees=False)
    return kep



def TEME_to_TLE_OPTIM(state, mjd0, kepler=False, tol=1e-6, tol_v=1e-7, method=None):
    '''Convert osculating orbital elements in TEME
    to mean elements used in two line element sets (TLE's).

    :param numpy.ndarray kep: Osculating State (position and velocity) vector in m and m/s, TEME frame. If :code:`kepler = True` then state is osculating orbital elements, in m and radians. Orbital elements are semi major axis (m), orbital eccentricity, orbital inclination (radians), right ascension of ascending node (radians), argument of perigee (radians), mean anomaly (radians)
    :param bool kepler: Indicates if input state is kepler elements or cartesian.
    :param float mjd0: Modified Julian date for state, important for SDP4 iteration.
    :param float tol: Wanted precision in position of mean element conversion in m.
    :param float tol_v: Wanted precision in velocity mean element conversion in m/s.
    :param str method: Forces use of SGP4 or SDP4 depending on string 'n' or 'd', if None method is automatically chosen based on orbital period.
    :return: mean elements of: semi major axis (km), orbital eccentricity, orbital inclination (radians), right ascension of ascending node (radians), argument of perigee (radians), mean anomaly (radians)
    :rtype: numpy.ndarray
    '''
    
    if kepler:
        state_cart = _sgp4_elems2cart(state)
        init_elements = state
    else:
        state_cart = state
        init_elements = _cart2sgp4_elems(state_cart)
    
    def find_mean_elems(mean_elements):
        # Mean elements and osculating state
        state_osc = propagator.pysgp4.sgp4_propagation(mjd0, mean_elements, B=0.0, dt=0.0, method=method)

        # Correction of mean state vector
        d = state_cart - state_osc
        return np.linalg.norm(d*1e3)

    bounds = [(None, None), (0.0, 0.999), (0.0, np.pi)] + [(0.0, np.pi*2.0)]*3

    opt_res = scipy.optimize.minimize(find_mean_elems, init_elements,
        #method='powell',
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': np.sqrt(tol**2 + tol_v**2)}
    )
    mean_elements = opt_res.x

    return mean_elements

def TLE_to_TEME(state, mjd0, kepler=False):
    '''Convert mean elements used in two line element sets (TLE's) to osculating orbital elements in TEME.
        :param list/numpy.ndarray state : [a0,e0,i0,raan0,aop0,M0]
    '''
    return propagator.pysgp4.sgp4_propagation(mjd0, state, B=0.0, dt=0.0)


def TEME_to_TLE(state, mjd0, kepler=False, tol=1e-3, tol_v=1e-4):
    '''Convert osculating orbital elements in TEME
    to mean elements used in two line element sets (TLE's).

    :param numpy.ndarray kep: Osculating State (position and velocity) vector in m and m/s, TEME frame. If :code:`kepler = True` then state is osculating orbital elements, in m and radians. Orbital elements are semi major axis (m), orbital eccentricity, orbital inclination (radians), right ascension of ascending node (radians), argument of perigee (radians), mean anomaly (radians)
    :param bool kepler: Indicates if input state is kepler elements or cartesian.
    :param float mjd0: Modified Julian date for state, important for SDP4 iteration.
    :param float tol: Wanted precision in position of mean element conversion in m.
    :param float tol_v: Wanted precision in velocity mean element conversion in m/s.
    :return: mean elements of: semi major axis (km), orbital eccentricity, orbital inclination (radians), right ascension of ascending node (radians), argument of perigee (radians), mean anomaly (radians)
    :rtype: numpy.ndarray
    '''
    
    if kepler:
        state_mean = _sgp4_elems2cart(state)
        state_kep = state
        state_cart = state_mean.copy()
    else:
        state_mean = state.copy()
        state_cart = state
        state_kep = _cart2sgp4_elems(state)

    #method        : Forces use of SGP4 or SDP4 depending on string 'n' or 'd', None is automatic method
    # Fix used model (SGP or SDP)
    T = 2.0*np.pi*np.sqrt(np.power(state_kep[0]*1e-3, 3) / constants.WGS72.MU_earth)/60.0
    if T > 220.0:
        method = 'd'
    else:
        method = None

    iter_max = 300  # Maximum number of iterations

    # Iterative determination of mean elements
    for it in range(iter_max):
        # Mean elements and osculating state
        mean_elements = _cart2sgp4_elems(state_mean)

        if it > 0 and mean_elements[1] > 1:
            #Assumptions of osculation within slope not working, go to general minimization algorithms
            mean_elements = TEME_to_TLE_OPTIM(state_cart, mjd0=mjd0, kepler=False, tol=tol, tol_v=tol_v, method=method)
            break

        state_osc = propagator.pysgp4.sgp4_propagation(mjd0, mean_elements, B=0.0, dt=0.0, method=method)

        # Correction of mean state vector
        d = state_cart - state_osc
        state_mean += d
        if it > 0:
            dr_old = dr
            dv_old = dv

        dr = np.linalg.norm(d[:3])  # Position change
        dv = np.linalg.norm(d[3:])  # Velocity change

        if it > 0:
            if dr_old < dr or dv_old < dv:
                #Assumptions of osculation within slope not working, go to general minimization algorithms
                mean_elements = TEME_to_TLE_OPTIM(state_cart, mjd0=mjd0, kepler=False, tol=tol, tol_v=tol_v, method=method)
                break

        if dr < tol and dv < tol_v:   # Iterate until position changes by less than eps
            break
        if it == iter_max - 1:
            #Iterative method not working, go to general minimization algorithms
            mean_elements = TEME_to_TLE_OPTIM(state_cart, mjd0=mjd0, kepler=False, tol=tol, tol_v=tol_v, method=method)

    return mean_elements



def polar_motion_matrix( xp, yp, ttt, opt ):
    '''This function calculates the transformation matrix that accounts for polar motion. both the 1980 and 2000 theories are handled. note that the rotation order is different between 1980 and 2000.

    *References:* Vallado 2004, 207-209, 211, 223-224.

    Author: David Vallado 719-573-2600   25 jun 2002.
    Adapted to Python, Daniel Kastinen 2018 (2020)


    :param float xp: x-axis polar motion coefficient in radians
    :param float yp: y-axis polar motion coefficient in radians
    :param float ttt: Julian centuries of tt (00 theory only)
    :param str opt: Model for polar motion to use, options are '01', '02', '80'.

    :return: Transformation matrix for ECEF to PEF
    :rtype: numpy.ndarray (3x3 matrix)

    '''
    cosxp = np.cos(xp)
    sinxp = np.sin(xp)
    cosyp = np.cos(yp)
    sinyp = np.sin(yp)

    pm = np.zeros([3,3])

    if opt == '80':
        pm[0,0] =  cosxp
        pm[0,1] =  0.0
        pm[0,2] = -sinxp
        pm[1,0] =  sinxp * sinyp
        pm[1,1] =  cosyp
        pm[1,2] =  cosxp * sinyp
        pm[2,0] =  sinxp * cosyp
        pm[2,1] = -sinyp
        pm[2,2] =  cosxp * cosyp

    else:
        convrt = np.pi / (3600.0*180.0)
        # approximate sp value in rad
        sp = -47.0e-6 * ttt * convrt
        cossp = np.cos(sp)
        sinsp = np.sin(sp)

        # form the matrix
        pm[0,0] =  cosxp * cossp
        pm[0,1] = -cosyp * sinsp + sinyp * sinxp * cossp
        pm[0,2] = -sinyp * sinsp - cosyp * sinxp * cossp
        pm[1,0] =  cosxp * sinsp
        pm[1,1] =  cosyp * cossp + sinyp * sinxp * sinsp
        pm[1,2] =  sinyp * cossp - cosyp * sinxp * sinsp
        pm[2,0] =  sinxp
        pm[2,1] = -sinyp * cosxp
        pm[2,2] =  cosyp * cosxp
    return pm
