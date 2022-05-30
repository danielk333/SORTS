#!/usr/bin/env python

'''Coordinate frame transformations and related functions. Main usage is the :code:`convert` function that wraps Astropy frame transformations.

'''

#Python standard import
from collections import OrderedDict

#Third party import
import numpy as np
import astropy.coordinates as coord
import astropy.units as units

from pyant.coordinates import cart_to_sph, sph_to_cart, vector_angle
from pyant.coordinates import rot_mat_x, rot_mat_y, rot_mat_z

try:
    from jplephem.spk import SPK
except ImportError:
    SPK = None

#Local import


'''List of astropy frames
'''
ASTROPY_FRAMES = {
    'TEME': 'TEME',
    'ITRS': 'ITRS',
    'ITRF': 'ITRS',
    'ICRS': 'ICRS',
    'ICRF': 'ICRS',
    'GCRS': 'GCRS',
    'GCRF': 'GCRS',
    'HCRS': 'HCRS',
    'HCRF': 'HCRS',
    'HeliocentricMeanEcliptic'.upper(): 'HeliocentricMeanEcliptic',
    'GeocentricMeanEcliptic'.upper(): 'GeocentricMeanEcliptic',
    'HeliocentricTrueEcliptic'.upper(): 'HeliocentricTrueEcliptic',
    'GeocentricTrueEcliptic'.upper(): 'GeocentricTrueEcliptic',
    'BarycentricMeanEcliptic'.upper(): 'BarycentricMeanEcliptic',
    'BarycentricTrueEcliptic'.upper(): 'BarycentricTrueEcliptic',
    'SPICEJ2000': 'ICRS',
}

ASTROPY_NOT_OBSTIME = [
    'ICRS',
    'BarycentricMeanEcliptic',
    'BarycentricTrueEcliptic',
]

'''Mapping from body name to integer id's used by the kernels.

Taken from `astropy.coordinates.solar_system`
'''
BODY_NAME_TO_KERNEL_SPEC = OrderedDict([
    ('sun', [(0, 10)]),
    ('mercury', [(0, 1), (1, 199)]),
    ('venus', [(0, 2), (2, 299)]),
    ('earth-moon-barycenter', [(0, 3)]),
    ('earth', [(0, 3), (3, 399)]),
    ('moon', [(0, 3), (3, 301)]),
    ('mars', [(0, 4)]),
    ('jupiter', [(0, 5)]),
    ('saturn', [(0, 6)]),
    ('uranus', [(0, 7)]),
    ('neptune', [(0, 8)]),
    ('pluto', [(0, 9)])
])


def not_geocentric(frame):
    '''Check if the given frame name is one of the non-geocentric frames.
    '''
    frame = frame.upper()
    return frame in ['ICRS', 'ICRF', 'HCRS', 'HCRF'] or frame.startswith('Heliocentric'.upper())

def is_geocentric(frame):
    '''Check if the frame name is a supported geocentric frame
    '''
    return not not_geocentric(frame)

def arctime_to_degrees(minutes, seconds):
    return (minutes + seconds/60.0)/60.0


def get_solarsystem_body_states(bodies, epoch, kernel, units=None):
    '''Open a kernel file and get the statates of the given bodies at epoch in ICRS.

    Note: All outputs from kernel computations are in the Barycentric (ICRS) "eternal" frame. 
    '''
    assert SPK is not None, 'jplephem package needed to directly interact with kernels'
    states = {}

    kernel = SPK.open(kernel)

    epoch_ = epoch.tdb #jplephem uses Barycentric Dynamical Time (TDB)
    jd1, jd2 = epoch_.jd1, epoch_.jd2

    for body in bodies:
        body_ = body.lower().strip()

        if body_ not in BODY_NAME_TO_KERNEL_SPEC:
            raise ValueError(f'Body name "{body}" not recognized')

        states[body] = np.zeros((6,), dtype=np.float64)

        #if there are multiple steps to go from states to 
        #ICRS barycentric, iterate trough and combine
        for pair in BODY_NAME_TO_KERNEL_SPEC[body_]:
            spk = kernel[pair]
            if spk.data_type == 3:
                # Type 3 kernels contain both position and velocity.
                posvel = spk.compute(jd1, jd2).flatten()
            else:
                pos_, vel_ = spk.compute_and_differentiate(jd1, jd2)
                posvel = np.zeros((6,), dtype=np.float64)
                posvel[:3] = pos_
                posvel[3:] = vel_
            
            states[body] += posvel

        #units from kernels are usually in km and km/day
        if units is None:
            states[body] *= 1e3
            states[body][3:] /= 86400.0
        else:
            states[body] *= units[0]
            states[body][3:] /= units[1]

    return states


def convert(t, states, in_frame, out_frame, logger=None, profiler=None, **kwargs):
    '''Perform predefined coordinate transformations using Astropy. Always returns a copy of the array.

    :param numpy.ndarray/float t: Absolute time corresponding to the input states.
    :param numpy.ndarray states: Size `(6,n)` matrix of states in SI units where rows 1-3 are position and 4-6 are velocity.
    :param str in_frame: Name of the frame the input states are currently in.
    :param str out_frame: Name of the state to transform to.
    :param Profiler profiler: Profiler instance for checking function performance.
    :param logging.Logger logger: Logger instance for logging the execution of the function.
    :rtype: numpy.ndarray
    :return: Size `(6,n)` matrix of states in SI units where rows 1-3 are position and 4-6 are velocity.

    '''

    if logger is not None:
        logger.info(f'frames:convert: in_frame={in_frame}, out_frame={out_frame}')
    if profiler is not None:
        profiler.start(f'frames:convert:{in_frame}->{out_frame}')

    in_frame = in_frame.upper()
    out_frame = out_frame.upper()

    if in_frame == out_frame:
        return states.copy()


    if in_frame in ASTROPY_FRAMES:
        in_frame_ = ASTROPY_FRAMES[in_frame]
        in_frame_cls = getattr(coord, in_frame_)
    else:
        raise ValueError(f'In frame "{in_frame}" not recognized, please check spelling or perform manual transformation')
    
    kw = {}
    kw.update(kwargs)
    if in_frame_ not in ASTROPY_NOT_OBSTIME:
        kw['obstime'] = t

    astropy_states = _convert_to_astropy(states, in_frame_cls, **kw)
    
    if out_frame in ASTROPY_FRAMES:
        out_frame_ = ASTROPY_FRAMES[out_frame]
        out_frame_cls = getattr(coord, out_frame_)
    else:
        raise ValueError(f'Out frame "{out_frame}" not recognized, please check spelling or perform manual transformation')

    kw = {}
    kw.update(kwargs)
    if out_frame_ not in ASTROPY_NOT_OBSTIME:
        kw['obstime'] = t

    out_states = astropy_states.transform_to(out_frame_cls(**kw))

    rets = states.copy()
    rets[:3,...] = out_states.cartesian.xyz.to(units.m).value
    rets[3:,...] = out_states.velocity.d_xyz.to(units.m / units.s).value
    
    if logger is not None:
        logger.info('frames:convert:completed')
    if profiler is not None:
        profiler.stop(f'frames:convert:{in_frame}->{out_frame}')

    return rets


def geodetic_to_ITRS(lat, lon, alt, radians=False, ellipsoid=None):
    '''Use `astropy.coordinates.EarthLocation` to transform from geodetic to ITRS.
    '''

    if not radians:
        lat, lon = np.radians(lat), np.radians(lon)

    cord = coord.EarthLocation.from_geodetic(
        lon=lon*units.rad, 
        lat=lat*units.rad, 
        height=alt*units.m,
        ellipsoid=ellipsoid,
    )
    x,y,z = cord.to_geocentric()

    pos = np.empty((3,), dtype=np.float64)

    pos[0] = x.to(units.m).value
    pos[1] = y.to(units.m).value
    pos[2] = z.to(units.m).value

    return pos


def ITRS_to_geodetic(x, y, z, radians=False, ellipsoid=None):
    '''Use `astropy.coordinates.EarthLocation` to transform from geodetic to ITRS.

    :param float x: X-coordinate in ITRS
    :param float y: Y-coordinate in ITRS
    :param float z: Z-coordinate in ITRS
    :param bool radians: If :code:`True` then all values are given in radians instead of degrees.
    :param str/None ellipsoid: Name of the ellipsoid model used for geodetic coordinates, for default value see Astropy `EarthLocation`.
    :rtype: numpy.ndarray
    :return: (3,) array of longitude, latitude and height above ellipsoid
    '''

    cord = coord.EarthLocation.from_geocentric(
        x=x*units.m, 
        y=y*units.m, 
        z=z*units.m,
    )
    lon, lat, height = cord.to_geodetic(ellipsoid=ellipsoid)
    
    llh = np.empty((3,), dtype=np.float64)
    
    if radians:
        u_ = units.rad
    else:
        u_ = units.deg
    llh[0] = lat.to(u_).value
    llh[1] = lon.to(u_).value
    llh[2] = height.to(units.m).value
    
    return llh


def _convert_to_astropy(states, frame, **kw):
    state_p = coord.CartesianRepresentation(states[:3,...]*units.m)
    state_v = coord.CartesianDifferential(states[3:,...]*units.m/units.s)
    astropy_states = frame(state_p.with_differentials(state_v), **kw)
    return astropy_states


def enu_to_ecef(lat, lon, alt, enu, radians=False):
    '''ENU (east/north/up) to ECEF coordinate system conversion, not including translation. 

    :param float lat: Latitude on the ellipsoid
    :param float lon: Longitude on the ellipsoid
    :param float alt: Altitude above ellipsoid, **Unused in this implementation**.
    :param numpy.ndarray enu: (3,n) input matrix of positions in the ENU-convention.
    :param bool radians: If :code:`True` then all values are given in radians instead of degrees.
    :rtype: numpy.ndarray
    :return: (3,n) array x,y and z coordinates in ECEF.
    '''
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    alt = np.asarray(alt)
    
    if np.shape(lat) != np.shape(lon) or np.shape(lat) != np.shape(alt): raise ValueError("lat, lon and alt must be the same shape.")
    
    if not radians:
        lat, lon = np.radians(lat), np.radians(lon)

    mx = np.array([[-np.sin(lon), -np.sin(lat) * np.cos(lon), np.cos(lat) * np.cos(lon)],
                [np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat) * np.sin(lon)],
                [np.zeros(np.size(lat, axis=0)), np.cos(lat), np.sin(lat)]], dtype=np.float64).reshape(np.size(lat, axis=0), 3, 3)

    ecef = np.tensordot(mx, enu, axes=([2],[0]))
    
    return ecef 

def ned_to_ecef(lat, lon, alt, ned, radians=False):
    '''NED (north/east/down) to ECEF coordinate system conversion, not including translation.

    :param float/ndarray lat: Latitude on the ellipsoid
    :param float/ndarray lon: Longitude on the ellipsoid
    :param float/ndarray alt: Altitude above ellipsoid, **Unused in this implementation**.
    :param numpy.ndarray ned: (3,n) input matrix of positions in the NED-convention.
    :param bool radians: If :code:`True` then all values are given in radians instead of degrees.
    :rtype: numpy.ndarray
    :return: (3,n) array x,y and z coordinates in ECEF.
    '''
    ned = np.asarray(ned)
    
    enu = np.empty([ned.size], dtype=ned.dtype)
    
    enu[0,...] = ned[1,...]
    enu[1,...] = ned[0,...]
    enu[2,...] = -ned[2,...]

    return enu_to_ecef(lat, lon, alt, enu, radians=radians)


def ecef_to_enu(lat, lon, alt, ecef, radians=False):
    '''ECEF coordinate system to local ENU (east,north,up), not including translation.

    :param float/ndarray lat: Latitude on the ellipsoid
    :param float/ndarray lon: Longitude on the ellipsoid
    :param float/ndarray alt: Altitude above ellipsoid, **Unused in this implementation**.
    :param numpy.ndarray ecef: (3,n) array x,y and z coordinates in ECEF.
    :param bool radians: If :code:`True` then all values are given in radians instead of degrees.
    :rtype: numpy.ndarray
    :return: (3,n) array x,y and z in local coordinates in the ENU-convention.
    '''
    
    ecef = np.asarray(ecef)
    
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    alt = np.asarray(alt)
    
    if np.shape(lat) != np.shape(lon) or np.shape(lat) != np.shape(alt): raise ValueError("lat, lon and alt must be the same shape.")    
    if len(lat.shape) == 0:
        lat=lat[None]
        lon=lon[None]
        alt=alt[None]
    
    if not radians:
        lat, lon = np.radians(lat), np.radians(lon)

    mx = np.array([[-np.sin(lon), -np.sin(lat) * np.cos(lon), np.cos(lat) * np.cos(lon)],
                [np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat) * np.sin(lon)],
                [np.zeros(np.size(lat, axis=0)), np.cos(lat), np.sin(lat)]]).reshape(np.shape(lat)[0], 3, 3)
    
    
    enu = np.tensordot(np.linalg.inv(mx), ecef, axes=([2],[0]))
    
    return enu


def azel_to_ecef(lat, lon, alt, az, el, radians=False):
    '''Radar pointing (az,el) to unit vector in ECEF, not including translation.

    TODO: Docstring
    '''
    az = np.asarray(az)
    el = np.asarray(el)
    
    if np.shape(az) != np.shape(el): raise ValueError("az and el must be the same shape.")    

    shape = (3,1)
    
    if isinstance(az,np.ndarray):
        if len(az.shape) == 0:
            az = float(az)
        elif len(az) > 1:
            shape = (3,len(az))
            az = az.flatten()
        else:
            az = az[0]
    
    if isinstance(el,np.ndarray):
        if len(el.shape) == 0:
            el = float(el)
        elif len(el) > 1:
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

def vec_to_vec(vec_in, vec_out):
    '''Get the rotation matrix that rotates `vec_in` to `vec_out` along the plane containing both. Uses quaternion calculations.
    '''

    N = len(vec_in)
    if N != len(vec_out):
        raise ValueError('Input and output vectors must be same dimensionality.')

    assert N == 3, 'Only implemented for 3d vectors'

    a = vec_in/np.linalg.norm(vec_in)
    b = vec_out/np.linalg.norm(vec_out)

    adotb = np.dot(a,b)
    axb = np.cross(a,b)
    axb_norm = np.linalg.norm(axb)

    #rotation in the plane frame of `vec_in` and `vec_out`
    G = np.zeros((N,N), dtype=vec_in.dtype)
    G[0,0] = adotb
    G[0,1] = -axb_norm
    G[1,0] = axb_norm
    G[1,1] = adotb
    G[2,2] = 1

    #inverse of change of basis from standard orthonormal to `vec_in` and `vec_out` plane
    F = np.zeros((N,N), dtype=vec_in.dtype)
    F[:,0] = a
    F[:,1] = (b - adotb*a)/np.linalg.norm(b - adotb*a)
    F[:,2] = axb

    #go to frame, rotation in plane, leave frame
    R = F @ G @ np.linalg.inv(F)

    return R