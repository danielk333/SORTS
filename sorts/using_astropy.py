# TEME:True equator, mean equinox frame use for NORAD TLE files

import astropy.units as u
from astropy.coordinates.matrix_utilities import matrix_transpose
from astropy.coordinates.builtin_frames.utils import get_polar_motion, get_jd12
from astropy.time import Time, TimeDelta
from astropy.coordinates import (BaseCoordinateFrame, CartesianRepresentation,
                                 CartesianDifferential, TimeAttribute, ITRS,
                                 frame_transform_graph,
                                 FunctionTransformWithFiniteDifference)
from astropy import _erfa as erfa
import numpy as np

# Stuff scarfed from gdar
def time_derivative(f, t, dt=None):
    """
    Approximate time derivative by calling a function of time twice,
    at instants `dt/2` before and after `t`, and dividing the difference
    by `dt`.

    The types of `t` and `dt` must be such that `dt` can be added to or
    subtracted from `t`, and `dt` must have a property `dt.value` which gives a
    number.  By default `dt` is 0.1 second as an `astropy.time.TimeDelta`,
    which means `t` should be of the class `astropy.time.Time`.
    """
    if dt is None:
        dt = TimeDelta(0.1, format='sec')
    return (f(t+dt/2) - f(t-dt/2))/dt.value

def function_and_derivative(f):
    """
    Decorator for functions f(t) of a single time variable.

    If the decorated function is called with the keyword
    parameter `derivative=True` then the function and its derivative
    (computed using `time_derivative`) will be returned.
    Otherwise, the function will be called as if undecorated.

    The time variable `t` must be of type `astropy.time.Time`,
    and the step `dt`, if given, must be of type
    `astropy.time.TimeDelta` (default = 0.1 second).
    """
    def r_rdot(t, derivative=None, dt=None, **kw):
        r = f(t)
        if derivative is None:
            return r
        rdot = time_derivative(f, t, dt=dt, **kw)
        return r, rdot
    return r_rdot


def coord_from_svec(svec, frame):
    """
    params:
      svec: structured scalar with fields POS, VEL, UTC
      frame: Some Cartesian coordinate frame, e.g. ITRS or TEME
    """
    p = CartesianRepresentation(svec.POS * u.m)
    v = CartesianDifferential(svec.VEL * u.m/u.s)
    return frame(p.with_differentials(v), obstime=Time(svec.UTC, scale='utc'))


class TEME(BaseCoordinateFrame):
    """
    A coordinate or frame in the True Equator Mean Equinox frame (TEME).

    This frame is a geocentric system similar to CIRS or geocentric apparent place,
    except that the mean sidereal time is used to rotate from TIRS. TEME coordinates
    are most often used in combination with orbital data for satellites in the
    two-line-ephemeris format.

    Different implementations of the TEME frame exist. For clarity, this frame follows the
    conventions and relations to other frames that are set out in Vallado et al (2006).
    """
    default_representation = CartesianRepresentation
    default_differential = CartesianDifferential
    obstime = TimeAttribute()
    # obstime = TimeAttribute(default=Time("J2000"))

@function_and_derivative
def teme_to_itrs_mat(time):
    # Sidereal time, rotates from ITRS to mean equinox
    gst = erfa.gmst82(*get_jd12(time, 'ut1'))
    # Polar Motion
    xp, yp = get_polar_motion(time)
    sp = erfa.sp00(*get_jd12(time, 'tt'))
    pmmat = erfa.pom00(xp, yp, sp)
    # rotation matrix
    return erfa.c2tcio(np.eye(3), gst, pmmat)

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, TEME, ITRS)
def teme_to_itrs(teme_coo, itrs_frame):
    # first get us to TEME at the target obstime
    # TODO: define self transform
    teme_coo2 = teme_coo.transform_to(TEME(obstime=itrs_frame.obstime))
    # now get the pmatrix
    pmat = teme_to_itrs_mat(itrs_frame.obstime)
    crepr = teme_coo2.cartesian.transform(pmat)
    return itrs_frame.realize_frame(crepr)

@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ITRS, TEME)
def itrs_to_teme(itrs_coo, teme_frame):
    # compute the pmatrix, and then multiply by its transpose
    pmat = teme_to_itrs_mat(itrs_coo.obstime)
    newrepr = itrs_coo.cartesian.transform(matrix_transpose(pmat))
    teme = TEME(newrepr, obstime=itrs_coo.obstime)
    # now do any needed offsets (no-op if same obstime)
    return teme.transform_to(teme_frame)

