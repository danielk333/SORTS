from collections import namedtuple

import erfa
import numpy as np

# import pyorb
from astropy.coordinates.builtin_frames.utils import get_jd12, get_polar_motion
from astropy.time import Time, TimeDelta

KeplerianElements = namedtuple("KeplerianElements", "a e i Omega omega M")

STATEVECTOR_DTYPE = [
    ("utc", "datetime64[ns]"),  # absolute time in UTC
    ("pos", np.double, (3,)),  # ECEF position [m]
    ("vel", np.double, (3,)),
]


def simple_decorator(decorator):
    """
    This decorator can be used to turn simple functions into well-behaved decorators,
    so long as the decorators are fairly simple. If a decorator expects a function and
    returns a function (no descriptors), and if it doesn't modify function attributes
    or docstring, then it is eligible to use this. Simply apply @simple_decorator to
    your decorator and it will automatically preserve the docstring and function
    attributes of functions to which it is applied.

    Taken from
    https://wiki.python.org/moin/PythonDecoratorLibrary#Creating_Well-Behaved_Decorators_.2F_.22Decorator_decorator.22
    """

    def new_decorator(f):
        g = decorator(f)
        g.__name__ = f.__name__
        g.__doc__ = f.__doc__
        g.__dict__.update(f.__dict__)
        return g

    # Now a few lines needed to make simple_decorator itself be a well-behaved decorator.
    new_decorator.__name__ = decorator.__name__
    new_decorator.__doc__ = decorator.__doc__
    new_decorator.__dict__.update(decorator.__dict__)
    return new_decorator


def time_derivative(f, t, dt=None, **kw):
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
        dt = TimeDelta(0.1, format="sec")
    return (f(t + dt / 2, **kw) - f(t - dt / 2, **kw)) / dt.value


@simple_decorator
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
        r = f(t, **kw)
        if derivative is None:
            return r
        rdot = time_derivative(f, t, dt=dt, **kw)
        return r, rdot

    return r_rdot


###############################################################################
# Rotation matrix between coordinates in Earth-centered Inertial (GCRS)
# and Earth-centered Earth-fixed (ITRS) frames.  Uses ERFA library from IERS,
# the shareable and linkable version of the SOFA standards,
# (wrappers included in the astropy package).
###############################################################################


@function_and_derivative
def itrs_from_gcrs(time, inverse=False):
    xp, yp = get_polar_motion(time)
    sp = erfa.sp00(*get_jd12(time, "tt"))
    pom = erfa.pom00(xp, yp, sp)  # Polar motion
    era = erfa.era00(*get_jd12(time, "ut1"))  # Earth rotation angle
    pn = erfa.c2i06a(*get_jd12(time, "tt"))  # Precession/nutation IAU2006A
    R_gi = erfa.c2tcio(pn, era, pom)
    if not inverse:
        return R_gi
    if len(R_gi.shape) == 2:
        return R_gi.T
    return np.swapaxes(R_gi, -2, -1)


def _get_earthmodel(model="EGM96"):
    """
    Retrieve the low-order coefficients for Earth gravitational model.
    Returns a tuple containing
      re     : Earth equatorial radius [m]
      xmu    : Earth gravitational constant [M*G]
      j2,3,4 : unnormalized zonal coefficient
    """


    EarthModel = namedtuple("EarthModel", "re xmu j2 j3 j4")
    em = {}
    match model.upper():
        case "EGM96":
            ## EGM96 gravitational model parameters
            # Earth equatorial radius  [m]
            em["re"] = 6378.1363e3
            # Earth graviational constant xmu = earth mass * gravitational constant
            em["xmu"] = 3.986005e14
            # unnormalized zonal gravity coefficients (jn = sqrt(2n+1) jnbar)
            em["j2"] = 1.08262668355e-3
            #       0.0010826266835531513
            em["j3"] = -2.53265648533e-6
            #           2.5326564853322355e-06
            em["j4"] = -1.61962159137e-6
            #           1.619621591367e-0
        case "JGM3":
            ## JGM-3 Joint Earth Gravity Model parameters
            ## ref: https://en.wikipedia.org/wiki/Geopotential_spherical_harmonic_model#Available_models
            # Earth equatorial radius  [m]
            em["re"] = 6378.1363e3
            # Earth graviational constant xmu = earth mass * gravitational constant
            em["xmu"] = 3.986004415e14
            # unnormalized zonal gravity coefficients
            em["j2"] = 1.082635854e-3
            em["j3"] = -2.532435346e-6
            em["j4"] = -1.619331205e-6
        case _:
            raise ValueError(f"Unknown earth model name: {model}")
    return EarthModel(**em)


def ecef_from_eci(sv_eci, inverse=False):
    t = Time(sv_eci["utc"], scale="utc")  # astropy time
    R, dR = itrs_from_gcrs(t, derivative=True, inverse=inverse)
    p, v = sv_eci["pos"], np.float64(sv_eci["vel"])
    sv_ecef = np.empty_like(sv_eci)
    sv_ecef["utc"] = sv_eci["utc"]
    if np.iterable(sv_ecef):
        for i, sv in enumerate(sv_ecef):
            sv["pos"] = R[i] @ p[i]
            sv["vel"] = R[i] @ v[i] + dR[i] @ p[i]
    else:
        sv_ecef["pos"] = R @ p
        sv_ecef["vel"] = R @ v + dR @ p
    return sv_ecef


def eci_from_ecef(sv_ecef):
    return ecef_from_eci(sv_ecef, inverse=True)


def arg2pi(th):
    return _modulus(th)


def argpi(th):
    return _modulus(th, center_zero=True)


def _modulus(x, m=2 * np.pi, center_zero=False):
    y = x if np.isscalar(x) else np.copy(x)
    y %= m
    if not center_zero:
        return y
    if np.isscalar(y):
        return y - m if y >= m / 2 else y
    y[y >= m / 2] -= m
    return y


def var_xgl(xm0, dt, earth_model, dxm=False):
    """
    variational equations for the long periodic changes in the
    orbital elements due to the second order earth oblateness
    potential including   j2, (j2)**2, j3, j4    .
    cf. :  liu , aiaa-paper 79-0123 , appendix c .

    i n p u t :
    xm0 - input mean orbital elements at ref. epoch
        xm0[0] = semimajor axis, m
        xm0[1] = eccentricity, -
        xm0[2] = inclination, rad
        xm0[3] = right ascension of ascending node, rad
        xm0[4] = argument of perigee, rad
        xm0[5] = mean anomaly, rad
    dt  - prediction time interval, sec

    o u t p u t :
    xm     - propagated mean orbital elements after "dt"
    dxdgm  - averaged time rate of change of "xm" due
             to the earth oblateness potential, [ ]/sec
             (averaging interval one anomalistic period)
    """

    re, xmu, j2, j3, j4 = earth_model

    j2p2 = j2**2

    a, ec, inc, _, om, _ = xm0
    p = a * (1 - ec**2)
    n = np.sqrt(xmu / a**3)
    reovp = re / p
    reovp2, reovp3, reovp4 = reovp**2, reovp**3, reovp**4
    sini = np.sin(inc)
    # sinip3 = sini**3
    sinip2, sinip4 = sini**2, sini**4
    cosi = np.cos(inc)
    cosip2 = cosi**2
    sin2i = 2 * sini * cosi
    sinom = np.sin(om)
    cosom = np.cos(om)
    sin2om = 2 * sinom * cosom
    cos2om = cosom**2 - sinom**2
    # sin4om = 2*sin2om*cos2om
    cos4om = cos2om**2 - sin2om**2
    ec2 = ec**2
    ec4 = ec**4
    t1ec2 = 1 - ec2
    r1ec2 = np.sqrt(t1ec2)

    xm = np.zeros(6)
    dxdgm = np.zeros(6)

    # averaged time rate of change of semi major axis :
    dxdgm[0] = 0
    # averaged time rate of change of eccentricity :
    dxdgm[1] = (
        -(
            (3 / 32 * n * j2p2 * reovp4 * sinip2)
            * ((14 - 15 * sinip2) * ec * t1ec2 * sin2om)
        )
        - (3 / 8 * n * j3 * reovp3 * sini * (4 - 5 * sinip2) * t1ec2 * cosom)
        - (15 / 32 * n * j4 * reovp4 * sinip2 * (6 - 7 * sinip2) * ec * t1ec2 * sin2om)
    )
    # averaged time rate of change of inclination :
    dxdgm[2] = (
        +(3 / 64 * n * j2p2 * reovp4 * sin2i * (14 - 15 * sinip2) * ec2 * sin2om)
        + (3 / 8 * n * j3 * reovp3 * cosi * (4 - 5 * sinip2) * ec * cosom)
        + (15 / 64 * n * j4 * reovp4 * sin2i * (6 - 7 * sinip2) * ec2 * sin2om)
    )
    # averaged time rate of change of ascending node :
    dxdgm[3] = (
        (-3 / 2 * n * j2 * reovp2 * cosi)
        - (3 / 2 * n * j2p2 * reovp4 * cosi)
        * (
            (9 / 4 + 3 / 2 * r1ec2)
            - (sinip2 * (5 / 2 + 9 / 4 * r1ec2))
            + ec2 / 4 * (1 + 5 / 4 * sinip2)
            + ec2 / 8 * (7 - 15 * sinip2) * cos2om
        )
        - (3 / 8 * n * j3 * reovp3 * (15 * sinip2 - 4) * ec * cosi / sini * sinom)
        + (15 / 16 * n * j4 * reovp4 * cosi)
        * (((4 - 7 * sinip2) * (1 + 3 / 2 * ec2)) - (3 - 7 * sinip2) * ec2 * cos2om)
    )
    # averaged time rate of change of argument of perigee :
    dxdgm[4] = (
        +(3 / 4 * n * j2 * reovp2 * (4 - 5 * sinip2))
        + (3 / 16 * n * j2p2 * reovp4)
        * (
            (48 - 103 * sinip2 + 215 / 4 * sinip4)
            + (7 - 9 / 2 * sinip2 - 45 / 8 * sinip4) * ec2
            + 6 * (1 - 3 / 2 * sinip2) * (4 - 5 * sinip2) * r1ec2
            - (
                (1 / 4)
                * (
                    2 * (14 - 15 * sinip2) * sinip2
                    - (28 - 158 * sinip2 + 135 * sinip4) * ec2
                )
            )
            * cos2om
        )
        + (3 / 8 * n * j3 * reovp3)
        * (
            (4 - 5 * sinip2) * (sinip2 - ec2 * cosip2) / (ec * sini)
            + 2 * sini * (13 - 15 * sinip2) * ec
        )
        * sinom
        - (15 / 32 * n * j4 * reovp4)
        * (
            (16 - 62 * sinip2 + 49 * sinip4)
            + 3 / 4 * (24 - 84 * sinip2 + 63 * sinip4) * ec2
            + (
                sinip2 * (6 - 7 * sinip2)
                - 1 / 2 * (12 - 70 * sinip2 + 63 * sinip4) * ec2
            )
            * cos2om
        )
    )
    # averaged time rate of change of mean anomaly (mean mean motion) :
    dxdgm[5] = (
        n * (1 + 3 / 2 * j2 * reovp2 * (1 - 3 / 2 * sinip2) * r1ec2)
        + (3 / 2 * n * j2p2 * reovp4)
        * (
            (np.abs(1 - 3 / 2 * sinip2)) ** 2 * t1ec2
            + (
                5 / 4 * (1 - 5 / 2 * sinip2 + 13 / 8 * sinip4)
                + 5 / 8 * (1 - sinip2 + 5 / 8 * sinip4) * ec2
                + 1 / 16 * sinip2 * (14 - 15 * sinip2) * (1 - 5 / 2 * ec2) * cos2om
            )
            * r1ec2
        )
        + (3 / 8 * n * j2p2 * reovp4 / r1ec2)
        * (
            3
            * (
                3
                - 15 / 2 * sinip2
                + 47 / 8 * sinip4
                + (3 / 2 - 5 * sinip2 + 117 / 6 * sinip4) * ec2
                - 1 / 8 * (1 + 5 * sinip2 - 101 / 8 * sinip4) * ec4
            )
            + ec2 / 8 * sinip2 * (70 - 123 * sinip2 + (56 - 66 * sinip2) * ec2) * cos2om
            + 27 / 128 * ec4 * sinip4 * cos4om
        )
        - (3 / 8 * n * j3 * reovp3 * sini)
        * (4 - 5 * sinip2)
        * (1 - 4 * ec2)
        / ec
        * r1ec2
        * sinom
        - 45 / 128 * n * j4 * reovp4 * (8 - 40 * sinip2 + 35 * sinip4) * ec2 * r1ec2
        + (15 / 64 * n * j4 * reovp4 * sinip2)
        * ((6 - 7 * sinip2) * (2 - 5 * ec2) * r1ec2 * cos2om)
    )

    # propagation of the mean elements over time interval dt :
    xm[0] = xm0[0]
    xm[1] = xm0[1] + dt * dxdgm[1]
    xm[2:6] = arg2pi(xm0[2:6] + dt * dxdgm[2:6])
    if dxm:
        return xm, dxdgm
    else:
        return xm


def var_xgs(xm, earth_model, dx=False):
    """
    yields first order short periodic variations of the orbital
    elements due to the first order earth oblateness potential (j2) .
    cf.:  liu , aiaa-paper 79-0123 , appendix b

    i n p u t :
         xm  -  mean orbital elements (see eq.3 from above)
            xm[0] = semimajor axis, km
            xm[1] = eccentricity, -
            xm[2] = inclination, rad
            xm[3] = right ascension of ascending node, rad
            xm[4] = argument of perigee, rad
            xm[5] = mean anomaly, rad

    o u t p u t :
         dxgs   ,dblarr(6) -  short periodic perturbation amplitudes of
                              the orbital elements due to j2
         x      ,dblarr(6) -  osculating elements (1-st order, due to j2)
    """

    re, j2 = earth_model.re, earth_model.j2

    a, ec, inc, _, om, m = xm

    f = true_from_mean(m, ec)
    fminm = argpi(f - m)
    sinf = np.sin(f)
    cosf = np.cos(f)
    sin2f = 2 * sinf * cosf
    cos2f = cosf**2 - sinf**2
    sin3f = sinf * (3 - 4 * sinf**2)
    cos3f = -cosf * (3 - 4 * cosf**2)
    sin2om = np.sin(2 * om)
    cos2om = np.cos(2 * om)
    sinmf2 = sin2om * cosf - cos2om * sinf
    cosmf2 = cos2om * cosf + sin2om * sinf
    sin1f2 = sin2om * cosf + cos2om * sinf
    cos1f2 = cos2om * cosf - sin2om * sinf
    sin2f2 = sin1f2 * cosf + cos1f2 * sinf
    cos2f2 = cos1f2 * cosf - sin1f2 * sinf
    sin3f2 = sin2f2 * cosf + cos2f2 * sinf
    cos3f2 = cos2f2 * cosf - sin2f2 * sinf
    sin4f2 = sin3f2 * cosf + cos3f2 * sinf
    cos4f2 = cos3f2 * cosf - sin3f2 * sinf
    sin5f2 = sin4f2 * cosf + cos4f2 * sinf
    cos5f2 = cos4f2 * cosf - sin4f2 * sinf
    ec2 = ec**2
    t1ec2 = 1 - ec2
    r1ec2 = np.sqrt(t1ec2)
    p = a * (1 - ec2)
    r = p / (1 + ec * cosf)
    reovp2 = (re / p) ** 2
    aovr3 = (a / r) ** 3
    sini = np.sin(inc)
    sinip2 = sini**2
    cosi = np.cos(inc)
    sin2i = 2 * sini * cosi

    dxgs = np.zeros(6)
    xe = np.zeros(6)
    x = np.zeros(6)

    # short periodic variation of semi major axis :
    dxgs[0] = (j2 * re**2 / a) * (
        aovr3 * ((1 - 3 / 2 * sinip2) + 3 / 2 * sinip2 * cos2f2)
        - (1 - 3 / 2 * sinip2) * r1ec2 ** (-3)
    )

    # short periodic variation of eccentricity :
    dxgs[1] = 1 / 2 * j2 * reovp2 * (1 - 3 / 2 * sinip2) * (
        1 / ec * (1 + 3 / 2 * ec2 - r1ec2**3)
        + 3 * (1 + ec2 / 4) * cosf
        + 3 / 2 * ec * cos2f
        + ec2 / 4 * cos3f
    ) + 3 / 8 * j2 * reovp2 * sinip2 * (
        (1 + 11 / 4 * ec2) * cos1f2
        + ec2 / 4 * cosmf2
        + 5 * ec * cos2f2
        + 1 / 3 * (7 + 17 / 4 * ec2) * cos3f2
        + 3 / 2 * ec * cos4f2
        + ec2 / 4 * cos5f2
        + 3 / 2 * ec * cos2om
    )
    # short periodic variation of inclination :
    dxgs[2] = 3 / 8 * j2 * reovp2 * sin2i * (ec * cos1f2 + cos2f2 + ec / 3 * cos3f2)
    # short periodic variation of the ascending node :
    dxgs[3] = (-3 / 2 * j2 * reovp2 * cosi) * (
        fminm + ec * sinf - ec / 2 * sin1f2 - 1 / 2 * sin2f2 - ec / 6 * sin3f2
    )
    # short periodic variation of the argument of perigee :
    dxgs[4] = (
        3 / 4 * j2 * reovp2 * (4 - 5 * sinip2) * (fminm + ec * sinf)
        + (3 / 2 * j2 * reovp2)
        * (1 - 3 / 2 * sinip2)
        * (1 / ec * (1 - 1 / 4 * ec2) * sinf + 1 / 2 * sin2f + 1 / 12 * ec * sin3f)
        - (3 / 2 * j2 * reovp2)
        * (
            1 / ec * (1 / 4 * sinip2 + ec2 / 2 * (1 - 15 / 8 * sinip2)) * sin1f2
            + ec / 16 * sinip2 * sinmf2
            + 1 / 2 * (1 - 5 / 2 * sinip2) * sin2f2
            - 1 / ec * (7 / 12 * sinip2 - ec2 / 6 * (1 - 19 / 8 * sinip2)) * sin3f2
            - 3 / 8 * sinip2 * sin4f2
            - 1 / 16 * ec * sinip2 * sin5f2
        )
        - (9 / 16 * j2 * reovp2 * sinip2 * sin2om)
    )
    # short periodic variation of the mean anomaly :
    dxgs[5] = (-3 / 2 * j2 * reovp2 * r1ec2 / ec) * (
        (1 - 3 / 2 * sinip2)
        * ((1 - 1 / 4 * ec2) * sinf + ec / 2 * sin2f + ec2 / 12 * sin3f)
        + (1 / 2 * sinip2)
        * (
            -1 / 2 * (1 + 5 / 4 * ec2) * sin1f2
            - ec2 / 8 * sinmf2
            + 7 / 6 * (1 - ec2 / 28) * sin3f2
            + 3 / 4 * ec * sin4f2
            + ec2 / 8 * sin5f2
        )
    ) + (9 / 16 * j2 * reovp2 * r1ec2 * sinip2 * sin2om)

    # conversion: mean to 1-st order (j2) osculating equinoctial elem.
    sinom = np.sin(xm[4])
    cosom = np.cos(xm[4])
    xe[0] = xm[0] + dxgs[0]
    xe[1] = xm[1] * cosom + dxgs[1] * cosom - xm[1] * dxgs[4] * sinom
    xe[2] = xm[1] * sinom + dxgs[1] * sinom + xm[1] * dxgs[4] * cosom
    xe[3] = xm[2] + dxgs[2]
    xe[4] = xm[3] + dxgs[3]
    xe[5] = xm[4] + dxgs[4] + xm[5] + dxgs[5]

    # reconversion to osculating kepler elements to 1-st order (j2)
    x[0] = xe[0]
    x[1] = np.sqrt(xe[1] ** 2 + xe[2] ** 2)
    x[2:4] = arg2pi(xe[3:5])
    x[4] = arg2pi(np.arctan2(xe[2], xe[1]))
    x[5] = arg2pi(xe[5] - x[4])

    if dx:
        return x, dxgs
    return x


def keplerian_elements(sv_eci, earth_model):
    pos, vel = sv_eci["pos"], sv_eci["vel"]

    xmu = earth_model.xmu
    rmu = np.sqrt(xmu)
    r = np.linalg.norm(pos, 2)
    v = np.linalg.norm(vel, 2)
    d = np.dot(pos, vel) / rmu

    # Semimajor axis [m]
    a = 1 / (2 / r - v**2 / xmu)

    dp = rmu * (1 / r - 1 / a)

    # Eccentricity
    e = np.sqrt((r**2 / xmu * dp**2) + d**2 / a)

    # eccentric anomaly
    sinE, cosE = d / (np.sqrt(a) * e), r * dp / (rmu * e)
    E = np.arctan2(sinE, cosE)

    # mean anomaly [rad]
    M = E - e * sinE  # Kepler's equation

    w = np.cross(pos, vel)
    w /= np.linalg.norm(w, 2)
    sini, cosi = np.linalg.norm(w[:2], 2), w[2]

    # Inclination [rad, 0-2pi]
    inc = arg2pi(np.arctan2(sini, cosi))

    sinO, cosO = w[0] / sini, -w[1] / sini

    # Right ascension of ascending node [rad, 0-2pi]
    Omega = arg2pi(np.arctan2(sinO, cosO))

    sinv, cosv = a * np.sqrt(1 - e**2) * sinE / r, a * (cosE - e) / r

    # True anomaly [rad, 0-2pi]
    nu = arg2pi(np.arctan2(sinv, cosv))

    cosw, sinw = (pos[0] * cosO + pos[1] * sinO) / r, pos[2] / (sini * r)

    # Argument of periapsis [rad, 0-2pi]
    omega = arg2pi(-nu + np.arctan2(sinw, cosw))

    return KeplerianElements(a=a, e=e, i=inc, Omega=Omega, omega=omega, M=M)


def kepler_to_equinoctial(x, dx=None):
    xe = np.zeros_like(x)
    if dx is None:
        xe[0] = x[0]
        xe[1] = x[1] * np.cos(x[4])
        xe[2] = x[1] * np.sin(x[4])
        xe[3] = x[2]
        xe[4] = x[3]
        xe[5] = arg2pi(x[4] + x[5])
    else:
        xe[0] = dx[0]
        xe[1] = dx[1] * np.cos(x[4]) - x[1] * dx[4] * np.sin(x[4])
        xe[2] = dx[1] * np.sin(x[4]) + x[1] * dx[4] * np.cos(x[4])
        xe[3] = dx[2]
        xe[4] = dx[3]
        xe[5] = dx[4] + dx[5]
    return KeplerianElements(*xe)


def equinoctial_to_kepler(x, dx=None):
    xe = np.array(x)
    ec2 = x[1] ** 2 + x[2] ** 2
    if dx is None:
        xe[1] = np.sqrt(ec2)
        xe[2] = x[3]
        xe[3] = x[4]
        xe[4] = np.arctan2(x[2] / xe[1], x[1] / xe[1])
        xe[5] = arg2pi(xe[5] - xe[4])
    else:
        ec = np.sqrt(ec2)
        xe[0] = dx[0]
        xe[1] = x[1] * dx[1] / ec + x[2] * dx[2] / ec
        xe[2] = dx[3]
        xe[3] = dx[4]
        xe[4] = x[1] * dx[2] / ec2 - x[2] * dx[1] / ec2
        xe[5] = dx[5] - xe[4]

    xe[2:6] = arg2pi(xe[2:6])
    return KeplerianElements(*xe)


def kepler_osculating_to_mean(osc, earth_model, niter=100):
    """
       converts a set of osculating orbital elements into a set of mean
       (averaged over the mean anomaly) elements.
       cf.:  j.j.f. liu, aiaa-paper #79-0123 (1979)

    i n p u t :
        osc - osculating kepler orbital elements
                     .a = semimajor axis, km
                     .e = eccentricity, -
                     .i = inclination, rad
                     .Omega = right ascension of ascending node, rad
                     .omega = argument of perigee, rad
                     .M = mean anomaly, rad

    o u t p u t :
        - mean orbital elements
          (angular variables returned as 0 < ang <+2*pi )
    """

    dtor = np.deg2rad(1)

    xeref = np.array([1e3, 1e-4, 1e-4 * dtor, 1e-4 * dtor, 1e-4 * dtor, 1e-4 * dtor])
    tol = 1e-3

    xe0 = kepler_to_equinoctial(osc)
    xmei = np.array(xe0)

    converged = False
    for i in range(niter):
        rms = 0.0
        # reconvert to mean kepler elements
        xmi = equinoctial_to_kepler(xmei)
        #  compute short periodic perturbations
        _, dxgms = var_xgs(xmi, earth_model, dx=True)
        # convert to variances of the mean equinoctial elements
        dxmei = kepler_to_equinoctial(xmi, dx=dxgms)

        xe = xmei + dxmei
        dxei = xe0 - xe
        rms += sum(dxei**2 / xeref**2)
        xmei += dxei
        if np.sqrt(rms) < tol:
            converged = True
            break
    if not converged:
        raise ValueError("No convergence in osculating to mean kepler conversion.")

    return equinoctial_to_kepler(xmei)


def true_from_eccentric(E, e):
    return 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.sin(E / 2) / np.cos(E / 2))


def eccentric_from_mean(M, e, niter=100):
    """
    Solve the Kepler equation
      M = E - e * sin(E)

    input:
        e, M - kepler elements (eccentricity, mean anomaly)

    output:
        E - eccentric anomaly [rad]
    """
    tol = 1e-15
    E = argpi(M)
    converged = False
    for _ in range(niter):
        f = E - e * np.sin(E) - M
        df = 1 - e * np.cos(E)
        dE = f / df
        E -= dE
        if abs(dE) < tol:
            converged = True
            break
    if not converged:
        raise ValueError("No convergence in Kepler equation solver.")
    return E


def true_from_mean(M, e, *kw):
    E = eccentric_from_mean(M, e, *kw)
    return true_from_eccentric(E, e)


def find_true_ascending_node(sv_ecef, earth_model, niter=100):
    """
    determination of the mean kepler state vector at true ascending
    node crossing from input osculating cartesian
    state vector.
    returns: mean kepler orbital elements at true node
                         xm[0] = semimajor axis, [m]
                         xm[1] = eccentricity, -
                         xm[2] = inclination, [rad]
                         xm[3] = right ascension of ascending node, [rad]
                         xm[4] = argument of perigee, [rad]
                         xm[5] = mean anomaly, [rad]
    ;          (angular variables returned as 0 < ang <+2*pi )
    """
    # true-of-date rotating, earthfixed system to inertial system
    sv_eci = eci_from_ecef(sv_ecef)
    # osculating cartesian --> osculating kepler state
    osc = keplerian_elements(sv_eci, earth_model)
    # osculating kepler --> mean kepler state
    xmp = kepler_osculating_to_mean(osc, earth_model)
    # iterate for true ascending node crossing (to 1.e-6 sec accuracy)
    tov2pi = np.sqrt(xmp[0] ** 3 / earth_model.xmu)
    dt = 0.0
    converged = False
    for _ in range(niter):
        xm = var_xgl(xmp, -dt, earth_model)
        x = var_xgs(xm, earth_model)
        nu_anx = true_from_mean(x[5], x[1])
        ulat = argpi(nu_anx + x[4])
        dt += ulat * tov2pi
        if np.abs(ulat * tov2pi) < 1e-10:
            converged = True
            break
    if not converged:
        raise ValueError("Non convergence in search for true ascending node.")

    utc_anx = sv_eci["utc"] - np.timedelta64(int(dt * 1e9 + 0.5), "ns")
    return KeplerianElements(*xm), nu_anx, utc_anx


def osculating_to_statevector(x, nu, em, acceleration=False):
    xmu, re, j2 = em.xmu, em.re, em.j2
    a, e, i, Omega, omega, _ = x
    #    namedtuple("KeplerianElements", "a e i Omega omega M")
    ulat = omega + nu
    culat, sulat = np.cos(ulat), np.sin(ulat)
    cgom, sgom = np.cos(Omega), np.sin(Omega)
    cinc, sinc = np.cos(i), np.sin(i)
    termv = 1 + e * np.cos(nu)
    terme = 1 - e**2
    r = a * terme / termv
    uu = np.float64(
        (
            culat * cgom - sulat * sgom * cinc,
            culat * sgom + sulat * cgom * cinc,
            sulat * sinc,
        )
    )
    #  radius vector in the geocentric coordinate system:
    pos = r * uu

    term = np.sqrt(xmu / (a * terme))
    rdu = term * e * np.sin(nu)
    rdv = term * termv
    uv = np.float64(
        (
            -sulat * cgom - culat * sgom * cinc,
            -sulat * sgom + culat * cgom * cinc,
            culat * sinc,
        )
    )
    #  orbit velocity vector in the equatorial system:
    vel = rdu * uu + rdv * uv
    if not acceleration:
        return pos, vel

    #  first order (j2) orbit acceleration vector in the equator syst.:
    term = xmu / r**2
    termj2 = -1.5 * term * j2 * (re / r) ** 2
    termi = 2 * termj2 * sinc * sulat
    r2du = -term - termj2 * (3 * (sinc * sulat) ** 2 - 1)
    r2dv = termi * sinc * culat
    r2dw = termi * cinc
    uw = np.float64((sgom * sinc, -cgom * sinc, cinc))
    acc = r2du * uu + r2dv * uv + r2dw * uw
    return pos, vel, acc


def _propagate(dt, kepler_mean_anx, earth_model):
    """
    General purpose orbit generator :
    First order prediction of mean elements with explicit second
    order corrections extended 1-st order (j2,j2**2,j3,j4)
    """

    #     prediction of the mean state
    kepler_mean = var_xgl(kepler_mean_anx, dt, earth_model)
    #     first order (j2) recovery of the osculating state
    k = KeplerianElements(*var_xgs(kepler_mean, earth_model))
    nu = true_from_mean(k.M, k.e)
    return k, nu


class Propagator:
    def __init__(self, statevector, earth_model=None):
        self.sv = np.copy(statevector)
        self.earth_model = _get_earthmodel() if earth_model is None else earth_model
        xm, nu_anx, utc_anx = find_true_ascending_node(self.sv, self.earth_model)
        self.utc_anx = utc_anx
        self.nu_anx = nu_anx
        self.kepler_mean = xm

    def propagate(self, t, eci=False):
        # TODO: convert to TAI to be robust wrt leap seconds
        dt = (t - self.utc_anx) / np.timedelta64(int(1e9), "ns")
        x, nu = _propagate(dt, self.kepler_mean, self.earth_model)
        pos, vel = osculating_to_statevector(x, nu, self.earth_model)
        sv_eci = np.array((t, pos, vel), dtype=STATEVECTOR_DTYPE)
        if eci:
            return sv_eci
        return ecef_from_eci(sv_eci)


if __name__ == "__main__":
    niter = 100
    sec = np.timedelta64(int(1e9), "ns")
    
    sv = np.array(
        (
            "2024-03-20T07:31:23.000000000",
            [6421750.29710361, -2382718.05529875, 1770357.98428398],
            [1194.0525, -2152.724, -7188.086],
        ),
        dtype=STATEVECTOR_DTYPE,
    )

    if False:
        sv_eci = eci_from_ecef(sv)
        ke = keplerian_elements(sv_eci, _get_earthmodel())
        mu = pyorb.M_earth * pyorb.G
        cart = np.concatenate((sv_eci["pos"], sv_eci["vel"]))
        k = pyorb.cart_to_kep(cart, mu=mu)
        k[-1] = pyorb.true_to_mean(k[-1], k[1])
        k[3], k[4] = k[4], k[3]
        assert np.allclose(k, np.array(ke)), "Keplerian elements differ: pyorb vs mine"

    p = Propagator(sv)
    t = (0.1 * np.arange(-500, 501)) * sec

    # svf = p.propagate(sv["utc"] + t)

    svi = np.array([p.propagate(sv["utc"] + tt, eci=True) for tt in t])
    svf = np.array([p.propagate(sv["utc"] + tt, eci=False) for tt in t])
    svg = ecef_from_eci(svi)
    assert np.all(svf == svg)

    pass
