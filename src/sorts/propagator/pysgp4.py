#!/usr/bin/env python

"""wrapper for the SGP4 propagator"""
from typing import Any
import logging
from dataclasses import dataclass
import numpy as np
from astropy.time import Time, TimeDelta
import scipy.optimize
import pyorb
import sgp4
from sgp4.api import Satrec, SGP4_ERRORS
import sgp4.earth_gravity
import spacecoords.celestial as cel

from sorts.utils import convert_to_relative_time
from sorts.space_object import SpaceObject
from sorts.types import Settings, GravModels, NDArray_6, Frames, NDArray_6xN, NDArray_N
from .base import Propagator

logger = logging.getLogger(__name__)


def _sgp4_elems2cart(
    kep: NDArray_6xN | NDArray_6,
    grav_model,
) -> NDArray_6xN | NDArray_6:
    """Orbital elements to cartesian coordinates.
    Wrap pyorb-function to use mean anomaly, km and reversed order on aoe and raan.
    Output in SI.

    Neglecting mass is sufficient for this calculation
    (the standard gravitational parameter is 24 orders larger then the change).
    """
    _kep = kep.copy()
    _kep[0, ...] *= 1e3
    tmp = _kep[4, ...].copy()
    _kep[4, ...] = _kep[3, ...]
    _kep[3, ...] = tmp
    _kep[5, ...] = pyorb.mean_to_true(_kep[5, ...], _kep[1, ...], degrees=False)
    cart = pyorb.kep_to_cart(_kep, mu=grav_model.mu * 1e9, degrees=False)
    return cart


def _cart2sgp4_elems(
    cart: NDArray_6xN | NDArray_6,
    grav_model,
    degrees: bool = False,
) -> NDArray_6xN | NDArray_6:
    """Cartesian coordinates to orbital elements.
    Wrap pyorb-function to use mean anomaly, km and reversed order on aoe and raan.

    Neglecting mass is sufficient for this calculation
    (the standard gravitational parameter is 24 orders larger then the change).
    """
    kep = pyorb.cart_to_kep(cart, mu=grav_model.mu * 1e9, degrees=False)
    kep[0, ...] *= 1e-3
    tmp = kep[4, ...].copy()
    kep[4, ...] = kep[3, ...]
    kep[3, ...] = tmp
    kep[5, ...] = pyorb.true_to_mean(kep[5, ...], kep[1, ...], degrees=False)
    return kep


def kep_to_mean_elements(orb: pyorb.Orbit, degrees: bool = False) -> NDArray_6xN:
    """Orbital elements to the convention used by mean elements. Input assumes SI units.

    Neglecting mass is sufficient for this calculation
    (the standard gravitational parameter is 24 orders larger then the change).
    """
    kep = np.empty_like(orb._kep)
    kep[0, ...] = orb.a * 1e-3
    kep[1, ...] = orb.e
    kep[2, ...] = orb.i
    kep[3, ...] = orb.Omega
    kep[4, ...] = orb.omega
    kep[5, ...] = orb.mean_anomaly

    if orb.degrees and not degrees:
        kep[2:, :] = np.radians(kep[2:, :])
    elif not orb.degrees and degrees:
        kep[2:, :] = np.degrees(kep[2:, :])

    return kep


def get_TLE_parameters(
    line1: str, line2: str, gravity_model: GravModels = "WGS84"
) -> dict[str, Any]:
    line1, line2 = line_decode(line1), line_decode(line2)

    grav_ind = getattr(sgp4.api, gravity_model.upper())
    satellite = Satrec.twoline2rv(line1, line2, grav_ind)
    ret = {}
    for key in ["bstar", "satnum", "jdsatepochF", "jdsatepoch"]:
        ret[key] = getattr(satellite, key)
    return ret


def line_decode(line: str | np.bytes_) -> str:
    if isinstance(line, np.bytes_):
        rline = str(line.astype("U"))
    elif not isinstance(line, str):
        try:
            rline = line.decode()
        except (UnicodeDecodeError, AttributeError):
            pass
    else:
        rline = line

    return rline


def get_B(properties):
    # TODO: we need a way to specify which properties can exist and what they are somehow....
    if "B" in properties:
        B = properties["B"]
    elif "area_to_mass" in properties:
        B = 0.5 * properties.get("C_D", 2.3) * properties["area_to_mass"]
    elif "A" in properties and "m" in properties:
        B = 0.5 * properties.get("C_D", 2.3) * properties["A"] / properties["m"]
    elif "d" in properties and "m" in properties:
        A = np.pi * (properties["d"] / 2) ** 2
        B = 0.5 * properties.get("C_D", 2.3) * A / properties["m"]
    else:
        B = 0.0
    return B


@dataclass
class Sgp4Settings(Settings):
    out_frame: Frames = "TEME"
    gravity_model: GravModels = "WGS84"
    mean_elements_input: bool = False
    tol: float = 1e-5
    tol_v: float = 1e-7
    sample_space_object_kepler_orbit: bool = False
    kepler_samples: int = 100
    kepler_extent: float = 0.2
    teme_to_tle_max_iter: int = 300
    teme_to_tle_minimize_start_samples: int = 1
    teme_to_tle_minimize_start_stds: tuple[float, ...] = (10.0, 0.01, 1.0, 2.0, 2.0, 2.0)
    teme_to_tle_minimize_bounds: tuple[tuple[float, float], ...] = (
        (6371.0, np.inf),
        (0, 1),
        (0, np.pi),
        (0, 2 * np.pi),
        (0, 2 * np.pi),
        (0, 2 * np.pi),
    )


class Sgp4(Propagator[Sgp4Settings]):
    """Propagator class implementing the SGP4 propagator."""

    def __init__(self, settings: Sgp4Settings):
        super().__init__(settings=settings)

        self.sgp4_mjd0 = Time("1949-12-31 00:00:00", format="iso", scale="ut1").mjd
        self.rho0 = 2.461e-5 / 6378.135e3  # kg/m^2/m

    def propagate_tle(
        self,
        line1: str,
        line2: str,
        times: Time | TimeDelta | NDArray_N,
    ) -> NDArray_6xN:
        """Propagate a TLE pair"""

        grav_ind = getattr(sgp4.api, self.settings.gravity_model.upper())
        # grav_model = getattr(sgp4.earth_gravity, self.settings.gravity_model.lower())
        line1, line2 = line_decode(line1), line_decode(line2)

        satellite = Satrec.twoline2rv(line1, line2, grav_ind)

        epoch = Time(satellite.jdsatepoch + satellite.jdsatepochF, format="jd", scale="utc")
        logger.debug(f"SGP4:propagate_tle:epoch={epoch}")
        tv = convert_to_relative_time(epoch, times)
        td = TimeDelta(tv, format="sec")

        times = epoch + td

        jd_f = times.jd2
        jd0 = times.jd1

        if isinstance(jd_f, float) or isinstance(jd_f, int):
            states = np.empty((6,), dtype=np.float64)
            error, r, v = satellite.sgp4(jd0, jd_f)
            states[:3] = r
            states[3:] = v
            errors = [error]
        else:
            states = np.empty((6, jd_f.size), dtype=np.float64)
            errors, r, v = satellite.sgp4_array(jd0, jd_f)
            states[:3, ...] = r.T
            states[3:, ...] = v.T

        for ind, err in enumerate(errors):
            if err != 0:
                logger.error(f"SGP4:propagate_tle:step-{ind}:{SGP4_ERRORS[err]}")

        states *= 1e3  # km to m, km/s to m/s

        if self.settings.out_frame != "TEME":
            states = cel.convert(
                times,
                states,
                in_frame="TEME",
                out_frame=self.settings.out_frame,
                frame_kwargs={},
            )

        return states

    def propagate(
        self,
        space_object: SpaceObject,
        times: Time | TimeDelta | NDArray_N,
    ) -> NDArray_6xN:
        """Propagate a state

        `space_object` properties only information needed for ballistic coefficient
        `B` used by SGP4. Either `B` or `C_D`, `A` and `m` must be supplied.

        - B: Ballistic coefficient
        - C_D: Drag coefficient
        - A: Cross-sectional Area
        - m: Mass
        """
        logger.debug("SGP4:propagate")
        if space_object.state.num > 1:
            t_samps = space_object.properties["state_sample_times"]
        else:
            t_samps = None

        tv = convert_to_relative_time(space_object.epoch, times)
        td = TimeDelta(tv, format="sec")
        t = space_object.epoch + td

        sgp4_epoch = space_object.epoch.mjd - self.sgp4_mjd0

        B = get_B(space_object.properties)
        logger.debug(f"SGP4:propagate:B = {B}")
        state0 = space_object.state.copy()

        if space_object.frame != "TEME":
            state0._cart = cel.convert(
                space_object.epoch,
                state0._cart,
                in_frame=space_object.frame,
                out_frame="TEME",
                frame_kwargs={},
            )
            state0.calculate_kepler()

        if self.settings.mean_elements_input:
            mean_elements = kep_to_mean_elements(state0, degrees=False)
            assert mean_elements.size == 6, "Can not propagate multiple objects"
            if len(mean_elements.shape) > 2:
                mean_elements.shape = (mean_elements.size,)
        else:
            if self.settings.sample_space_object_kepler_orbit:
                mean_elements = self.space_object_to_mean_elements(
                    space_object,
                )
            else:
                mean_elements = self.TEME_to_TLE(
                    state0._cart,
                    t=t_samps,
                    epoch=space_object.epoch,
                    B=B,
                    tol=self.settings.tol,
                    tol_v=self.settings.tol_v,
                )

        if np.any(np.isnan(mean_elements)):
            raise Exception("Could not compute SGP4 initial state: {}".format(mean_elements))

        states = self.propagate_mean_elements(
            t.jd1,
            t.jd2,
            mean_elements,
            sgp4_epoch,
            B,
        )

        if self.settings.out_frame != "TEME":
            states = cel.convert(
                t,
                states,
                in_frame="TEME",
                out_frame=self.settings.out_frame,
                frame_kwargs={},
            )

        logger.debug("SGP4:propagate:completed")

        return states

    def get_mean_elements(
        self,
        line1: str,
        line2: str,
        radians: bool = False,
    ) -> tuple[NDArray_6, float, Time]:
        """Extract the mean elements in SI units (a [m], e [1], inc [deg],
        raan [deg], aop [deg], mu [deg]), B-parameter (not bstar) and epoch
        from a two line element pair.
        """

        grav_ind = getattr(sgp4.api, self.settings.gravity_model.upper())
        grav_model = getattr(sgp4.earth_gravity, self.settings.gravity_model.lower())
        line1, line2 = line_decode(line1), line_decode(line2)

        xpdotp = 1440.0 / (2.0 * np.pi)  # 229.1831180523293

        satrec = Satrec.twoline2rv(line1, line2, grav_ind)

        B = satrec.bstar / (grav_model.radiusearthkm * 1e3) * 2 / self.rho0

        epoch = Time(satrec.jdsatepoch + satrec.jdsatepochF, format="jd", scale="utc")

        mean_elements = np.zeros((6,), dtype=np.float64)

        n0 = satrec.no_kozai * xpdotp / (86400.0 / (2 * np.pi))

        mean_elements[0] = (np.sqrt(grav_model.mu) / n0) ** (2.0 / 3.0) * 1e3
        mean_elements[1] = satrec.ecco
        mean_elements[2] = satrec.inclo
        mean_elements[3] = satrec.nodeo
        mean_elements[4] = satrec.argpo
        mean_elements[5] = satrec.mo
        if not radians:
            mean_elements[2:] = np.degrees(mean_elements[2:])

        return mean_elements, B, epoch

    def propagate_mean_elements(
        self,
        jd0: NDArray_N | float,
        jd_f: NDArray_N | float,
        mean_elements: NDArray_6,
        sgp4_epoch: float,
        B: float,
    ) -> NDArray_6xN | NDArray_6:
        """Propagate sgp4 mean elements."""

        grav_ind = getattr(sgp4.api, self.settings.gravity_model.upper())
        grav_model = getattr(sgp4.earth_gravity, self.settings.gravity_model.lower())
        # Compute ballistic coefficient
        bstar = 0.5 * B * self.rho0  # B* in [1/m] using Density at q0[kg/m^3]
        n0 = np.sqrt(grav_model.mu) / ((mean_elements[0]) ** 1.5)

        # Scaling
        n0 = n0 * (86400.0 / (2 * np.pi))  # Convert to [rev/d]
        bstar = bstar * (grav_model.radiusearthkm * 1e3)  # Convert from [1/m] to [1/R_EARTH]

        satellite = Satrec()
        satellite.sgp4init(
            grav_ind,  # gravity model
            "i",  # 'a' = old AFSPC mode, 'i' = improved mode
            42,  # satnum: Satellite number
            sgp4_epoch,  # epoch: days since 1949 December 31 00:00 UT
            bstar,  # bstar: drag coefficient (/earth radii)
            0.0,  # [IGNORED BY SGP4] ndot: ballistic coefficient (revs/day)
            0.0,  # [IGNORED BY SGP4] nddot: second derivative of mean motion (revs/day^3)
            mean_elements[1],  # ecco: eccentricity
            mean_elements[4],  # argpo: argument of perigee (radians)
            mean_elements[2],  # inclo: inclination (radians)
            mean_elements[5],  # mo: mean anomaly (radians)
            n0 / (1440.0 / (2.0 * np.pi)),  # no_kozai: mean motion (radians/minute)
            mean_elements[3],  # nodeo: right ascension of ascending node (radians)
        )

        if isinstance(jd_f, float) or isinstance(jd_f, int):
            states = np.empty((6,), dtype=np.float64)
            error, r, v = satellite.sgp4(jd0, jd_f)
            states[:3] = r
            states[3:] = v
            errors = [error]
        else:
            states = np.empty((6, jd_f.size), dtype=np.float64)
            errors, r, v = satellite.sgp4_array(jd0, jd_f)
            states[:3, ...] = r.T
            states[3:, ...] = v.T

        for ind, err in enumerate(errors):
            if err != 0:
                logger.error(f"SGP4:propagate:step-{ind}:{SGP4_ERRORS[err]}")

        states *= 1e3  # km to m and km/s to m/s

        return states

    def TEME_to_TLE_OPTIM(
        self,
        cart: NDArray_6xN | NDArray_6,
        epoch: Time,
        t: NDArray_N | None = None,
        B: float = 0.0,
        tol: float = 1e-8,
        tol_v: float = 1e-9,
    ) -> NDArray_6:
        """Convert osculating orbital elements in TEME
        to mean elements used in two line element sets (TLE's).
        """
        logger.debug("SGP4:TEME_to_TLE_OPTIM")
        grav_model = getattr(sgp4.earth_gravity, self.settings.gravity_model.lower())

        if len(cart.shape) == 1:
            cart.shape = (cart.size, 1)

        if cart.shape[1] > 1 and t is None:
            raise ValueError(
                'Cannot convert TEME sampling to TLE without sample times "state_sample_times"'
            )
        elif t is None:
            t = np.array([0.0])

        t_min = np.argmin(np.abs(t))
        if t[t_min] > 1e-6:
            raise ValueError(
                "There is no sampling point at the epoch (t=0) to use as initial guess..."
            )

        init_elements = _cart2sgp4_elems(cart[:, t_min], grav_model=grav_model, degrees=False)

        t = epoch + TimeDelta(t, format="sec")

        def find_mean_elems(mean_elements):
            # Mean elements and osculating state
            state_osc = self.propagate_mean_elements(
                t.jd1,
                t.jd2,
                mean_elements,
                epoch.mjd - self.sgp4_mjd0,
                B=B,
            )

            d = cart - state_osc
            return np.mean(np.linalg.norm(d, axis=0))

        dx_std = np.array(self.settings.teme_to_tle_minimize_start_stds)
        samps = self.settings.teme_to_tle_minimize_start_samples
        bounds = self.settings.teme_to_tle_minimize_bounds

        opt_res = None
        for j in range(samps):
            _init_elements = init_elements.copy()
            if j > 0:
                _init_elements += np.random.randn(6) * dx_std

            for mni in range(6):
                if _init_elements[mni] < bounds[mni][0]:
                    _init_elements[mni] = bounds[mni][0]
                elif _init_elements[mni] > bounds[mni][1]:
                    _init_elements[mni] = bounds[mni][1]

            _opt_res = scipy.optimize.minimize(
                find_mean_elems,
                _init_elements,
                method="Nelder-Mead",
                bounds=bounds,
                options={
                    "fatol": np.sqrt(tol**2 + tol_v**2),
                },
            )
            if j > 0:
                if _opt_res.fun < opt_res.fun:  # type: ignore
                    opt_res = _opt_res
            else:
                opt_res = _opt_res

        mean_elements = opt_res.x  # type: ignore

        logger.debug("SGP4:TEME_to_TLE_OPTIM:completed")

        return mean_elements

    def TEME_to_TLE(
        self,
        cart: NDArray_6xN | NDArray_6,
        epoch: Time,
        t: NDArray_N | None = None,
        B: float = 0.0,
        tol: float = 1e-5,
        tol_v: float = 1e-7,
    ) -> NDArray_6:
        """Convert osculating orbital elements in TEME
        to mean elements used in two line element sets (TLE's).

        Parameters
        ----------
        cart
            Osculating State (position and velocity) vector in m and m/s,
            TEME frame.
        tol
            Wanted precision in position of mean element conversion in m.
        tol_v
            Wanted precision in velocity mean element conversion in m/s.

        Notes
        ----------
        mean elements of:
        - semi major axis (km)
        - orbital eccentricity
        - orbital inclination (radians)
        - right ascension of ascending node (radians)
        - argument of perigee (radians)
        - mean anomaly (radians)

        """
        logger.debug("SGP4:TEME_to_TLE")
        mean_elements = None
        grav_model = getattr(sgp4.earth_gravity, self.settings.gravity_model.lower())

        if len(cart.shape) > 1:
            if cart.size > 6:
                mean_elements = self.TEME_to_TLE_OPTIM(
                    cart,
                    epoch=epoch,
                    t=t,
                    B=B,
                    tol=tol,
                    tol_v=tol_v,
                )

                logger.debug("SGP4:TEME_to_TLE:completed")

                return mean_elements
            else:
                cart.shape = (cart.size,)

        state_mean = cart.copy()
        iter_max = self.settings.teme_to_tle_max_iter
        dr = 0
        dv = 0
        # Iterative determination of mean elements
        for it in range(iter_max):
            # Mean elements and osculating state
            mean_elements = _cart2sgp4_elems(state_mean, grav_model, degrees=False)

            if it > 0 and mean_elements[1] > 1:
                # Assumptions of osculation within slope not working
                # go to general minimization algorithms
                mean_elements = self.TEME_to_TLE_OPTIM(
                    cart,
                    epoch=epoch,
                    B=B,
                    tol=tol,
                    tol_v=tol_v,
                )
                break

            state_osc = self.propagate_mean_elements(
                epoch.jd1,
                epoch.jd2,
                mean_elements,
                epoch.mjd - self.sgp4_mjd0,
                B=B,
            )

            # Correction of mean state vector
            d = cart - state_osc
            state_mean += d
            if it > 0:
                dr_old = dr
                dv_old = dv

            dr = np.linalg.norm(d[:3, ...], axis=0)  # Position change
            dv = np.linalg.norm(d[3:, ...], axis=0)  # Velocity change

            if it > 0:
                if dr_old < dr or dv_old < dv:
                    # Assumptions of osculation within slope not working
                    # go to general minimization algorithms
                    mean_elements = self.TEME_to_TLE_OPTIM(
                        cart,
                        epoch=epoch,
                        B=B,
                        tol=tol,
                        tol_v=tol_v,
                    )
                    break

            if dr < tol and dv < tol_v:  # Iterate until position changes by less than eps
                break
            if it == iter_max - 1:
                # Iterative method not working, go to general minimization algorithms
                mean_elements = self.TEME_to_TLE_OPTIM(
                    cart,
                    epoch=epoch,
                    B=B,
                    tol=tol,
                    tol_v=tol_v,
                )

        logger.debug("SGP4:TEME_to_TLE:completed")

        return mean_elements  # type: ignore

    def space_object_to_mean_elements(
        self,
        space_object: SpaceObject,
    ) -> NDArray_6:
        assert space_object.state.num <= 1
        samples = self.settings.kepler_samples
        max_ang = 360.0 if space_object.state.degrees else 2 * np.pi
        max_ang *= self.settings.kepler_extent

        _orb = space_object.state.copy()
        _orb.allocate(samples)
        _orb._kep[()] = space_object.state._kep[()]
        _orb._kep[5, :] = np.linspace(-max_ang/2, max_ang/2, num=samples, endpoint=False)
        _orb.calculate_cartesian()
        t_vec = _orb.mean_anomaly / _orb.mean_motion

        return self.TEME_to_TLE_OPTIM(
            cart=_orb._cart,
            epoch=space_object.epoch,
            t=t_vec,
            B=get_B(space_object.properties),
            tol=self.settings.tol,
            tol_v=self.settings.tol_v,
        )
