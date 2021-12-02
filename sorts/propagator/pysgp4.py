#!/usr/bin/env python

'''rapper for the SGP4 propagator

'''

#Python standard import
from copy import copy

#Third party import
import numpy as np
from astropy.time import Time, TimeDelta
import scipy.optimize
import pyorb

import sgp4
from sgp4.api import Satrec, SGP4_ERRORS
import sgp4.earth_gravity

#Local import
from .base import Propagator
from .. import dates
from .. import frames






class SGP4(Propagator):
    '''Propagator class implementing the SGP4 propagator.

    Frame options are found in the `sorts.frames.convert` function.

    :ivar str in_frame: String identifying the input frame. 
    :ivar str out_frame: String identifying the output frame. 

    :param str in_frame: String identifying the input frame.
    :param str out_frame: String identifying the output frame. 
    '''

    DEFAULT_SETTINGS = copy(Propagator.DEFAULT_SETTINGS)
    DEFAULT_SETTINGS.update(
        dict(
            out_frame = 'TEME',
            in_frame = 'TEME',
            gravity_model = 'WGS84',
            TEME_to_TLE_max_iter = 300,
            tle_input = False,
        )
    )
    

    def __init__(self, settings=None, **kwargs):
        super(SGP4, self).__init__(settings=settings, **kwargs)
        if self.logger is not None:
            self.logger.debug(f'sorts.propagator.SGP4:init')

        self.sgp4_mjd0 = Time('1949-12-31 00:00:00', format='iso', scale='ut1').mjd
        self.rho0 = 2.461e-5/6378.135e3 #kg/m^2/m

        self.grav_ind = getattr(sgp4.api, self.settings['gravity_model'].upper())
        self.grav_model = getattr(sgp4.earth_gravity, self.settings['gravity_model'].lower())


    @staticmethod
    def get_TLE_parameters(line1, line2, gravity_model = 'WGS84'):

        line1, line2 = SGP4.line_decode(line1), SGP4.line_decode(line2)

        grav_ind = getattr(sgp4.api, gravity_model.upper())
        satellite = Satrec.twoline2rv(line1, line2, grav_ind)
        ret = {}
        for key in ['bstar', 'satnum', 'jdsatepochF', 'jdsatepoch']:
            ret[key] = getattr(satellite, key)
        return ret


    @staticmethod
    def line_decode(line):

        if isinstance(line, np.bytes_):
            line = line.astype('U')
        elif not isinstance(line, str):
            try:
                line = line.decode()
            except (UnicodeDecodeError, AttributeError):
                pass

        return line


    def propagate_tle(self, t, line1, line2, **kwargs):
        '''Propagate a TLE pair
        '''

        line1, line2 = SGP4.line_decode(line1), SGP4.line_decode(line2)

        satellite = Satrec.twoline2rv(line1, line2, self.grav_ind)

        epoch = Time(satellite.jdsatepoch + satellite.jdsatepochF, format='jd', scale='utc')
        if self.logger is not None:
            self.logger.debug(f'SGP4:propagate_tle:epoch={epoch}')

        t, epoch = self.convert_time(t, epoch)
        times = epoch + t

        jd_f = times.jd2
        jd0 = times.jd1

        logger_profiler_on = kwargs.get('logger_profiler_on', True)
        
        if self.profiler is not None:
            self.profiler.start('SGP4:propagate_tle:steps')

        if isinstance(jd_f, float) or isinstance(jd_f, int):
            states = np.empty((6,), dtype=np.float64)

            error, r, v = satellite.sgp4(jd0, jd_f)

            if logger_profiler_on:
                if error != 0 and self.logger is not None:
                    self.logger.error(f'SGP4:propagate:step-{ind}:{SGP4_ERRORS[error]}')

            states[:3] = r
            states[3:] = v
            errors = [error]
        else:
            states = np.empty((6,jd_f.size), dtype=np.float64)

            if self.settings['heartbeat']:
                errors = []
                for tind in range(jd_f.size):
                    error, r, v = satellite.sgp4(jd0[tind], jd_f[tind])
                    errors.append(error)
                    states[:3,tind] = r
                    states[3:,tind] = v
                    self.heartbeat(jd0[tind] + jd_f[tind], states[:,tind], satellite=satellite)
            else:
                errors, r, v = satellite.sgp4_array(jd0, jd_f)
                states[:3,...] = r.T
                states[3:,...] = v.T


        for ind, err in enumerate(errors):
            if err != 0 and self.logger is not None:
                self.logger.error(f'SGP4:propagate_tle:step-{ind}:{SGP4_ERRORS[err]}')

        states *= 1e3 #km to m, km/s to m/s

        states = frames.convert(
            times, 
            states, 
            in_frame='TEME',
            out_frame=self.settings['out_frame'],
            profiler = self.profiler,
            logger = self.logger,
        )

        if self.profiler is not None:
            self.profiler.stop('SGP4:propagate_tle:steps')

        return states


    def propagate(self, t, state0, epoch=None, **kwargs):
        '''Propagate a state

        #TODO: UPDATE THIS DOCSTRING

        All state-vector are given in SI units.

        Keyword arguments contain only information needed for ballistic coefficient :code:`B` used by SGP4. Either :code:`B` or :code:`C_D`, :code:`A` and :code:`m` must be supplied.
        They also contain a option to give angles in radians or degrees. By default input is assumed to be degrees.

        **Frame:**

        The input frame is ECI (TEME) for orbital elements and Cartesian. The output frame is as standard ECEF (ITRF). But can be set to TEME.

        :param float/list/numpy.ndarray/astropy.time.TimeDelta t: Time to propagate relative the initial state epoch.
        :param float/astropy.time.Time epoch: The epoch of the initial state.
        :param numpy.ndarray state0: 6-D Cartesian state vector in SI-units.
        :param float B: Ballistic coefficient
        :param float C_D: Drag coefficient
        :param float A: Cross-sectional Area
        :param float m: Mass
        :param bool radians: If true, all angles are assumed to be in radians.
        :param bool SGP4_mean_elements: If True, the input is not cartesian state but SGP4 mean elements.
        :return: 6-D Cartesian state vectors in SI-units.

        '''
        if self.profiler is not None:
            self.profiler.start('SGP4:propagate')
        if self.logger is not None:
            self.logger.debug(f'SGP4:propagate:len(t) = {len(t)}')

        if self.settings['tle_input']:
            if isinstance(state0, np.ndarray):
                if state0.size == 1:
                    state0 = state0[0]
            line1, line2 = state0
            states = self.propagate_tle(t, line1, line2, **kwargs)

        else:

            if epoch is None:
                raise ValueError('Need epoch when propagating state and not TLE')

            t, epoch = self.convert_time(t, epoch)

            epoch0 = epoch.mjd - self.sgp4_mjd0
            times = epoch + t

            if 'B' in kwargs:
                B = kwargs.pop('B')
            else:
                B = 0.5*kwargs.pop('C_D',2.3)*kwargs.pop('A',1.0)/kwargs.pop('m',1.0)

            if self.logger is not None:
                self.logger.debug(f'SGP4:propagate:B = {B}')

            input_mean = kwargs.get('SGP4_mean_elements', False)

            if input_mean:
                if self.settings['in_frame'] != 'TEME':
                    raise Exception(f'Cannot input mean elements in other frame than TEME (currently set to "{self.settings["in_frame"]}")')
                mean_elements = state0.copy()
                if not kwargs.get('radians', False):
                    mean_elements[2:,...] = np.radians(mean_elements[2:,...])

                mean_elements[0,...] *= 1e-3 #m to km

            else:
                if isinstance(state0, pyorb.Orbit):
                    state0_cart = np.squeeze(state0.cartesian)
                else:
                    state0_cart = state0

                state0_cart = frames.convert(
                    epoch, 
                    state0_cart, 
                    in_frame=self.settings['in_frame'], 
                    out_frame='TEME',
                    profiler = self.profiler,
                    logger = self.logger,
                )

                if state0_cart.size > 6:
                    t_samps = kwargs.get('state_sample_times')
                else:
                    t_samps = None

                mean_elements = self.TEME_to_TLE(state0_cart, t=t_samps, epoch=epoch, B=B, kepler=False)

                if np.any(np.isnan(mean_elements)):
                    raise Exception('Could not compute SGP4 initial state: {}'.format(mean_elements))

            states = self.propagate_mean_elements(times.jd1, times.jd2, mean_elements, epoch0, B, **kwargs)

            states = frames.convert(
                times, 
                states, 
                in_frame='TEME',
                out_frame=self.settings['out_frame'],
                profiler = self.profiler,
                logger = self.logger,
            )

        if self.profiler is not None:
            self.profiler.stop('SGP4:propagate')
        if self.logger is not None:
            self.logger.debug(f'SGP4:propagate:completed')

        return states

    
    def get_mean_elements(self, line1, line2, radians=False):
        '''Extract the mean elements in SI units (a [m], e [1], inc [deg], raan [deg], aop [deg], mu [deg]), B-parameter (not bstar) and epoch from a two line element pair.
        '''

        line1, line2 = SGP4.line_decode(line1), SGP4.line_decode(line2)

        xpdotp = 1440.0/(2.0*np.pi)  # 229.1831180523293

        satrec = Satrec.twoline2rv(line1, line2, self.grav_ind)

        B = satrec.bstar/(self.grav_model.radiusearthkm*1e3)*2/self.rho0

        epoch = Time(satrec.jdsatepoch + satrec.jdsatepochF, format='jd', scale='utc')

        mean_elements = np.zeros((6,), dtype=np.float64)

        n0 = satrec.no_kozai*xpdotp/(86400.0/(2*np.pi))

        mean_elements[0] = (np.sqrt(self.grav_model.mu)/n0)**(2.0/3.0)*1e3
        mean_elements[1] = satrec.ecco
        mean_elements[2] = satrec.inclo
        mean_elements[3] = satrec.nodeo
        mean_elements[4] = satrec.argpo
        mean_elements[5] = satrec.mo
        if not radians:
            mean_elements[2:] = np.degrees(mean_elements[2:])

        return mean_elements, B, epoch


    def propagate_mean_elements(self, jd0, jd_f, mean_elements, epoch0, B, **kwargs):
        '''Propagate sgp4 mean elements.
        '''

        # Compute ballistic coefficient
        bstar = 0.5*B*self.rho0 # B* in [1/m] using Density at q0[kg/m^3]
        n0 = np.sqrt(self.grav_model.mu) / ((mean_elements[0])**1.5)
        
        # Scaling
        n0 = n0*(86400.0/(2*np.pi)) # Convert to [rev/d]
        bstar = bstar*(self.grav_model.radiusearthkm*1e3)     # Convert from [1/m] to [1/R_EARTH]
    

        satellite = Satrec()
        satellite.sgp4init(
            self.grav_ind, # gravity model
            'i', # 'a' = old AFSPC mode, 'i' = improved mode
            int(kwargs.get('oid',42)), # satnum: Satellite number
            epoch0, # epoch: days since 1949 December 31 00:00 UT
            bstar, # bstar: drag coefficient (/earth radii)
            0.0, #[IGNORED BY SGP4] ndot: ballistic coefficient (revs/day) 
            0.0, #[IGNORED BY SGP4] nddot: second derivative of mean motion (revs/day^3)
            mean_elements[1], # ecco: eccentricity
            mean_elements[4], # argpo: argument of perigee (radians)
            mean_elements[2], # inclo: inclination (radians)
            mean_elements[5], # mo: mean anomaly (radians)
            n0/(1440.0/(2.0*np.pi)), # no_kozai: mean motion (radians/minute)
            mean_elements[3], # nodeo: right ascension of ascending node (radians)
        )

        logger_profiler_on = kwargs.get('logger_profiler_on', True)

        if self.profiler is not None and logger_profiler_on:
            self.profiler.start('SGP4:propagate:steps')

        if isinstance(jd_f, float) or isinstance(jd_f, int):
            states = np.empty((6,), dtype=np.float64)

            error, r, v = satellite.sgp4(jd0, jd_f)

            if logger_profiler_on:
                if error != 0 and self.logger is not None:
                    self.logger.error(f'SGP4:propagate:step:{SGP4_ERRORS[error]}')

            states[:3] = r
            states[3:] = v

        else:
            states = np.empty((6,jd_f.size), dtype=np.float64)

            if self.settings['heartbeat']:
                errors = []
                for tind in range(jd_f.size):
                    error, r, v = satellite.sgp4(jd0[tind], jd_f[tind])
                    errors.append(error)
                    states[:3,tind] = r
                    states[3:,tind] = v
                    self.heartbeat(jd0[tind] + jd_f[tind], states[:,tind], satellite=satellite)
            else:
                errors, r, v = satellite.sgp4_array(jd0, jd_f)
                states[:3,...] = r.T
                states[3:,...] = v.T

            if logger_profiler_on:
                for ind, err in enumerate(errors):
                    if err != 0 and self.logger is not None:
                        self.logger.error(f'SGP4:propagate:step-{ind}:{SGP4_ERRORS[err]}')


        states *= 1e3 #km to m and km/s to m/s

        if self.profiler is not None and logger_profiler_on:
            self.profiler.stop('SGP4:propagate:steps')

        return states


    def TEME_to_TLE_OPTIM(self, state, epoch, t=None, B=0.0, kepler=False, tol=1e-8, tol_v=1e-9):
        '''Convert osculating orbital elements in TEME
        to mean elements used in two line element sets (TLE's).

        :param numpy.ndarray kep: Osculating State (position and velocity) vector in m and m/s, TEME frame. If :code:`kepler = True` then state is osculating orbital elements, in m and radians. Orbital elements are semi major axis (m), orbital eccentricity, orbital inclination (radians), right ascension of ascending node (radians), argument of perigee (radians), mean anomaly (radians)
        :param bool kepler: Indicates if input state is kepler elements or cartesian.
        :param float epoch0: Epoch in days since 1949 December 31 00:00 UT
        :param float tol: Wanted precision in position of mean element conversion in m.
        :param float tol_v: Wanted precision in velocity mean element conversion in m/s.
        :return: mean elements of: semi major axis (km), orbital eccentricity, orbital inclination (radians), right ascension of ascending node (radians), argument of perigee (radians), mean anomaly (radians)
        :rtype: numpy.ndarray
        '''
        if self.profiler is not None:
            self.profiler.start('SGP4:TEME_to_TLE_OPTIM')
        if self.logger is not None:
            self.logger.info('SGP4:TEME_to_TLE_OPTIM')

        if len(state.shape) == 1:
            state.shape = (state.size, 1)

        if state.shape[1] > 1 and t is None:
            raise ValueError('Cannot convert TEME sampling to TLE without sample times "state_sample_times"')
        elif t is None:
            t = np.array([0.0])

        t_min = np.argmin(np.abs(t))
        if t[t_min] > 1e-6:
            raise ValueError('There is not sampling point at the epoch (t=0) to use as initial guess...')

        if kepler:
            state_cart = self._sgp4_elems2cart(state)
            init_elements = state[:, t_min]
        else:
            state_cart = state
            init_elements = self._cart2sgp4_elems(state_cart[:, t_min])

        tv = epoch + TimeDelta(t, format='sec')

        def find_mean_elems(mean_elements):
            # Mean elements and osculating state
            state_osc = self.propagate_mean_elements(
                tv.jd1, 
                tv.jd2, 
                mean_elements, 
                epoch.mjd - self.sgp4_mjd0, 
                B=B, 
                logger_profiler_on=False,
            )

            d = state_cart - state_osc
            return np.mean(np.linalg.norm(d, axis=0))

        opt_res = scipy.optimize.minimize(
            find_mean_elems, 
            init_elements,
            method='Nelder-Mead',
            options={'ftol': np.sqrt(tol**2 + tol_v**2)},
        )
        mean_elements = opt_res.x

        if self.profiler is not None:
            self.profiler.stop('SGP4:TEME_to_TLE_OPTIM')
        if self.logger is not None:
            self.logger.info(f'SGP4:TEME_to_TLE_OPTIM:completed')

        return mean_elements


    def TEME_to_TLE(self, state, epoch, t=None, B=0.0, kepler=False, tol=1e-5, tol_v=1e-7):
        '''Convert osculating orbital elements in TEME
        to mean elements used in two line element sets (TLE's).

        :param numpy.ndarray kep: Osculating State (position and velocity) vector in m and m/s, TEME frame. If :code:`kepler = True` then state is osculating orbital elements, in m and radians. Orbital elements are semi major axis (m), orbital eccentricity, orbital inclination (radians), right ascension of ascending node (radians), argument of perigee (radians), mean anomaly (radians)
        :param bool kepler: Indicates if input state is kepler elements or cartesian.
        :param astropy.time.Time epoch: Epoch of the orbit
        :param float tol: Wanted precision in position of mean element conversion in m.
        :param float tol_v: Wanted precision in velocity mean element conversion in m/s.
        :return: mean elements of: semi major axis (km), orbital eccentricity, orbital inclination (radians), right ascension of ascending node (radians), argument of perigee (radians), mean anomaly (radians)
        :rtype: numpy.ndarray
        '''
        if self.profiler is not None:
            self.profiler.start('SGP4:TEME_to_TLE')
        if self.logger is not None:
            self.logger.info('SGP4:TEME_to_TLE')

        mean_elements = None

        if len(state.shape) > 1:
            if state.size > 6:
                mean_elements = self.TEME_to_TLE_OPTIM(
                    state, 
                    epoch=epoch, 
                    t=t,
                    B=B, 
                    kepler=kepler, 
                    tol=tol, 
                    tol_v=tol_v,
                )

                if self.profiler is not None:
                    self.profiler.stop('SGP4:TEME_to_TLE')
                if self.logger is not None:
                    self.logger.info(f'SGP4:TEME_to_TLE:completed')

                return mean_elements
            else:
                state.shape = (state.size, )

        if kepler:
            state_mean = self._sgp4_elems2cart(state)
            state_cart = state_mean.copy()
        else:
            state_mean = state.copy()
            state_cart = state

        iter_max = self.settings['TEME_to_TLE_max_iter']  # Maximum number of iterations

        # Iterative determination of mean elements
        for it in range(iter_max):
            # Mean elements and osculating state
            mean_elements = self._cart2sgp4_elems(state_mean)

            if it > 0 and mean_elements[1] > 1:
                # Assumptions of osculation within slope not working, go to general minimization algorithms
                mean_elements = self.TEME_to_TLE_OPTIM(
                    state_cart, 
                    epoch=epoch, 
                    B=B, 
                    kepler=False, 
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
                logger_profiler_on=False,
            )

            # Correction of mean state vector
            d = state_cart - state_osc
            state_mean += d
            if it > 0:
                dr_old = dr
                dv_old = dv

            dr = np.linalg.norm(d[:3, ...], axis=0)  # Position change
            dv = np.linalg.norm(d[3:, ...], axis=0)  # Velocity change

            if it > 0:
                if dr_old < dr or dv_old < dv:
                    # Assumptions of osculation within slope not working, go to general minimization algorithms
                    mean_elements = self.TEME_to_TLE_OPTIM(
                        state_cart, 
                        epoch=epoch, 
                        B=B, 
                        kepler=False, 
                        tol=tol, 
                        tol_v=tol_v,
                    )
                    break

            if dr < tol and dv < tol_v:   # Iterate until position changes by less than eps
                break
            if it == iter_max - 1:
                # Iterative method not working, go to general minimization algorithms
                mean_elements = self.TEME_to_TLE_OPTIM(
                    state_cart, 
                    epoch=epoch, 
                    B=B, 
                    kepler=False, 
                    tol=tol, 
                    tol_v=tol_v,
                )

        if self.profiler is not None:
            self.profiler.stop('SGP4:TEME_to_TLE')
        if self.logger is not None:
            self.logger.info(f'SGP4:TEME_to_TLE:completed')

        return mean_elements


    def _sgp4_elems2cart(self, kep):
        '''Orbital elements to cartesian coordinates. Wrap pyorb-function to use mean anomaly, km and reversed order on aoe and raan. Output in SI.
        
        Neglecting mass is sufficient for this calculation (the standard gravitational parameter is 24 orders larger then the change).
        '''
        _kep = kep.copy()
        _kep[0, ...] *= 1e3
        tmp = _kep[4, ...]
        _kep[4, ...] = _kep[3, ...]
        _kep[3, ...] = tmp
        _kep[5, ...] = pyorb.mean_to_true(_kep[5, ...], _kep[1, ...], degrees=False)
        cart = pyorb.kep_to_cart(kep, mu=self.grav_model.mu*1e9, degrees=False)
        return cart

    def _cart2sgp4_elems(self, cart, degrees=False):
        '''Cartesian coordinates to orbital elements. Wrap pyorb-function to use mean anomaly, km and reversed order on aoe and raan.
        
        Neglecting mass is sufficient for this calculation (the standard gravitational parameter is 24 orders larger then the change).
        '''
        kep = pyorb.cart_to_kep(cart, mu=self.grav_model.mu*1e9, degrees=False)
        kep[0, ...] *= 1e-3
        tmp = kep[4, ...]
        kep[4, ...] = kep[3, ...]
        kep[3, ...] = tmp
        kep[5, ...] = pyorb.true_to_mean(kep[5, ...], kep[1, ...], degrees=False)
        return kep


