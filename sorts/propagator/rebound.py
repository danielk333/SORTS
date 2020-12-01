#!/usr/bin/env python

'''Wrapper for the REBOUND propagator into SORTS format.
'''
import copy
import pathlib

import numpy as np
import scipy

import pyorb

try:
    import rebound
    import spiceypy as spice
except ImportError:
    rebound = None
    spice = None

#Local import
from .base import Propagator
from .. import dates as dates
from .. import frames


class Rebound(Propagator):
    '''Propagator class implementing the REBOUND propagator.
    
    Frame options are found in the `sorts.frames.convert` function. 
    As the planetary/object positions not specified by adding objects or 
    calling propagate are added by calling SPICE. The common frame chosen is the 
    SPICE "ECLIPJ2000". This is equivalent to the Astropy HeliocentricMeanEcliptic frame.
    As such, internally in the Rebound simulation HeliocentricMeanEclipticJ2000 is always used.


    #TODO: add to this documentation
    '''

    MAIN_PLANETS = ['Mercury','Venus','Earth','Mars','Jupiter','Saturn','Uranus','Neptune']

    DEFAULT_SETTINGS = copy.copy(Propagator.DEFAULT_SETTINGS)
    DEFAULT_SETTINGS.update(
        dict(
            out_frame = 'HeliocentricMeanEcliptic',
            in_frame = 'HeliocentricMeanEcliptic',
            integrator = 'IAS15',
            time_step = 60.0,
            termination_check = None,
            termination_check_interval = 1,
            massive_objects = ['Moon'] + MAIN_PLANETS,
            use_sim_geocentric = True,
        )
    )
    
    def __init__(
            self, 
            spice_meta,
            settings=None, 
            **kwargs
        ):
        self.sim = None
        assert rebound is not None, 'Rebound python package not found'
        assert spice is not None, 'spiceypy package not found'

        super(Rebound, self).__init__(settings=settings, **kwargs)
        if self.logger is not None:
            self.logger.debug(f'sorts.propagator.Rebound:init')

        self.settings['massive_objects'] = [x.strip().capitalize() for x in self.settings['massive_objects']]

        spice_meta_path = pathlib.Path(spice_meta)
        assert spice_meta_path.is_file(), f'Could not find {spice_meta} Meta-kernel file.'
        spice.furnsh(str(spice_meta_path))

        self.planets_mass = {
            key:val for key,val in zip(
                ['Moon','Mercury','Venus','Earth','Mars','Jupiter','Saturn','Uranus','Neptune'],
                [7.34767309e22, 0.330104e24, 4.86732e24, 5.97219e24, 0.641693e24, 1898.13e24, 568.319e24, 86.8103e24, 102.410e24],
            )
        }

        self.m_sun = 1.98855e30
        self.spice_frame = 'ECLIPJ2000'
        self.internal_frame = 'HeliocentricMeanEcliptic'
        self.geo_internal_frame = 'GeocentricMeanEcliptic'
        self._earth_ind = self.settings['massive_objects'].index('Earth')
        self.N_massive = len(self.settings['massive_objects']) + 1;


    def __str__(self):
        from rebound import __version__, __build__
        s= ""
        s += "---------------------------------\n"
        s += "REBOUND version:     \t%s\n" %__version__
        s += "REBOUND built on:    \t%s\n" %__build__
        if self.sim is not None:
            s += "Number of particles: \t%d\n" %self.sim.N       
            s += "Selected integrator: \t" + self.sim.integrator + "\n"       
            s += "Simulation time:     \t%.16e\n" %self.sim.t
            s += "Current timestep:    \t%f\n" %self.sim.dt
        s += "---------------------------------"
        return s


    def _setup_sim(self, mjd0):
        self.sim = rebound.Simulation()
        self.sim.units = ('m', 's', 'kg')
        self.sim.integrator = self.settings['integrator']
        self._et = dates.mjd_to_j2000(mjd0)*3600.0*24.0
        
        self.sim.add(m=self.m_sun)
        for i in range(0,len(self.settings['massive_objects'])):
            if self.settings['massive_objects'][i] in self.MAIN_PLANETS:
                plt_str_ = self.settings['massive_objects'][i] + ' BARYCENTER'
            else:
                plt_str_ = self.settings['massive_objects'][i]

            #Units are always km and km/sec. 
            state, lightTime = spice.spkezr(
                plt_str_,
                self._et,
                self.spice_frame,
                'NONE',
                'SUN',
            )
            self.sim.add(m=self.planets_mass[self.settings['massive_objects'][i]],
                x=state[0]*1e3,  y=state[1]*1e3,  z=state[2]*1e3,
                vx=state[3]*1e3, vy=state[4]*1e3, vz=state[5]*1e3,
            )

        self.sim.N_active = self.N_massive
        self.sim.dt = self.settings['time_step']


    def _add_state(self, state, m):
        x, y, z, vx, vy, vz = state
        self.sim.add(
            x = x,
            y = y,
            z = z,
            vx = vx,
            vy = vy,
            vz = vz,
            m = m,
        )


    def _get_state(self, states, ti, n):
        particle = self.sim.particles[self.N_massive + n]
        states[0,ti,n] = particle.x
        states[1,ti,n] = particle.y
        states[2,ti,n] = particle.z
        states[3,ti,n] = particle.vx
        states[4,ti,n] = particle.vy
        states[5,ti,n] = particle.vz
        return states

    def _get_earth_state(self, earth_states, ti):
        earth = self.sim.particles[self._earth_ind]
        earth_states[0,ti] = earth.x
        earth_states[1,ti] = earth.y
        earth_states[2,ti] = earth.z
        earth_states[3,ti] = earth.vx
        earth_states[4,ti] = earth.vy
        earth_states[5,ti] = earth.vz
        return earth_states

    def propagate(self, t, state0, epoch, **kwargs):
        '''Propagate a state

        #TODO: add possible different equinox?
        #TODO: UPDATE THIS DOCSTRING
        '''

        if self.profiler is not None:
            self.profiler.start('Rebound:propagate')
        if self.logger is not None:
            self.logger.debug(f'Rebound:propagate:len(t) = {len(t)}')

        t, epoch = self.convert_time(t, epoch)
        times = epoch + t
        
        t_order = np.argsort(t.sec)
        t_restore = np.argsort(t_order)

        t = t[t_order]
        times = times[t_order]

        self._setup_sim(epoch.mjd)

        if isinstance(state0, pyorb.Orbit):
            state0_cart = np.squeeze(state0.cartesian)
        else:
            state0_cart = state0

        if len(state0_cart.shape) > 1:
            if state0_cart.shape[1] > 1:
                N_testparticle = state0_cart.shape[1]
            else:
                N_testparticle = 1
        else:
            N_testparticle = 1

        m = kwargs.get('m', np.zeros((N_testparticle,), dtype=np.float64))
        if isinstance(m, float) or isinstance(m, int):
            m = np.zeros((N_testparticle,), dtype=np.float64)*m

        if self.settings['use_sim_geocentric'] and frames.is_geocentric(self.settings['in_frame']):
            earth_state = np.zeros((6,1), dtype=np.float64)
            self._get_earth_state(earth_state, ti=0)
            earth_state.shape = (6,)

            state0_cart = frames.convert(
                epoch, 
                state0_cart, 
                in_frame = self.settings['in_frame'], 
                out_frame = self.geo_internal_frame,
                profiler = self.profiler,
                logger = self.logger,
            )
            if len(state0_cart.shape) > 1:
                state0_cart = state0_cart + earth_state[:,None]
            else:
                state0_cart = state0_cart + earth_state
        else:
            state0_cart = frames.convert(
                epoch, 
                state0_cart, 
                in_frame = self.settings['in_frame'], 
                out_frame = self.internal_frame,
                profiler = self.profiler,
                logger = self.logger,
            )

        if len(state0_cart.shape) > 1:
            for ni in range(N_testparticle):
                self._add_state(state0_cart[:,ni], m[ni])
        else:
           self._add_state(state0_cart, m[0])

        if self.settings['use_sim_geocentric']:
            self.earth_states = np.empty((6, len(t)), dtype=np.float64)
        else:
            self.earth_states = None

        states = np.empty((6, len(t), N_testparticle), dtype=np.float64)
        

        end_ind = len(t)

        for ti in range(len(t)):

            self.sim.integrate(t[ti].sec)

            for ni in range(N_testparticle):
                self._get_state(states, ti, ni)

            if self.settings['use_sim_geocentric']:
                self._get_earth_state(self.earth_states, ti)

            if self.settings['termination_check'] is not None and ti % self.settings['termination_check_interval'] == 0:
                termination_check = self.settings['termination_check']
                if termination_check(t[ti], states[:,ti,:]):
                    end_ind = ti+1
                    break

        t = t[0:end_ind]
        times = times[0:end_ind]
        t_restore = t_restore[0:end_ind]
        states = states[:, 0:end_ind, :]
        self.earth_states = self.earth_states[:, 0:end_ind]

        for ni in range(N_testparticle):
            if self.settings['use_sim_geocentric'] and frames.is_geocentric(self.settings['out_frame']):
                states[:,:,ni] = states[:,:,ni] - self.earth_states

                states[:,:,ni] = frames.convert(
                    times, 
                    states[:,:,ni], 
                    in_frame = self.geo_internal_frame,
                    out_frame = self.settings['out_frame'],
                    profiler = self.profiler,
                    logger = self.logger,
                )
            else:
                states[:,:,ni] = frames.convert(
                    times, 
                    states[:,:,ni], 
                    in_frame = self.internal_frame,
                    out_frame = self.settings['out_frame'],
                    profiler = self.profiler,
                    logger = self.logger,
                )

        if self.settings['use_sim_geocentric']:
            self.earth_states = self.earth_states[:, t_restore]
            if not frames.is_geocentric(self.settings['out_frame']):
                self.earth_states = frames.convert(
                    times, 
                    self.earth_states, 
                    in_frame = self.internal_frame,
                    out_frame = self.settings['out_frame'],
                    profiler = self.profiler,
                    logger = self.logger,
                )

        states = states[:, t_restore, :]
        if N_testparticle == 1:
            states.shape = states.shape[:2]

        return states



