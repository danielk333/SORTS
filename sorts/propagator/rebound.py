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

from astropy.time import Time

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
    As such, internally in the Rebound simulation inertial BarycentricMeanEclipticJ2000 is always used.

    GeocentricMeanEcliptic includes the Earth velocity, so any manual transformation to HeliocentricMeanEcliptic
    must also use the earth velocity.

    If the output frame is geocentric, the states input to the termination check is in GeocentricMeanEcliptic, else they are in HeliocentricMeanEcliptic.


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
            termination_check = False,
            termination_check_interval = 1,
            massive_objects = ['Moon'] + MAIN_PLANETS,
            save_massive_states = False,
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

        self.spice_meta = spice_meta

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


    def planet_index(self, name):
        return self.settings['massive_objects'].index(name)


    def _setup_sim(self, epoch, init_massive_states=None):
        spice_meta_path = pathlib.Path(self.spice_meta)
        if init_massive_states is None:
            assert spice_meta_path.is_file(), f'Could not find {spice_meta} Meta-kernel file.'
            spice.furnsh(str(spice_meta_path))
            self._et = spice.utc2et(epoch.utc.isot)

        self.sim = rebound.Simulation()
        self.sim.units = ('m', 's', 'kg')
        self.sim.integrator = self.settings['integrator']
        
        if init_massive_states is None:
            self.sim.add(m=self.m_sun)
        else:
            state = init_massive_states[:,0]
            self.sim.add(m=self.m_sun,
                x=state[0],  y=state[1],  z=state[2],
                vx=state[3], vy=state[4], vz=state[5],
            )

        for i in range(0,len(self.settings['massive_objects'])):
            if init_massive_states is None:
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
            else:
                state = init_massive_states[:,i+1]*1e-3

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


    def termination_check(self, t, step_index, massive_states, particle_states):
        raise NotImplementedError('Users need to implement this method to use termination checks')


    def _put_simulation_state(self, massive_states, particle_states, ti):
        if massive_states is not None:
            for p_n in range(self.N_massive):
                particle = self.sim.particles[p_n]
                massive_states[0,ti,p_n] = particle.x
                massive_states[1,ti,p_n] = particle.y
                massive_states[2,ti,p_n] = particle.z
                massive_states[3,ti,p_n] = particle.vx
                massive_states[4,ti,p_n] = particle.vy
                massive_states[5,ti,p_n] = particle.vz
        if particle_states is not None:
            for p_n in range(len(self.sim.particles) - self.N_massive):
                particle = self.sim.particles[self.N_massive + p_n]
                particle_states[0,ti,p_n] = particle.x
                particle_states[1,ti,p_n] = particle.y
                particle_states[2,ti,p_n] = particle.z
                particle_states[3,ti,p_n] = particle.vx
                particle_states[4,ti,p_n] = particle.vy
                particle_states[5,ti,p_n] = particle.vz
        return massive_states, particle_states


    def _get_helio_state(self):
        sun_state = np.zeros((6,), dtype=np.float64)
        sun = self.sim.particles[0]
        sun_state[0] = sun.x
        sun_state[1] = sun.y
        sun_state[2] = sun.z
        sun_state[3] = sun.vx
        sun_state[4] = sun.vy
        sun_state[5] = sun.vz
        return sun_state

    def _get_earth_state(self):
        earth_state = np.zeros((6,), dtype=np.float64)
        earth = self.sim.particles[self._earth_ind]
        earth_state[0] = earth.x
        earth_state[1] = earth.y
        earth_state[2] = earth.z
        earth_state[3] = earth.vx
        earth_state[4] = earth.vy
        earth_state[5] = earth.vz
        return earth_state


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

        if np.any(t.sec < 0):
            t_order = np.argsort(-t.sec)
        else:
            t_order = np.argsort(t.sec)

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

        if not (np.all(t.sec <= 0) or np.all(t.sec >= 0)):
            states = np.empty((6, len(t), N_testparticle), dtype=np.float64)
            if N_testparticle == 1:
                states.shape = states.shape[:2]

            ret_backward = self.propagate(t[t.sec < 0], state0, epoch, **kwargs)
            ret_forward = self.propagate(t[t.sec >= 0], state0, epoch, **kwargs)
            
            if self.settings['save_massive_states']:
                massive_states = np.empty((6, len(t), self.N_massive), dtype=np.float64)
                
                states_b, massive_b = ret_backward
                states_f, massive_f = ret_forward
                
                massive_states[:, t.sec < 0, :] = massive_b
                massive_states[:, t.sec >= 0, :] = massive_f
            else:
                states_b = ret_backward
                states_f = ret_forward

            states[:, t.sec < 0, ...] = states_b
            states[:, t.sec >= 0, ...] = states_f

            if self.settings['save_massive_states']:
                return states, massive_states
            else:
                return states

        if np.all(t.sec <= 0):
            backwards_integration = True
            t = -t
        else:
            backwards_integration = False


        t_restore = np.argsort(t_order)
        t = t[t_order]
        times = times[t_order]

        init_massive_states = kwargs.get('massive_states', None)

        self._setup_sim(epoch, init_massive_states=init_massive_states)


        m = kwargs.get('m', np.zeros((N_testparticle,), dtype=np.float64))
        if isinstance(m, float) or isinstance(m, int):
            m = np.zeros((N_testparticle,), dtype=np.float64)*m


        #DEBUG#
        #it is on the other side of the solar system?????
        earth_state = self._get_earth_state()

        state0_cart0 = frames.convert(
            epoch, 
            state0_cart, 
            in_frame = self.settings['in_frame'], 
            out_frame = self.geo_internal_frame,
            profiler = self.profiler,
            logger = self.logger,
        )

        if len(state0_cart.shape) > 1:
            state0_cart0 = state0_cart0 + earth_state[:,None]
        else:
            state0_cart0 = state0_cart0 + earth_state
        #else:
        state0_cart1 = frames.convert(
            epoch, 
            state0_cart, 
            in_frame = self.settings['in_frame'], 
            out_frame = self.internal_frame,
            profiler = self.profiler,
            logger = self.logger,
        )
        print(state0_cart)
        print(f'- {state0_cart0}')
        print(f'- {state0_cart1}')
        print(state0_cart0 - state0_cart1)
        print(state0_cart1 - earth_state)

        assert 0, 'DEBUG'
        #DEBUG#


        if frames.is_geocentric(self.settings['in_frame']):
            earth_state = self._get_earth_state()

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

        self.sim.move_to_com()

        #Fix for backwards integration, Rebound cannot handle negative times as it is trivial to reverse time
        if backwards_integration:
            for p_ in self.sim.particles:
                p_.vx = -p_.vx
                p_.vy = -p_.vy
                p_.vz = -p_.vz

        if self.settings['save_massive_states']:
            massive_states = np.empty((6, len(t), self.N_massive), dtype=np.float64)
        else:
            massive_states = None

        states = np.empty((6, len(t), N_testparticle), dtype=np.float64)
        end_ind = len(t)

        for ti in range(len(t)):

            self.sim.integrate(t[ti].sec)

            massive_states, states = self._put_simulation_state(massive_states, states, ti)

            if frames.is_geocentric(self.settings['out_frame']):
                earth_state = self._get_earth_state()
                states[:,ti,:] -= earth_state[:,None]
                if self.settings['save_massive_states']:
                    massive_states[:,ti,:] -= earth_state[:,None]
            else:
                sun_state = self._get_helio_state()
                states[:,ti,:] -= sun_state[:,None]
                if self.settings['save_massive_states']:
                    massive_states[:,ti,:] -= sun_state[:,None]

            if self.settings['termination_check'] and ti % self.settings['termination_check_interval'] == 0:
                if backwards_integration:
                    t__ = -t[ti]
                else:
                    t__ = t[ti]
                if self.termination_check(t__, ti, massive_states, states):
                    end_ind = ti+1
                    break

        if backwards_integration:
            states[3:,:,:] = -states[3:,:,:]
            t = -t

        t = t[0:end_ind]
        times = times[0:end_ind]
        t_restore = t_restore[0:end_ind]
        states = states[:, 0:end_ind, :]

        if frames.is_geocentric(self.settings['out_frame']):
            int_frame_ = self.geo_internal_frame
        else:
            int_frame_ = self.internal_frame

        for ni in range(N_testparticle):
            states[:,:,ni] = frames.convert(
                times, 
                states[:,:,ni], 
                in_frame = int_frame_,
                out_frame = self.settings['out_frame'],
                profiler = self.profiler,
                logger = self.logger,
            )

        states = states[:, t_restore, :]
        if N_testparticle == 1:
            states.shape = states.shape[:2]

        if self.settings['save_massive_states']:
            if backwards_integration:
                massive_states[3:,:,:] = -massive_states[3:,:,:]
            massive_states = massive_states[:, 0:end_ind, :]

            for ni in range(self.N_massive):
                massive_states[:,:,ni] = frames.convert(
                    times, 
                    massive_states[:,:,ni], 
                    in_frame = int_frame_,
                    out_frame = self.settings['out_frame'],
                    profiler = self.profiler,
                    logger = self.logger,
                )

            massive_states = massive_states[:, t_restore, :]

            return states, massive_states
        else:
            return states



