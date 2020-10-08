import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import unittest
import numpy as np
import numpy.testing as nt
import scipy.constants as consts

from sgp4.io import twoline2rv
import sgp4

import sorts
from sorts.propagator import SGP4
from sorts import frames

from astropy.time import Time

R_e = 6.3781e6

def print_np(vec):
    st = '{:.8f}, '*vec.size
    vec = vec.flatten()
    print(st.format(*vec.tolist()))

class TestPropagatorSGP4(unittest.TestCase):

    def setUp(self):
        self.epoch0 = Time(2457126.2729, format='jd', scale='utc')
        self.params = dict(
            C_D = 2.3,
            m = 8000,
            A = 1.0,
        )
        self.prop = SGP4()

    def test_SGP4_propagate(self):

        t = np.arange(0,24*360, dtype=np.float)*10.0

        ecefs = self.prop.get_orbit()

        assert ecefs.shape == (6, t.size)
        assert isinstance(ecefs, np.ndarray)

        ecefs = self.prop.get_orbit()

        assert ecef.shape == (6, 1)
        assert isinstance(ecef, np.ndarray)

    def test_SGP4_propagate_B(self):

        t = np.arange(0,24*360, dtype=np.float)*10.0

        ecefs_B = self.prop.get_orbit()

        assert ecefs.shape == (6, t.size)
        assert isinstance(ecefs, np.ndarray)

    def test_SGP4_settings(self):
        assert 0



    def test_propagator_sgp4_mjd_invaraiance(self):
        #make sure that initial MJD does NOT matter when the frame is kept as TEME

        # Assemble all data that is required for SGP4 propagation
        # Epoch, mean orbital elements, ballistic coefficient

        mjd0 = 54729.51782528
            
        i0    = np.radians(51.6416)
        e0    = 0.0006703
        M0    = np.radians(325.0288)
        raan0 = np.radians(247.4627)
        aop0  = np.radians(130.5360)

        # Convert the unit of the ballistic coefficient to [1/m] 
        bstar =  -0.11606E-4 / (SGP4_module_wrapper.R_EARTH*1.0e3)
        # Ballistic coefficient, B [m^2/kg] 
        B = 2.0 * bstar / SGP4_module_wrapper.RHO0
        
        # Mean motion [1/s]
        n0    = 15.72125391 * (2*np.pi) / 86400.0
        a0    = (np.sqrt(SGP4_module_wrapper.GM) / (n0))**(2.0/3.0)
        
        mean_elements = [a0,e0,i0,raan0,aop0,M0]

        for di, dt in enumerate(np.linspace(0,365*10,num=400,dtype=np.float)):

            state = sgp4_propagation(mjd0 + dt, mean_elements, B=0, dt=0.0)
            if di == 0:
                state_prev = state
            else:
                nt.assert_almost_equal(state, state_prev, decimal=9)




class SentinelTestSGP4(unittest.TestCase):

    def setUp(self):
        self.prop = SGP4(settings=dict(out_frame='ITRF'))

    def test_tg_cart0(self):
        '''
        See if cartesian orbit interface recovers starting state
        '''
        
        # Statevector from Sentinel-1 precise orbit (in ECEF frame)
        sv = np.array([('2015-04-30T05:45:44.000000000',
            [2721793.785377, 1103261.736653, 6427506.515945],
            [ 6996.001258,  -171.659563, -2926.43233 ])],
              dtype=[('utc', '<M8[ns]'), ('pos', '<f8', (3,)), ('vel', '<f8', (3,))])
        pos = np.array([float(i) for i in sv[0]['pos']])         # m
        vel = np.array([float(i) for i in sv[0]['vel']])         # m/s
        mjd0 = (sv[0]['utc'] - np.datetime64('1858-11-17')) / np.timedelta64(1, 'D')

        pos.shape = 3,1
        vel.shape = 3,1

        t = [0]
        # ecef2teme works in km and km/s inn and out.
        teme = frames.ECEF_to_TEME(np.array([0.0]), pos*1e-3, vel*1e-3, mjd0=mjd0) * 1e3
        x, y, z, vx, vy, vz = teme.T[0]

        pv = self.prop.get_orbit_cart([0.0], x, y, z, vx, vy, vz, mjd0,
                              m=2300., C_R=1., C_D=2.3, A=4*2.3)

        print('pos error:')
        dp = sv['pos'] - pv[:3].T
        print('{:.5e}, {:.5e}, {:.5e}'.format(*dp[0].tolist()))

        print('vel error:')
        dv = sv['vel'] - pv[3:].T
        print('{:.5e}, {:.5e}, {:.5e}'.format(*dv[0].tolist()))

        nt.assert_array_almost_equal(sv['pos'] / pv[:3].T, np.ones((1,3)), decimal=7)
        nt.assert_array_almost_equal(sv['vel'] / pv[3:].T, np.ones((1,3)), decimal=7)

    def test_tg_cart6(self):
        '''
        See if cartesian orbit propagation interface matches actual orbit
        '''

        # Statevector from Sentinel-1 precise orbit (in ECEF frame)
        sv = np.array([
            ('2015-04-30T05:45:44.000000000',
                [2721793.785377, 1103261.736653, 6427506.515945],
                [ 6996.001258,  -171.659563, -2926.43233 ]),
            ('2015-04-30T05:45:54.000000000',
                [2791598.832403, 1101432.471307, 6397880.289842],
                [ 6964.872299,  -194.182612, -2998.757484]),
            ('2015-04-30T05:46:04.000000000',
                [2861088.520266, 1099378.309568, 6367532.487662],
                [ 6932.930021,  -216.638226, -3070.746198]),
            ('2015-04-30T05:46:14.000000000',
                [2930254.733863, 1097099.944255, 6336466.514344], 
                [ 6900.178053,  -239.022713, -3142.39037 ]), 
            ('2015-04-30T05:46:24.000000000',
                [2999089.394834, 1094598.105058, 6304685.855646],
                [ 6866.620117,  -261.332391, -3213.681933]),
            ('2015-04-30T05:46:34.000000000', 
                [3067584.462515, 1091873.55841 , 6272194.077798], 
                [ 6832.260032,  -283.563593, -3284.612861])],
            dtype=[('utc', '<M8[ns]'), ('pos', '<f8', (3,)), ('vel', '<f8', (3,))])

        pos = np.array([float(i) for i in sv[0]['pos']])         # m
        vel = np.array([float(i) for i in sv[0]['vel']])         # m/s
        mjd0 = (sv[0]['utc'] - np.datetime64('1858-11-17')) / np.timedelta64(1, 'D')

        pos.shape = 3,1
        vel.shape = 3,1

        t = 10*np.arange(6)
        teme = frames.ECEF_to_TEME(np.array([0.0]), pos*1e-3, vel*1e-3, mjd0=mjd0) * 1e3

        pv = self.prop.propagate(t, teme[:,0], mjd0,
                              m=2300., C_R=1., C_D=2.3, A=4*2.3)

        print('pos error:')
        dp = sv['pos'] - pv[:3].T
        print('{:.5e}, {:.5e}, {:.5e}'.format(*dp[0].tolist()))

        print('vel error:')
        dv = sv['vel'] - pv[3:].T
        print('{:.5e}, {:.5e}, {:.5e}'.format(*dv[0].tolist()))

        nt.assert_array_almost_equal(sv['pos'] / pv[:3].T, np.ones((6,3)), decimal=4)
        nt.assert_array_almost_equal(sv['vel'] / pv[3:].T, np.ones((6,3)), decimal=5)   # FAILS
        # nt.assert_array_less(np.abs(sv['pos'] - pv[:3].T), np.full((6,3), 1.0, dtype=pv.dtype)) #m
        # nt.assert_array_less(np.abs(sv['vel'] - pv[3:].T), np.full((6,3), 1.0e-3, dtype=pv.dtype)) #mm/s

