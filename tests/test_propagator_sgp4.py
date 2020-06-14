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
from sorts.propagator.pysgp4 import sgp4_propagation
from sorts.propagator.pysgp4 import SGP4_module_wrapper
from sorts import frames

R_e = 6.3781e6

def print_np(vec):
    st = '{:.8f}, '*vec.size
    vec = vec.flatten()
    print(st.format(*vec.tolist()))

class TestPropagatorSGP4(unittest.TestCase):

    def setUp(self):
        self.mjd0 = sorts.dates.jd_to_mjd(2457126.2729)         # 2015-04-13T18:32:58.560019
        self.C_D = 2.3
        self.m = 8000
        self.A = 1.0
        self.prop = SGP4()

    def test_class_implementation(self):
        
        # Simple test with ISS example from Wikipedia
        # https://en.wikipedia.org/wiki/Two-line_element_set

        dt = 600.0 # 10 minutes
        
        # Using the sgp4 module directly
        
        # ISS (ZARYA)
        line1 = '1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927'
        line2 = '2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537'
        
        sat = twoline2rv(line1, line2, sgp4.earth_gravity.wgs72)
        
        # Assemble all data that is required for SGP4 propagation
        # Epoch, mean orbital elements, ballistic coefficient

        mjd0 = 54729.51782528
        
        mjd_ = mjd0 + dt / 86400.0 
            
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

        # Create own SGP4 object
        
        obj = SGP4_module_wrapper(mjd0, mean_elements, B)
        
        print('Compare satellite structs')
        print('-------------------------')

        for v in vars(sat):
            if (type(getattr(sat, v)) == float):
                if v in vars(obj.sat):
                    nt.assert_almost_equal(getattr(obj.sat, v), getattr(sat, v), decimal=9)
                

        # Evaluate function at epoch
        pos, vel = sgp4.propagation.sgp4(sat, dt/60.0)
        pos = np.array(pos)
        vel = np.array(vel)
        
        y1 = obj.state(mjd_)
        
        y2 = sgp4_propagation(mjd0, mean_elements, B, dt)
        
        nt.assert_array_almost_equal(pos/R_e, y1[0:3]/R_e, decimal=6)
        nt.assert_array_almost_equal(vel*1e-3, y1[3:7]*1e-3, decimal=6)

        nt.assert_array_almost_equal(pos/R_e, y2[0:3]/R_e, decimal=6)
        nt.assert_array_almost_equal(vel*1e-3, y2[3:7]*1e-3, decimal=6)

        nt.assert_array_almost_equal(y2[0:3]/R_e, y1[0:3]/R_e, decimal=6)
        nt.assert_array_almost_equal(y2[3:7]*1e-3, y1[3:7]*1e-3, decimal=6)


    def test_PropagatorSGP4_get_orbit(self):

        t = np.arange(0,24*360, dtype=np.float)*10.0

        ecefs = self.prop.get_orbit(
            t=t, mjd0=self.mjd0,
            a=7000e3, e=0.0, inc=90.0, 
            raan=10, aop=10, mu0=40.0, 
            C_D=self.C_D, m=self.m, A=self.A,
        )

        assert ecefs.shape == (6, t.size)
        assert isinstance(ecefs, np.ndarray)

        ecef = self.prop.get_orbit(
            t=t[0], mjd0=self.mjd0,
            a=7000e3, e=0.0, inc=90.0, 
            raan=10, aop=10, mu0=40.0, 
            C_D=self.C_D, m=self.m, A=self.A,
        )

        assert ecef.shape == (6, 1)
        assert isinstance(ecef, np.ndarray)

    def test_PropagatorSGP4_get_orbit_B(self):

        t = np.arange(0,24*360, dtype=np.float)*10.0

        ecefs = self.prop.get_orbit(
            t=t, mjd0=self.mjd0,
            a=7000e3, e=0.0, inc=90.0, 
            raan=10, aop=10, mu0=40.0, 
            B=0.5*self.C_D*self.A/self.m,
        )

        assert ecefs.shape == (6, t.size)
        assert isinstance(ecefs, np.ndarray)


    def test_ecef_teme_inverse(self):
        ecef = np.array([-1.531e+03, -4.235e+03,  5.353e+03,  1.656e+00,  5.549e+00, 4.853e+00], dtype=np.float)*1e3
        p = ecef[:3]*1e-3
        p.shape=(3,1)
        v = ecef[3:]*1e-3
        v.shape=(3,1)
        teme = frames.ECEF_to_TEME(np.array([0.0]), p, v)*1e3
        ecef_ref = frames.TEME_to_ECEF(np.array([0.0]), teme[:3,:1]*1e-3, teme[3:,:1]*1e-3)*1e3

        nt.assert_array_almost_equal(ecef_ref[:3,0]/R_e, ecef[:3]/R_e, decimal=3)
        nt.assert_array_almost_equal(ecef_ref[3:,0]*1e-3, ecef[3:]*1e-3, decimal=6)

    def test_PropagatorSGP4_get_orbit_cart(self):

        test = False
        try: 
            ecefs_kep = self.prop.get_orbit_cart(
                t=t, mjd0=self.mjd0,
                x=1, y=0.1, z=90.0, 
                vx=10, vy=10, vz=40.0, 
                C_D=self.C_D, m=self.m, A=self.A,
            )
        except:
            test = True

        assert test

    def test_PropagatorSGP4_polar_motion(self):

        prop = SGP4(settings=dict(polar_motion=True))

        t = np.arange(0,24*360, dtype=np.float)*10.0

        ecefs = self.prop.get_orbit(
            t=t, mjd0=self.mjd0,
            a=7000e3, e=0.0, inc=90.0, 
            raan=10, aop=10, mu0=40.0, 
            C_D=self.C_D, m=self.m, A=self.A,
        )

        assert ecefs.shape == (6, t.size)
        assert isinstance(ecefs, np.ndarray)

    def test_PropagatorSGP4_polar_motion00(self):

        prop = SGP4(settings=dict(polar_motion=True, polar_motion_model='00'))

        t = np.arange(0,24*360, dtype=np.float)*10.0

        ecefs = self.prop.get_orbit(
            t=t, mjd0=self.mjd0,
            a=7000e3, e=0.0, inc=90.0, 
            raan=10, aop=10, mu0=40.0, 
            C_D=self.C_D, m=self.m, A=self.A,
        )

        assert ecefs.shape == (6, t.size)
        assert isinstance(ecefs, np.ndarray)

    def test_PropagatorSGP4_cart(self):

        prop = SGP4()

        t = np.arange(0,24*360, dtype=np.float)*10.0

        ecefs = self.prop.get_orbit_cart(
            t=t, mjd0=self.mjd0,
            x=-5339.76186573000e3, y=5721.43584226500e3, z=921.276953805000e3, 
            vx=-4.88969089550000e3, vy=-3.83304653050000e3, vz=3.18013811100000e3, 
            C_D=self.C_D, m=self.m, A=self.A,
        )

        assert ecefs.shape == (6, t.size)
        assert isinstance(ecefs, np.ndarray)

    def test_PropagatorSGP4_cart_polar_motion00(self):

        prop = SGP4(settings=dict(polar_motion=True, polar_motion_model='00'))

        t = np.arange(0,24*360, dtype=np.float)*10.0

        ecefs = self.prop.get_orbit_cart(
            t=t, mjd0=self.mjd0,
            x=-5339.76186573000e3, y=5721.43584226500e3, z=921.276953805000e3, 
            vx=-4.88969089550000e3, vy=-3.83304653050000e3, vz=3.18013811100000e3, 
            C_D=self.C_D, m=self.m, A=self.A,
        )

        assert ecefs.shape == (6, t.size)
        assert isinstance(ecefs, np.ndarray)


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


    def test_PropagatorSGP4_cart_kep_inverse(self):

            prop = SGP4(settings=dict(polar_motion=False))

            t = np.array([0], dtype=np.float)

            reci = np.array([
                    -5339.76186573000,
                    5721.43584226500,
                    921.276953805000,
                ], dtype=np.float)

            veci = np.array([
                    -4.88969089550000,
                    -3.83304653050000,
                    3.18013811100000,
                ], dtype=np.float)

            state_TEME = np.concatenate((reci*1e3, veci*1e3), axis=0)

            M_earth = SGP4_module_wrapper.GM*1e9/consts.G

            kep = dpt.cart2kep(state_TEME, m=self.m, M_cent=M_earth, radians=False)
            kep[5] = dpt.true2mean(kep[5], kep[1], radians=False)

            ecefs_kep = prop.get_orbit(
                t=t, mjd0=self.mjd0,
                a=kep[0], e=kep[1], inc=kep[2],
                raan=kep[4], aop=kep[3], mu0=kep[5],
                C_D=self.C_D, m=self.m, A=self.A,
            )
            p = ecefs_kep[:3]*1e-3
            p.shape=(3,1)
            v = ecefs_kep[3:]*1e-3
            v.shape=(3,1)
            kep_TEME = frames.ECEF_to_TEME(np.array([0.0]), p, v, mjd0=self.mjd0)*1e3
            kep_TEME.shape = (6,)

            state_diff1 = np.abs(kep_TEME - state_TEME)

            nt.assert_array_less(state_diff1[:3], np.full((3,), 1e-2, dtype=state_diff1.dtype))
            nt.assert_array_less(state_diff1[3:], np.full((3,), 1e-5, dtype=state_diff1.dtype))

            ecefs_cart = prop.get_orbit_cart(
                t=t, mjd0=self.mjd0,
                x=-5339.76186573000e3, y=5721.43584226500e3, z=921.276953805000e3, 
                vx=-4.88969089550000e3, vy=-3.83304653050000e3, vz=3.18013811100000e3, 
                C_D=self.C_D, m=self.m, A=self.A,
            )
            p = ecefs_cart[:3]*1e-3
            p.shape=(3,1)
            v = ecefs_cart[3:]*1e-3
            v.shape=(3,1)
            cart_TEME = frames.ECEF_to_TEME(np.array([0.0]), p, v, mjd0=self.mjd0)*1e3
            cart_TEME.shape = (6,)

            state_diff2 = np.abs(cart_TEME - state_TEME)

            nt.assert_array_less(state_diff2[:3], np.full((3,), 1e-2, dtype=state_diff2.dtype))
            nt.assert_array_less(state_diff2[3:], np.full((3,), 1e-5, dtype=state_diff2.dtype))

            state_diff = np.abs(cart_TEME - kep_TEME)

            nt.assert_array_less(state_diff[:3], np.full((3,), 1e-2, dtype=state_diff.dtype))
            nt.assert_array_less(state_diff[3:], np.full((3,), 1e-5, dtype=state_diff.dtype))


    def test_PropagatorSGP4_cart_kep_cases(self):

        R_E = 6353.0e3
        mjd0 = dpt.jd_to_mjd(2457126.2729)
        a = R_E*2.0
        orb_init_list = []

        orb_range = np.array([a, 0.9, 180, 360, 360, 360], dtype=np.float)
        orb_offset = np.array([R_E*1.1, 0, 0.0, 0.0, 0.0, 0.0], dtype=np.float)
        test_n = 100
        while len(orb_init_list) < test_n:
            orb = np.random.rand(6)
            orb = orb_offset + orb*orb_range
            if orb[0]*(1.0 - orb[1]) > R_E+200e3:
                orb_init_list.append(orb)

        orb_init_list.append(np.array([R_E*1.2, 0, 0.0, 0.0, 0.0, 0.0], dtype=np.float))
        orb_init_list.append(np.array([R_E*1.2, 0, 0.0, 0.0, 0.0, 270], dtype=np.float))
        orb_init_list.append(np.array([R_E*1.2, 1e-9, 0.0, 0.0, 0.0, 0.0], dtype=np.float))
        orb_init_list.append(np.array([R_E*1.2, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float))
        orb_init_list.append(np.array([R_E*1.2, 0.1, 75.0, 0.0, 0.0, 0.0], dtype=np.float))
        orb_init_list.append(np.array([R_E*1.2, 0.1, 0.0, 120.0, 0.0, 0.0], dtype=np.float))
        orb_init_list.append(np.array([R_E*1.2, 0.1, 0.0, 0.0, 35.0, 0.0], dtype=np.float))
        orb_init_list.append(np.array([R_E*1.2, 0.1, 75.0, 120.0, 0.0, 0.0], dtype=np.float))
        orb_init_list.append(np.array([R_E*1.2, 0.1, 75.0, 0.0, 35.0, 0.0], dtype=np.float))
        orb_init_list.append(np.array([R_E*1.2, 0.1, 75.0, 120.0, 35.0, 0.0], dtype=np.float))

        prop = SGP4(settings=dict(polar_motion=False))

        t = np.linspace(0, 12*3600, num=100, dtype=np.float)
        M_earth = SGP4_module_wrapper.GM*1e9/consts.G
        
        for kep in orb_init_list:
            state_TEME = dpt.kep2cart(kep, m=self.m, M_cent=M_earth, radians=False)

            ecefs_kep = prop.get_orbit(
                t=t, mjd0=mjd0,
                a=kep[0], e=kep[1], inc=kep[2],
                raan=kep[4], aop=kep[3], mu0=dpt.true2mean(kep[5], kep[1], radians=False),
                C_D=self.C_D, m=self.m, A=self.A,
                radians=False,
            )
            ecefs_cart = prop.get_orbit_cart(
                t=t, mjd0=mjd0,
                x=state_TEME[0], y=state_TEME[1], z=state_TEME[2],
                vx=state_TEME[3], vy=state_TEME[4], vz=state_TEME[5],
                C_D=self.C_D, m=self.m, A=self.A,
            )

            p = ecefs_kep[:3,:]*1e-3
            v = ecefs_kep[3:,:]*1e-3
            kep_TEME = frames.ECEF_to_TEME(t, p, v, mjd0=mjd0)*1e3

            p = ecefs_kep[:3,:]*1e-3
            v = ecefs_kep[3:,:]*1e-3
            cart_TEME = frames.ECEF_to_TEME(t, p, v, mjd0=mjd0)*1e3

            state_diff1 = np.abs(kep_TEME - cart_TEME)

            nt.assert_array_less(state_diff1[:3,:], np.full((3,t.size), 1e-5, dtype=state_diff1.dtype))
            nt.assert_array_less(state_diff1[3:,:], np.full((3,t.size), 1e-7, dtype=state_diff1.dtype))


    def test_PropagatorSGP4_cart_kep_inverse_cases(self):

        R_E = 6353.0e3
        mjd0 = dpt.jd_to_mjd(2457126.2729)
        a = R_E*2.0
        orb_init_list = []

        orb_range = np.array([a, 0.9, 180, 360, 360, 360], dtype=np.float)
        orb_offset = np.array([R_E*1.1, 0, 0.0, 0.0, 0.0, 0.0], dtype=np.float)
        test_n = 5000
        while len(orb_init_list) < test_n:
            orb = np.random.rand(6)
            orb = orb_offset + orb*orb_range
            if orb[0]*(1.0 - orb[1]) > R_E+200e3:
                orb_init_list.append(orb)

        orb_init_list.append(np.array([R_E*1.2, 0, 0.0, 0.0, 0.0, 0.0], dtype=np.float))
        orb_init_list.append(np.array([R_E*1.2, 0, 0.0, 0.0, 0.0, 270], dtype=np.float))
        orb_init_list.append(np.array([R_E*1.2, 1e-9, 0.0, 0.0, 0.0, 0.0], dtype=np.float))
        orb_init_list.append(np.array([R_E*1.2, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float))
        orb_init_list.append(np.array([R_E*1.2, 0.1, 75.0, 0.0, 0.0, 0.0], dtype=np.float))
        orb_init_list.append(np.array([R_E*1.2, 0.1, 0.0, 120.0, 0.0, 0.0], dtype=np.float))
        orb_init_list.append(np.array([R_E*1.2, 0.1, 0.0, 0.0, 35.0, 0.0], dtype=np.float))
        orb_init_list.append(np.array([R_E*1.2, 0.1, 75.0, 120.0, 0.0, 0.0], dtype=np.float))
        orb_init_list.append(np.array([R_E*1.2, 0.1, 75.0, 0.0, 35.0, 0.0], dtype=np.float))
        orb_init_list.append(np.array([R_E*1.2, 0.1, 75.0, 120.0, 35.0, 0.0], dtype=np.float))

        prop = SGP4(settings=dict(polar_motion=False))

        t = np.array([0], dtype=np.float)
        M_earth = SGP4_module_wrapper.GM*1e9/consts.G

        fail_inds = []
        errs = np.empty((len(orb_init_list),2), dtype=np.float)
        for ind, kep in enumerate(orb_init_list):

            state_TEME = dpt.kep2cart(kep, m=self.m, M_cent=M_earth, radians=False)

            ecefs_kep = self.prop.get_orbit(
                t=t, mjd0=mjd0,
                a=kep[0], e=kep[1], inc=kep[2],
                raan=kep[4], aop=kep[3], mu0=dpt.true2mean(kep[5],kep[1],radians=False),
                C_D=self.C_D, m=self.m, A=self.A,
            )
            p = ecefs_kep[:3]*1e-3
            p.shape=(3,1)
            v = ecefs_kep[3:]*1e-3
            v.shape=(3,1)
            kep_TEME = frames.ECEF_to_TEME(np.array([0.0]), p, v, mjd0=mjd0)*1e3
            kep_TEME.shape = (6,)

            ecefs_cart = self.prop.get_orbit_cart(
                t=t, mjd0=mjd0,
                x=kep_TEME[0], y=kep_TEME[1], z=kep_TEME[2], 
                vx=kep_TEME[3], vy=kep_TEME[4], vz=kep_TEME[5], 
                C_D=self.C_D, m=self.m, A=self.A,
            )
            p = ecefs_cart[:3]*1e-3
            p.shape=(3,1)
            v = ecefs_cart[3:]*1e-3
            v.shape=(3,1)
            cart_TEME = frames.ECEF_to_TEME(np.array([0.0]), p, v, mjd0=mjd0)*1e3
            cart_TEME.shape = (6,)

            state_diff1 = np.abs(kep_TEME - state_TEME)
            try:
                nt.assert_array_less(state_diff1[:3], np.full((3,), 1e-2, dtype=state_diff1.dtype))
                nt.assert_array_less(state_diff1[3:], np.full((3,), 1e-5, dtype=state_diff1.dtype))
            except AssertionError as err:
                if ind not in fail_inds:
                    fail_inds.append(ind)

            state_diff2 = np.abs(cart_TEME - state_TEME)

            try:
                nt.assert_array_less(state_diff2[:3], np.full((3,), 1e-2, dtype=state_diff2.dtype))
                nt.assert_array_less(state_diff2[3:], np.full((3,), 1e-5, dtype=state_diff2.dtype))
            except AssertionError as err:
                if ind not in fail_inds:
                    fail_inds.append(ind)

            state_diff = np.abs(cart_TEME - kep_TEME)

            try:
                nt.assert_array_less(state_diff[:3], np.full((3,), 1e-2, dtype=state_diff.dtype))
                nt.assert_array_less(state_diff[3:], np.full((3,), 1e-5, dtype=state_diff.dtype))
            except AssertionError as err:
                if ind not in fail_inds:
                    fail_inds.append(ind)

            er_r = np.linalg.norm(state_diff1[:3])
            er_v = np.linalg.norm(state_diff1[3:])
            errs[ind,0] = er_r
            errs[ind,1] = er_v

        if len(fail_inds) > 0:
            print('FAIL / TOTAL: {} / {}'.format(len(fail_inds), len(orb_init_list)))

        assert np.median(errs[:,0]) < 1e-2
        assert np.median(errs[:,1]) < 1e-4

        assert len(fail_inds) < float(test_n)/100.


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

