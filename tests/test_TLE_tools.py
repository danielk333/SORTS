import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import time

import unittest
import numpy as n
import numpy.testing as nt
import scipy
import scipy.constants as consts
import TLE_tools as tle
import dpt_tools as dpt

class TestTLE(unittest.TestCase):

    def setUp(self):
        self.line1 = '1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927'
        self.line2 = '2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537'

    def test_TLE_to_TEME(self):

        state, epoch = tle.TLE_to_TEME(self.line1, self.line2)
        yy, mm, dd = dpt.jd_to_date(epoch)

        nt.assert_almost_equal(yy, 2008, decimal=8)
        nt.assert_almost_equal(mm, 9, decimal=8)
        nt.assert_almost_equal(dd, 20.51782528, decimal=8)
        assert state.shape==(6,)

    def test_tle_id(self):
        assert tle.tle_id(self.line1) == '25544'

    def test_tle_jd(self):
        jd = tle.tle_jd(self.line1)
        jd0 = dpt.date_to_jd(2008, 9, 20.51782528)
        nt.assert_almost_equal(jd, jd0, decimal = 8)

    def test_tle_bstar(self):
        nt.assert_almost_equal(tle.tle_bstar(self.line1), -11606.0e-4, decimal=8)

    def test_get_IERS_EOP(self):

        data, header = tle.get_IERS_EOP()

        assert isinstance(data, n.ndarray)
        assert data.shape[1] == len(header)

    def test_get_DUT(self):
        yy = [1,2,3,4]
        jds = []
        for y in yy:
            jds.append(dpt.date_to_jd(2008 + y, 9, 20.51782528))
        jds = n.array(jds)

        data = tle.get_DUT(jds)
        assert data.shape==(len(yy),1)

        #DUT always less then 1 sec
        nt.assert_array_less(n.abs(data), 1.0)

        data = tle.get_DUT(jds[0])
        assert data.shape==(1,1)

    def test_get_Polar_Motion(self):
        days = [1,2,3,4]
        jds = []
        for day in days:
            jds.append(dpt.date_to_jd(2008, 9, day+20.51782528))
        jds = n.array(jds)

        data = tle.get_Polar_Motion(jds)
        assert data.shape==(len(days),2)

        data = tle.get_Polar_Motion(jds[0])
        assert data.shape==(1,2)

    def test_TEME_to_ITRF_equatorial_circle(self):
        states = n.zeros((6,100),dtype=n.float)
        states[0,:] = 1.0
        day_f = n.linspace(0,1,endpoint=True,num=100,dtype=n.float)
        jds = n.empty(day_f.shape, dtype=n.float)
        for ind in range(len(day_f)):
            jds[ind] = dpt.date_to_jd(2008, 9, day_f[ind]+20.0)

        #this should produce a circle
        ITRF = tle.TEME_to_ITRF(states, jds[0], xp=0, yp=0)
        assert ITRF.shape == states.shape
        assert isinstance(ITRF, n.ndarray)
        assert ITRF.dtype == states.dtype

        for ind, jd in enumerate(jds):
            ITRF[:,ind] = tle.TEME_to_ITRF(states[:,ind], jd, xp=0, yp=0)

        r = n.sqrt(n.sum(ITRF[:3,:]**2,axis=0))
        v = n.sqrt(n.sum(ITRF[3:,:]**2,axis=0))

        theta, theta_dot = tle.theta_GMST1982(jds)

        nt.assert_almost_equal(r, n.ones(r.shape, dtype=r.dtype), decimal = 9)
        nt.assert_almost_equal(v, theta_dot/(3600.0*24.0), decimal = 9)

        self.assertLess(n.linalg.norm(ITRF[:,0] - ITRF[:,-1]), 0.1)
        self.assertGreater(n.linalg.norm(ITRF[:,0] - ITRF[:,len(day_f)//2]), 1.5)

    def test_TEME_to_ITRF_circle_polar(self):
        states = n.zeros((6,100),dtype=n.float)
        states[2,:] = 1.0
        day_f = n.linspace(0,1,endpoint=True,num=100,dtype=n.float)
        jds = n.empty(day_f.shape, dtype=n.float)
        for ind in range(len(day_f)):
            jds[ind] = dpt.date_to_jd(2008, 9, day_f[ind]+20.0)

        ITRF = n.empty(states.shape, dtype=states.dtype)
        for ind, jd in enumerate(jds):
            ITRF[:,ind] = tle.TEME_to_ITRF(states[:,ind], jd, xp=0, yp=0)

        r = n.sqrt(n.sum(ITRF[:3,:]**2,axis=0))
        v = n.sqrt(n.sum(ITRF[3:,:]**2,axis=0))

        dITRF = ITRF
        for ind in range(6):
            dITRF[ind,:] -= dITRF[ind,0]
        dr = n.sqrt(n.sum(dITRF[:3,:]**2,axis=0))

        nt.assert_almost_equal(r, n.ones(r.shape, dtype=r.dtype), decimal = 9)
        nt.assert_almost_equal(v, n.zeros(r.shape, dtype=r.dtype), decimal = 9)
        nt.assert_almost_equal(dr, n.zeros(r.shape, dtype=r.dtype), decimal = 9)


    def test_theta_GMST1982(self):
        day_f = n.arange(0.2,4.2,1.0,dtype=n.float)
        jds = n.empty(day_f.shape, dtype=n.float)
        for ind in range(len(day_f)):
            jds[ind] = dpt.date_to_jd(2008, 9, day_f[ind]+20.0)

        theta, theta_dot = tle.theta_GMST1982(jds)

        integrated = theta.copy()
        for ind in range(1,len(day_f)):
            integrated[ind] = integrated[ind-1] + theta_dot[ind-1]*(day_f[ind] - day_f[ind-1])

        nt.assert_array_less(n.std(theta - (integrated % (2.0*n.pi))), 0.00001)
        nt.assert_almost_equal(theta_dot/(2.0*n.pi), n.full(theta_dot.shape, 24.0/23.93, dtype=theta_dot.dtype), decimal=2)
    

    def test_TEME_to_TLE_cases(self):
        import propagator_sgp4

        R_E = 6353.0e3
        mjd0 = dpt.jd_to_mjd(2457126.2729)
        a = R_E*2.0
        orb_init_list = []

        orb_range = n.array([a, 0.9, 180, 360, 360, 360], dtype=n.float)
        orb_offset = n.array([R_E*1.1, 0, 0.0, 0.0, 0.0, 0.0], dtype=n.float)
        test_n = 5000
        while len(orb_init_list) < test_n:
            orb = n.random.rand(6)
            orb = orb_offset + orb*orb_range
            if orb[0]*(1.0 - orb[1]) > R_E+200e3:
                orb_init_list.append(orb)

        orb_init_list.append(n.array([R_E*1.2, 0, 0.0, 0.0, 0.0, 0.0], dtype=n.float))
        orb_init_list.append(n.array([R_E*1.2, 0, 0.0, 0.0, 0.0, 270], dtype=n.float))
        orb_init_list.append(n.array([R_E*1.2, 1e-9, 0.0, 0.0, 0.0, 0.0], dtype=n.float))
        orb_init_list.append(n.array([R_E*1.2, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=n.float))
        orb_init_list.append(n.array([R_E*1.2, 0.1, 75.0, 0.0, 0.0, 0.0], dtype=n.float))
        orb_init_list.append(n.array([R_E*1.2, 0.1, 0.0, 120.0, 0.0, 0.0], dtype=n.float))
        orb_init_list.append(n.array([R_E*1.2, 0.1, 0.0, 0.0, 35.0, 0.0], dtype=n.float))
        orb_init_list.append(n.array([R_E*1.2, 0.1, 75.0, 120.0, 0.0, 0.0], dtype=n.float))
        orb_init_list.append(n.array([R_E*1.2, 0.1, 75.0, 0.0, 35.0, 0.0], dtype=n.float))
        orb_init_list.append(n.array([R_E*1.2, 0.1, 75.0, 120.0, 35.0, 0.0], dtype=n.float))
        
        fail_inds = []
        errs = n.empty((len(orb_init_list),2), dtype=n.float)
        for ind, kep in enumerate(orb_init_list):

            M_earth = propagator_sgp4.SGP4.GM*1e9/consts.G

            state_TEME = dpt.kep2cart(kep, m=0.0, M_cent=M_earth, radians=False)*1e-3

            mean_elements = tle.TEME_to_TLE(state_TEME, mjd0=mjd0, kepler=False)

            state = propagator_sgp4.sgp4_propagation(mjd0, mean_elements, B=0, dt=0.0)

            state_diff = n.abs(state - state_TEME)*1e3
            er_r = n.linalg.norm(state_diff[:3])
            er_v = n.linalg.norm(state_diff[3:])
            errs[ind,0] = er_r
            errs[ind,1] = er_v
            try:
                self.assertLess(er_r, 10.0)
                self.assertLess(er_v, 1.0)
            except AssertionError as err:
                fail_inds.append(ind)

        if len(fail_inds) > 0:
            print('FAIL / TOTAL: {} / {}'.format(len(fail_inds), len(orb_init_list)))

        self.assertLess(n.median(errs[:,0]), 1e-2)
        self.assertLess(n.median(errs[:,1]), 1e-4)

        assert len(fail_inds) < float(test_n)/100.


    def test_TEME_to_TLE(self):
        import propagator_sgp4

        mjd0 = dpt.jd_to_mjd(2457126.2729)

        reci = n.array([
                -5339.76186573000,
                5721.43584226500,
                921.276953805000,
            ], dtype=n.float)

        veci = n.array([
                -4.88969089550000,
                -3.83304653050000,
                3.18013811100000,
            ], dtype=n.float)

        state_TEME = n.concatenate((reci, veci), axis=0)

        mean_elements = tle.TEME_to_TLE(state_TEME, mjd0=mjd0, kepler=False)

        state = propagator_sgp4.sgp4_propagation(mjd0, mean_elements, B=0, dt=0.0)

        state_diff = n.abs(state - state_TEME)*1e3

        nt.assert_array_less(state_diff[:3], n.full((3,), 1e-2, dtype=state_diff.dtype))
        nt.assert_array_less(state_diff[3:], n.full((3,), 1e-5, dtype=state_diff.dtype))


    def test_yearday_to_monthday(self):
        mm, dd = dpt.yearday_to_monthday(264.51782528, True) #2008
        nt.assert_almost_equal(mm, 9, decimal=8)
        nt.assert_almost_equal(dd, 20.51782528, decimal=8)

if __name__ == '__main__':
    unittest.main(verbosity=2)
