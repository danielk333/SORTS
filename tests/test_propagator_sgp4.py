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

from astropy.time import Time, TimeDelta



class TestSGP4(unittest.TestCase):

    def setUp(self):
        self.epoch0 = Time(2457126.2729, format='jd', scale='utc')
        self.params = dict(
            C_D = 2.3,
            m = 8000,
            A = 1.0,
        )
        self.state0 = np.array([7000e3, 0.0, 0.0, 0.0, 0.0, 7e3])
        self.settings = dict(in_frame='TEME', out_frame='TEME')


    def test_init(self):
        prop = SGP4(settings=self.settings)


    def test_SGP4_propagate(self):

        prop = SGP4(settings=self.settings)

        t = np.arange(0,24*360, dtype=np.float)*10.0

        ecefs = prop.propagate(t, self.state0, self.epoch0, **self.params)

        assert ecefs.shape == (6, t.size)
        assert isinstance(ecefs, np.ndarray)

        ecef = prop.propagate(0, self.state0, self.epoch0, **self.params)

        assert ecef.shape == (6, )
        assert isinstance(ecef, np.ndarray)

        nt.assert_almost_equal(ecefs[:,0], ecef)


    def test_SGP4_propagate_B(self):

        B = 0.5*self.params['C_D']*self.params['A']/self.params['m']
        
        prop = SGP4(settings=self.settings)

        t = np.arange(0,24*360, dtype=np.float)*10.0

        ecefs = prop.propagate(t, self.state0, self.epoch0, **self.params)
        ecefs_B = prop.propagate(t, self.state0, self.epoch0, B=B)

        assert ecefs.shape == ecefs_B.shape
        nt.assert_almost_equal(ecefs, ecefs_B)


    def test_propagator_sgp4_mjd_invaraiance(self):
        #make sure that initial MJD does NOT matter when the frame is kept as TEME

        # Assemble all data that is required for SGP4 propagation
        # Epoch, mean orbital elements, ballistic coefficient

        prop = SGP4(settings=self.settings)

        t = np.arange(0,24*360, dtype=np.float)*10.0

        ecefs = prop.propagate(t, self.state0, self.epoch0, **self.params)

        mjd = np.linspace(0,40,num=33)
        for ind in range(len(mjd)):
            ecefs1 = prop.propagate(t, self.state0, self.epoch0 + TimeDelta(mjd[ind]*24*3600.0, format='sec'), **self.params)
            assert ecefs.shape == ecefs1.shape
            nt.assert_almost_equal(ecefs, ecefs1, decimal=5)


    def test_get_mean_elements(self):

        prop = SGP4(settings=self.settings)

        l1 = '1     5U 58002B   20251.29381767 +.00000045 +00000-0 +68424-4 0  9990'
        l2 = '2     5 034.2510 336.1746 1845948 000.5952 359.6376 10.84867629214144'

        mean, B, epoch = prop.get_mean_elements(l1, l2)

        prop.settings['tle_input'] = True

        t = np.arange(0,24*3600, dtype=np.float)
        states0 = prop.propagate(t, (l1,l2), epoch)

        prop.settings['tle_input'] = False

        states1 = prop.propagate(t, mean, epoch, B=B, SGP4_mean_elements=True)

        nt.assert_almost_equal(states0, states1, decimal=1)
