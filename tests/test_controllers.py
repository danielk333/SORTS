import sys
import os

import unittest
import numpy as np
import numpy.testing as nt

import h5py
import scipy.constants

import sorts
from sorts import controller as ctrl


class TestTracker(unittest.TestCase):


    def test_init(self):
        radar = sorts.radars.mock

        ecefs = radar.tx[0].ecef.copy().reshape(3,1)*2
        t = np.array([2.0])
        rc = ctrl.Tracker(radar, t, ecefs, t0=0.0, dwell=0.1, return_copy=False)


    def test_call(self):
        radar = sorts.radars.mock

        ecefs = radar.tx[0].ecef.copy().reshape(3,1)*2
        t = np.array([2.0])
        rc = ctrl.Tracker(radar, t, ecefs, t0=0.0, dwell=0.1, return_copy=False)

        rc.point(rc.radar, np.array([1,0,0]))
        
        tt = [1.0,2.0,3.0]
        for ind, mrad in zip(range(len(tt)), rc(tt)):
            radar, meta = mrad
            if ind == 1:
                for st in radar.tx + radar.rx:
                    assert st.enabled
                nt.assert_almost_equal(st.beam.pointing, np.array([0,0,1]), decimal=2)

            else:
                for st in radar.tx + radar.rx:
                    assert not st.enabled





class TestScanner(unittest.TestCase):

    def test_init(self):

        class MyScan(sorts.radar.Scan):
            def __init__(self):
                super().__init__(coordinates='azelr')

            def pointing(self, t):
                sph = np.ones((3,tn), dtype=np.float64)
                sph[0,:] = 0.0
                sph[1,:] = 90.0

                sph[1,t < 10] = 0.0

                return sph

        sc = MyScan()

        radar = sorts.radars.mock
        rc = ctrl.Scanner(radar, sc, profiler=None, logger=None, return_copy=False)


    def test_call(self):

        class MyScan(sorts.radar.Scan):
            def __init__(self):
                super().__init__(coordinates='azelr')

            def pointing(self, t):
                try:
                    tn = len(t)
                except:
                    tn = 1
                    t = np.array([t])
                sph = np.ones((3,tn), dtype=np.float64)
                sph[0,:] = 0.0
                sph[1,:] = 90.0

                sph[1,t > 10] = 0.0

                return sph

        sc = MyScan()

        radar = sorts.radars.mock
        rc = ctrl.Scanner(radar, sc, profiler=None, logger=None, return_copy=False)

        tt = np.array([5.0,10.0,20.0])
        for ind, mrad in zip(range(len(tt)), rc(tt)):
            radar, meta = mrad

            if ind < 2:
                nt.assert_almost_equal(radar.tx[0].beam.pointing, np.array([0,0,1], dtype=np.float64), decimal = 6)
            else:
                nt.assert_almost_equal(radar.tx[0].beam.pointing, np.array([0,1,0], dtype=np.float64), decimal = 6)






class TestStatic(unittest.TestCase):

    def test_init(self):
        radar = sorts.radars.mock
        rc = ctrl.Static(radar, azimuth = 0.0, elevation=0.0)
