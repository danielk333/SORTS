import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import unittest
import numpy as np
import numpy.testing as nt

from sorts.radar import Scan
import sorts

class TestScan(unittest.TestCase):
    
    def test_init(self):
        class MyScan(Scan):
            def __init__(self):
                super().__init__(coordinates='azelr')

            def pointing(self, t):
                pass

        sc = MyScan()


    def test_init_abstractmenthod(self):

        class MyScan(Scan):
            def __init__(self):
                super().__init__(coordinates='azelr')

        with self.assertRaises(TypeError):
            sc = MyScan()

    
    def test_pointing(self):
        class MyScan(Scan):
            def __init__(self):
                super().__init__(coordinates='azelr')

            def pointing(self, t):
                try:
                    tn = len(t)
                except:
                    tn = 1
                sph = np.ones((3,tn), dtype=np.float64)
                sph[0,:] = 0.0
                sph[1,:] = 90.0
                return sph

        sc = MyScan()

        nt.assert_almost_equal(sc.pointing(0.0).flatten(), np.array([0,90,1], dtype=np.float64), decimal=4)

    def test_enu_pointing(self):
        class MyScan(Scan):
            def __init__(self):
                super().__init__(coordinates='azelr')

            def pointing(self, t):
                try:
                    tn = len(t)
                except:
                    tn = 1
                sph = np.ones((3,tn), dtype=np.float64)
                sph[0,:] = 0.0
                sph[1,:] = 90.0
                return sph

        sc = MyScan()

        nt.assert_almost_equal(sc.enu_pointing(0.0).flatten(), np.array([0,0,1], dtype=np.float64), decimal=4)

    def test_ecef_pointing(self):

        class MyScan(Scan):
            def __init__(self):
                super().__init__(coordinates='azelr')

            def pointing(self, t):
                try:
                    tn = len(t)
                except:
                    tn = 1
                sph = np.ones((3,tn), dtype=np.float64)
                sph[0,:] = 0.0
                sph[1,:] = 90.0
                return sph

        sc = MyScan()

        radar = sorts.radars.mock

        #at north pole, zenith is +z
        nt.assert_almost_equal(sc.ecef_pointing(0.0, radar.tx[0]).flatten(), np.array([0,0,1], dtype=np.float64), decimal=4)
        
        radar.tx[0].rebase(0, 0, 0) #equator prime meridian

        #then zenith is towards +x
        nt.assert_almost_equal(sc.ecef_pointing(0.0, radar.tx[0]).flatten(), np.array([1,0,0], dtype=np.float64), decimal=4)

