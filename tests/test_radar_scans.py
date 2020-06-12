import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import unittest
import numpy as n
import numpy.testing as nt

from radar_scans import RadarScan
import coord

class TestRadarScan(unittest.TestCase):
    
    def test_init(self):
        def pfun(t, **kw):
            return 0, 90, 1
        scan = RadarScan(23, 123, 101, pfun, 0.1, pointing_coord='azel', name='a scan')

    
    def test_keyword_arguments_local_p(self):
        def pfun(t, test, dwell_time, **kw):
            return test % 360, (45 + t) % 90, 1
        scan = RadarScan(23, 123, 101, pfun, 0.1, pointing_coord='azel', name='a scan')
        
        scan.keyword_arguments(test=10, dwell_time=1)
        
        az,el,r = scan._pointing(0)
        self.assertAlmostEqual(az, 10.)
        self.assertAlmostEqual(el, 45.)
        
        az,el,r = scan._pointing(10)
        self.assertAlmostEqual(az, 10.)
        self.assertAlmostEqual(el, 55.)
        
        scan.keyword_arguments(test=22)
        az,el,r = scan._pointing(10)
        self.assertAlmostEqual(az, 22.)
        self.assertAlmostEqual(el, 55.)

        def pfun(t, **kw):
            if t > 10:
                return 0, 45, 1
            else:    
                return 0, 90, 1
        scan = RadarScan(23, 123, 101, pfun, 0.1, pointing_coord='azel', name='a scan')
        
        kx,ky,kz = scan.local_pointing(0)
        self.assertAlmostEqual(kx, 0)
        self.assertAlmostEqual(ky, 0)
        self.assertAlmostEqual(kz, 1)
        
        kx,ky,kz = scan.local_pointing(20)
        self.assertAlmostEqual(kx, 0)
        self.assertAlmostEqual(ky, 1.0/n.sqrt(2))
        self.assertAlmostEqual(kz, 1.0/n.sqrt(2))
        
        
    
    def test_antenna_pointing(self):
        def pfun(t, test, dwell_time, **kw):
            if t > 5:
                return 0., 90., 1.
            else:
                return 0., 0., 1.
        scan = RadarScan(0, 0, 101, pfun, 0.1, pointing_coord='azel', name='a scan')
        
        scan.keyword_arguments(test=10, dwell_time=1)
        
        p0, k0 = scan.antenna_pointing(10)
        
        nt.assert_almost_equal(p0[0], coord.a + 101.0, decimal=5)
        nt.assert_almost_equal(k0[0], 1.0, decimal=5)
        
        p0, k0 = scan.antenna_pointing(0)
        
        nt.assert_almost_equal(p0[0], coord.a + 101.0, decimal=5)
        nt.assert_almost_equal(k0[2], 1.0, decimal=5)
    

    def test_dwell_time(self):
        def pfun(t, test, dwell_time, **kw):
            return test % 360, (45 + t) % 90, 1
        scan = RadarScan(23, 123, 101, pfun, 0.1, pointing_coord='azel', name='a scan')
        
        scan.keyword_arguments(dwell_time=1)
        dw = scan.dwell_time()
        self.assertAlmostEqual(dw, 1)
        
        
    def test_check_tx_compatibility(self):
        from antenna import AntennaTX
        
        ant = AntennaTX(
            name='a tx',
            lat=2.43,
            lon=11.0,
            el_thresh=30.0,
            freq=1e4,
            rx_noise=10e3,
            beam=None,
            alt = 0.0,
            scan=None,
            tx_power=1.0,
            tx_bandwidth=1.0,
            duty_cycle=1.0,
            pulse_length=1e-2,
            ipp=1e-1,
            n_ipp=10,
        )
        def pfun(t, **kw):
            return 360, (45 + t) % 90, 1
        scan = RadarScan(23, 123, 101, pfun, 0.1, pointing_coord='azel', name='a scan')
        
        
        scan.keyword_arguments(dwell_time = 0.5)
        
        with self.assertRaises(Exception):
            scan.check_tx_compatibility(ant)
            
        scan.keyword_arguments(dwell_time = 2)
        scan.check_tx_compatibility(ant)

        scan.keyword_arguments(dwell_time = 1.2)
        scan.check_tx_compatibility(ant)
        
        scan.keyword_arguments(dwell_time = [1.2, 3, 4, 6])
        scan.check_tx_compatibility(ant)
        
        scan.keyword_arguments(dwell_time = [1.2, 3, 0.4, 6])
        
        with self.assertRaises(Exception):
            scan.check_tx_compatibility(ant)

    def test_dwell_time(self):
        def pfun(t, **kw):
            return 360, (45 + t) % 90, 1
        scan = RadarScan(23, 123, 101, pfun, 0.1, pointing_coord='azel', name='a scan')
        
        self.assertAlmostEqual(scan.min_dwell_time, 0.1)
        scan.min_dwell_time = 0.2
        self.assertAlmostEqual(scan.min_dwell_time, 0.2)
        self.assertAlmostEqual(scan.min_dwell_time, scan._function_data['min_dwell_time'])

            

