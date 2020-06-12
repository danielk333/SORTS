import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import time

import unittest
import numpy as n
import numpy.testing as nt
import scipy

from radar_config import RadarSystem
import antenna

class TestRadarSystem(unittest.TestCase):

    def setUp(self):
        self.beam_u = antenna.BeamPattern(lambda k, sf: sf.I_0, az0=0.0, el0=90.0, I_0=1.0, f=1.0, beam_name='Test')

        self.tx_lst = []

        self.tx_lst.append(
            antenna.AntennaTX(
                name='a tx',
                lat=2.43, 
                lon=11.0, 
                alt=0.0, 
                el_thresh=30.0, 
                freq=1e4, 
                rx_noise=10e3, 
                beam=self.beam_u, 
                scan=None, 
                tx_power=1.0, 
                tx_bandwidth=1.0, 
                duty_cycle=1.0,
            )
        )
        self.tx_lst.append(
            antenna.AntennaTX(
                name='a tx',
                lat=2.43, 
                lon=12.0, 
                alt=0.0, 
                el_thresh=30.0, 
                freq=1e4, 
                rx_noise=10e3, 
                beam=self.beam_u, 
                scan=None, 
                tx_power=1.0, 
                tx_bandwidth=1.0, 
                duty_cycle=1.0,
            )
        )

        self.rx_lst = []
        for of in range(10):
            self.tx_lst.append(
                antenna.AntennaRX(name='a rx', lat=2.43+of, lon=11.0, alt=0.0, el_thresh=30.0, freq=1e4, rx_noise=10e3, beam=self.beam_u)
            )

    def _add_radar(self):
        self.radar = RadarSystem(self.tx_lst, self.rx_lst, 'a system', max_on_axis=15.0, min_SNRdb=10.0)

    def test_init(self):
        self._add_radar()

    def test_set_FOV(self):
        self._add_radar()
        self.radar.set_FOV(max_on_axis=23.0, horizon_elevation=30.0)

        nt.assert_almost_equal(self.radar._horizon_elevation, 30.0, decimal=9)
        nt.assert_almost_equal(self.radar.max_on_axis, 23.0, decimal=9)
        for tx in self.radar._tx:
            nt.assert_almost_equal(tx.el_thresh, 30.0, decimal=9)
        for rx in self.radar._rx:
            nt.assert_almost_equal(rx.el_thresh, 30.0, decimal=9)

    def test_set_SNR(self):
        self._add_radar()
        self.radar.set_SNR_limits(min_total_SNRdb=10.0, min_pair_SNRdb=5.0)

        nt.assert_almost_equal(self.radar.min_SNRdb, 10.0, decimal=9)
        for tx in self.radar._tx:
            nt.assert_almost_equal(10.0*n.log10(tx.enr_thresh), 5.0, decimal=9)

    def test_set_scan(self):
        self._add_radar()
        from radar_scans import RadarScan
        def pfun(t, **kw):
            return 0, 90, 1
        scan = RadarScan(23, 123, 101, pfun, 0.1, pointing_coord='azel', name='a scan')
        scan1 = RadarScan(23, 123, 101, pfun, 0.1, pointing_coord='azel', name='1')
        scan2 = RadarScan(23, 123, 101, pfun, 0.1, pointing_coord='azel', name='2')

        self.radar.set_scan(scan)

        for tx in self.radar._tx:
            assert tx.scan is not scan, 'Scan not copied!'
            
            self.assertEqual(tx.scan._lat, tx.lat)
            self.assertEqual(tx.scan._lon, tx.lon)
            self.assertEqual(tx.scan._alt, tx.alt)
            assert tx.scan._pointing_function is scan._pointing_function, 'not same pointing func'
            self.assertEqual(tx.scan._pointing_coord, scan._pointing_coord)
            self.assertEqual(tx.scan.name, scan.name)
            self.assertDictEqual(tx.scan._function_data, scan._function_data)
            self.assertEqual(tx.scan.min_dwell_time, scan.min_dwell_time)


        self.radar.set_scan(scan, [scan1,scan2])

        for tx in self.radar._tx:
            self.assertEqual(tx.extra_scans[0].name, scan1.name)
            self.assertEqual(tx.extra_scans[1].name, scan2.name)
            assert tx.extra_scans[0] is not scan1, 'Scan not copied!'
            assert tx.extra_scans[1] is not scan2, 'Scan not copied!'