import sys
import os

import unittest
import numpy as np
import numpy.testing as nt

import h5py
import scipy.constants

import sorts
from sorts.controller import RadarController


class TestRadarController(unittest.TestCase):

    def test_init_base(self):
        radar = sorts.radars.mock

        with self.assertRaises(TypeError):
            rc = RadarController(radar)

    def test_subclass_init(self):
        radar = sorts.radars.mock

        class Ctrl(RadarController):
            def generator(self, t, **kwargs):
                for ti in t:
                    yield self.radar, self.default_meta()


        rc = Ctrl(radar)

        for rad, meta in rc([1,2,3,4]):
            assert rad is radar

    def test_meta_subclass(self):
        radar = sorts.radars.mock

        class Ctrl(RadarController):
            META_FIELDS = RadarController.META_FIELDS + [
                'test',
            ]

            def generator(self, t, **kwargs):
                for ti in t:
                    yield self.radar, self.default_meta()


        rc = Ctrl(radar)

        for rad, meta in rc([1,2,3,4]):
            for key in RadarController.META_FIELDS:
                assert key in meta
            for key in Ctrl.META_FIELDS:
                assert key in meta
            assert meta['test'] is None


    def test_point_rx_ecef(self):
        radar = sorts.radars.mock
        
        ecef_p = radar.rx[0].ecef + np.array([1,0,0]) 
        #to +x i.e down along prime meridian at 0deg elevation
        #then locally, since south is "behind", locally we should be pointing towards y=-1

        RadarController.point_rx_ecef(radar, ecef_p)

        p1 = radar.rx[0].beam.pointing.copy()

        #back to zenith
        RadarController.point_rx_ecef(radar, radar.rx[0].ecef*2)
        p2 = radar.rx[0].beam.pointing.copy()

        nt.assert_almost_equal(p1, np.array([0,-1,0]), decimal=1)
        nt.assert_almost_equal(p2, np.array([0,0,1]), decimal=1)
        
    def test_point_tx_ecef(self):
        radar = sorts.radars.mock
        
        ecef_p = radar.tx[0].ecef + np.array([1,0,0]) 
        #to +x i.e down along prime meridian at 0deg elevation
        #then locally, since south is "behind", locally we should be pointing towards y=-1

        RadarController.point_tx_ecef(radar, ecef_p)

        p1 = radar.tx[0].beam.pointing.copy()

        #back to zenith
        RadarController.point_tx_ecef(radar, radar.tx[0].ecef*2)
        p2 = radar.tx[0].beam.pointing.copy()

        nt.assert_almost_equal(p1, np.array([0,-1,0]), decimal=1)
        nt.assert_almost_equal(p2, np.array([0,0,1]), decimal=1)
        

    def test_point(self):
        radar = sorts.radars.mock

        enu = np.array([1,0,0])

        RadarController.point(radar, enu)

        p0 = radar.tx[0].beam.pointing.copy()

        nt.assert_almost_equal(p0, enu, decimal=8)


    def test_enabled(self):
        radar = sorts.radars.mock

        RadarController.turn_off(radar)

        for st in radar.tx + radar.rx:
            assert not st.enabled

        RadarController.turn_on(radar)

        for st in radar.tx + radar.rx:
            assert st.enabled
