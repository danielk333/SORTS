import sys
import os

import unittest
import numpy as np
import numpy.testing as nt

import h5py
import scipy.constants

from sorts.propagator import Kepler
import sorts


class TestSimulateScan(unittest.TestCase):

    def setUp(self):
        self.p = Kepler(in_frame='EME', out_frame='ITRF')
        self.radar = mock_radar()
        self.big_radar = mock_radar_mult()
        self.orbit = {
            'a': 7500,
            'e': 0,
            'i': 90.0,
            'raan': 0,
            'aop': 0,
            'mu0': 0.0,
            'mjd0': 57125.7729,
            'm': 0,
        }
        self.T = n.pi*2.0*n.sqrt(7500e3**3/MU_earth)

        self.o = SpaceObject(
            C_D=2.3, A=1.0,
            C_R=1.0, oid=42,
            d=1.0,
            propagator = PropagatorKepler,
            propagator_options = {
                'in_frame': 'EME',
                'out_frame': 'ITRF',
            },
            **self.orbit
        )

        #from pen and paper geometry we know that for a circular orbit and the radius of the earth at pole
        #that the place where object enters FOV is
        #true = mean = arccos(R_E / semi major)
        self.rise_ang = 90.0 - n.degrees(n.arccos(wgs84_a/7500e3))
        self.fall_ang = 180.0 - self.rise_ang

        #then we can find the time as a fraction of the orbit traversal from the angle
        self.rise_T = self.rise_ang/360.0*self.T
        self.fall_T = self.fall_ang/360.0*self.T

        self.num = simulate_tracking.find_linspace_num(t0=0.0, t1=self.T, a=7500e3, e=0.0, max_dpos=1e3)
        self.full_t = n.linspace(0, self.T, num=self.num)
        #Too see that this thoery is correct, run:
        #
        #ecef = self.o.get_orbit(n.linspace(self.rise_T, self.fall_T, num=self.num))
        #import dpt_tools as dpt
        #dpt.orbit3D(ecef)
        #


    def test_get_detections(self):
        det_times = simulate_scan.get_detections(self.o, self.radar, 0.0, self.T, logger=None, pass_dt=0.05)
        
        #should be detected
        assert len(det_times[0]['tm']) > 0

        best = n.argmin(det_times[0]['on_axis_angle'])

        self.assertLess(n.abs(det_times[0]['tm'][best] - self.T*0.25), 2.0) #best detection is overhead and should be after quater orbit, 2sec tol
        self.assertLess(n.abs(det_times[0]['range'][best] - (7500e3 - wgs84_a)), 1.0) #range is knownish, 1 meters res

        self.assertLess(n.abs(det_times[0]['t0'][0] - self.rise_T), self.T/self.num*20.0)
        self.assertLess(n.abs(det_times[0]['t1'][0] - self.fall_T), self.T/self.num*20.0)

        self.assertLess(n.min(det_times[0]['on_axis_angle']), 0.1) #less then 1/10 deg

