import sys
import os

import unittest
import numpy as np
import numpy.testing as nt

import h5py
import scipy.constants


class TestSimulateScan(unittest.TestCase):

    def setUp(self):
        self.p = PropagatorKepler(in_frame='EME', out_frame='ITRF')
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



class TestSimulateTracking(unittest.TestCase):

    def setUp(self):
        self.p = PropagatorKepler(in_frame='EME', out_frame='ITRF')
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

    def test_find_linspace_num(self):

        a = 7000e3
        e = 0.0
        T = n.pi*2.0*n.sqrt(a**3/MU_earth)
        circ = n.pi*2.0*a

        num = simulate_tracking.find_linspace_num(t0=0.0, t1=T, a=a, e=e, max_dpos=circ/10.0)

        self.assertEqual(num, 10)

    def test_find_pass_interval(self):
        
        passes, passes_id, idx_v, postx_v, posrx_v = simulate_tracking.find_pass_interval(self.full_t, self.o, self.radar)

        assert len(passes) == len(self.radar._tx)
        assert len(passes_id) == len(self.radar._tx)
        
        assert len(passes[0]) == 1
        assert len(passes_id[0]) == 1

        self.assertLess(n.abs(passes[0][0][0] - self.rise_T), self.T/self.num*20.0)
        self.assertLess(n.abs(passes[0][0][1] - self.fall_T), self.T/self.num*20.0)

        self.assertAlmostEqual(self.full_t[idx_v][passes_id[0][0][0]], passes[0][0][0])
        self.assertAlmostEqual(self.full_t[idx_v][passes_id[0][0][1]], passes[0][0][1])

    def test_get_passes(self):
        pass_struct = simulate_tracking.get_passes(
            self.o, 
            self.radar, 
            t0=0.0, 
            t1=self.T, 
            max_dpos=1e3, 
            logger = None, 
            plot = False,
        )

        assert 't' in pass_struct
        assert 'snr' in pass_struct

        assert len(pass_struct['t']) == len(self.radar._tx)
        assert len(pass_struct['snr']) == len(self.radar._tx)
        
        assert len(pass_struct['t'][0]) == 1
        assert len(pass_struct['snr'][0]) == 1

        assert len(pass_struct['t'][0][0]) == 2
        assert len(pass_struct['snr'][0][0][0]) == 2

        ecef = self.o.get_orbit(pass_struct['snr'][0][0][0][1])

        rel_vec = ecef.T - self.radar._rx[0].ecef #check that its above the radar at max

        self.assertLess(n.linalg.norm(rel_vec) - (7500e3 - wgs84_a), 1.0) #1m

        self.assertLess(n.abs(pass_struct['t'][0][0][0] - self.rise_T), self.T/self.num*20.0)
        self.assertLess(n.abs(pass_struct['t'][0][0][1] - self.fall_T), self.T/self.num*20.0)



    def test_get_passes_many_tx(self):
        pass_struct = simulate_tracking.get_passes(
            self.o, 
            self.big_radar, 
            t0=0.0, 
            t1=self.T, 
            max_dpos=1e3, 
            logger = None, 
            plot = False,
        )

        assert 't' in pass_struct
        assert 'snr' in pass_struct

        assert len(pass_struct['t']) == len(self.big_radar._tx)
        assert len(pass_struct['snr']) == len(self.big_radar._tx)
        
        assert len(pass_struct['t'][0]) == 1
        assert len(pass_struct['snr'][0]) == 1

        assert len(pass_struct['snr'][0][0]) == len(self.big_radar._rx)

        assert len(pass_struct['t'][0][0]) == 2
        assert len(pass_struct['snr'][0][0][0]) == 2


        for ind, times in enumerate(pass_struct['snr'][0][0]):#they are in series along lat line and thus the times should be in decreeing order
            if ind == 0:
                last_time = times[1]
            else:
                assert last_time > times[1]
            assert times[0] > 0.0, 'snr should always be above 0 in this situation'

        self.assertLess(n.abs(pass_struct['t'][0][0][0] - self.rise_T), self.T/self.num*20.0)
        self.assertLess(n.abs(pass_struct['t'][0][0][1] - self.fall_T), self.T/self.num*20.0)

        self.assertLess(n.abs(pass_struct['t'][1][0][0] - self.rise_T), 600.0) #should be close to same rise fall time
        self.assertLess(n.abs(pass_struct['t'][1][0][1] - self.fall_T), 600.0)

        ecef = self.o.get_orbit(pass_struct['snr'][0][0][2][1])
        rel_vec = ecef.T - self.big_radar._rx[2].ecef #check that its above the radar at max
        self.assertLess(n.linalg.norm(rel_vec) - (7500e3 - wgs84_a), 1) #1m

    def test_get_angles(self):
        pass_struct = simulate_tracking.get_passes(
            self.o, 
            self.radar, 
            t0=0.0, 
            t1=self.T, 
            max_dpos=1e3, 
            logger = None, 
            plot = False,
        )
        t, angles = simulate_tracking.get_angles(pass_struct, self.o, self.radar, dt=0.1)

        self.assertLess(n.min(angles[0]), 0.1)

