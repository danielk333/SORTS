import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import unittest
import numpy as n
import numpy.testing as nt

import h5py
import scipy.constants

import dpt_tools as dpt


import simulate_scan
import simulate_tracking
import simulate_tracklet

from propagator_kepler import PropagatorKepler
import antenna_library as alib
import antenna
import radar_scan_library as rslib
from propagator_sgp4 import MU_earth
from radar_config import RadarSystem
from space_object import SpaceObject
import radar_library as rlib
import population_library
import ccsds_write
import correlator

def mock_radar():

    lat = 90.0
    lon = 0.0
    alt = 0.0

    rx_beam = alib.planar_beam(
            az0 = 0,
            el0 = 90,
            I_0=10**4.5,
            f=233e6,
            a0=40,
            az1=0.0,
            el1=90.0,
    )
    tx_beam = alib.planar_beam(
            az0 = 0,
            el0 = 90,
            I_0=10**4.5,
            f=233e6,
            a0=40,
            az1=0.0,
            el1=90.0,
    )


    rx = antenna.AntennaRX(
        name="Top",
        lat=lat,
        lon=lon,
        alt=alt,
        el_thresh=30,
        freq=233e6,
        rx_noise=120,
        beam=rx_beam,
    )

    scan = rslib.beampark_model(
        az = 0.0, 
        el = 90.0, 
        lat = lat, 
        lon = lon,
        alt = alt,
    )

    tx = antenna.AntennaTX(
        name="Top TX",
        lat=lat,
        lon=lon,
        alt=alt,
        el_thresh=30,
        freq=233e6,
        rx_noise=120,
        beam=tx_beam,
        scan=scan,
        tx_power=5.0e6,
        tx_bandwidth=1e6,  # 1 MHz
        duty_cycle=0.25,
        pulse_length=30.0*64.0*1e-6,
        ipp=20e-3,
        n_ipp=10.0,
    )

    tx=[tx]
    rx=[rx]

    Mock = RadarSystem(tx, rx, 'Mock radar')
    Mock.set_FOV(max_on_axis = 90.0, horizon_elevation = 0.0)
    return Mock



def mock_radar_mult():

    lat = [85.0, 89.0, 90.0, 89.0, 85.0]
    lon = [0, 90.0, 0, 270.0, 180]
    alt = 0.0
    tx_l=[]
    rx_l=[]

    for ind in range(5):
        rx_beam = alib.planar_beam(
                az0 = 0,
                el0 = 90,
                I_0=10**4.5,
                f=233e6,
                a0=40,
                az1=0.0,
                el1=90.0,
        )
        rx = antenna.AntennaRX(
            name="Top",
            lat=lat[ind],
            lon=lon[ind],
            alt=alt,
            el_thresh=30,
            freq=233e6,
            rx_noise=120,
            beam=rx_beam,
        )
        rx_l.append(rx)

    lat = [90.0, 87.0]
    lon = [0, 0.0]

    for ind in range(2):
        tx_beam = alib.planar_beam(
                az0 = 0,
                el0 = 90,
                I_0=10**4.5,
                f=233e6,
                a0=40,
                az1=0.0,
                el1=90.0,
        )

        scan = rslib.beampark_model(
            az = 0.0, 
            el = 90.0, 
            lat=lat[ind],
            lon=lon[ind],
            alt = alt,
        )

        tx = antenna.AntennaTX(
            name="Top TX",
            lat=lat[ind],
            lon=lon[ind],
            alt=alt,
            el_thresh=30,
            freq=233e6,
            rx_noise=120,
            beam=tx_beam,
            scan=scan,
            tx_power=5.0e6,
            tx_bandwidth=1e6,  # 1 MHz
            duty_cycle=0.25,
            pulse_length=30.0*64.0*1e-6,
            ipp=20e-3,
            n_ipp=10.0,
        )
        tx_l.append(tx)


    Mock = RadarSystem(tx_l, rx_l, 'bIG Mock radar')
    Mock.set_FOV(max_on_axis = 90.0, horizon_elevation = 0.0)
    return Mock

wgs84_a = 6356.7523142*1e3

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


class TestSimulateTracklet(unittest.TestCase):

    def test_create_tracklet(self):

        radar = rlib.eiscat_uhf()
        radar.set_FOV(30.0, 25.0)

        #tle files for envisat in 2016-09-05 to 2016-09-07 from space-track.
        TLEs = [
            ('1 27386U 02009A   16249.14961597  .00000004  00000-0  15306-4 0  9994',
            '2 27386  98.2759 299.6736 0001263  83.7600 276.3746 14.37874511760117'),
            ('1 27386U 02009A   16249.42796553  .00000002  00000-0  14411-4 0  9997',
            '2 27386  98.2759 299.9417 0001256  82.8173 277.3156 14.37874515760157'),
            ('1 27386U 02009A   16249.77590267  .00000010  00000-0  17337-4 0  9998',
            '2 27386  98.2757 300.2769 0001253  82.2763 277.8558 14.37874611760201'),
            ('1 27386U 02009A   16250.12384028  .00000006  00000-0  15974-4 0  9995',
            '2 27386  98.2755 300.6121 0001252  82.5872 277.5467 14.37874615760253'),
            ('1 27386U 02009A   16250.75012691  .00000017  00000-0  19645-4 0  9999',
            '2 27386  98.2753 301.2152 0001254  82.1013 278.0311 14.37874790760345'),
        ]

        pop = population_library.tle_snapshot(TLEs, sgp4_propagation=True)

        #it seems to around 25m^2 area
        d = n.sqrt(25.0*4/n.pi)
        pop.add_column('d', space_object_uses=True)
        pop['d'] = d

        ccsds_file = './data/uhf_test_data/events/2002-009A-1473150428.tdm'

        obs_data = ccsds_write.read_ccsds(ccsds_file)
        jd_obs = dpt.mjd_to_jd(dpt.npdt2mjd(obs_data['date']))

        date_obs = obs_data['date']
        sort_obs = n.argsort(date_obs)
        date_obs = date_obs[sort_obs]
        r_obs = obs_data['range'][sort_obs]

        jd_sort = jd_obs.argsort()
        jd_obs = jd_obs[jd_sort]

        jd_det = jd_obs[0]

        jd_pop = dpt.mjd_to_jd(pop['mjd0'])

        pop_id = n.argmin(n.abs(jd_pop - jd_det))
        obj = pop.get_object(pop_id)

        print(obj)

        jd_obj = dpt.mjd_to_jd(obj.mjd0)

        print('Day difference detection - TLE: {}'.format(jd_det- jd_obj))

        t_obs = (jd_obs - jd_obj)*(3600.0*24.0)

        meas, fnames, ecef_stdevs = simulate_tracklet.create_tracklet(
            obj,
            radar,
            t_obs,
            hdf5_out=True,
            ccsds_out=True,
            dname="./tests/tmp_test_data",
            noise=False,
        )

        out_h5 = fnames[0] + '.h5'
        out_ccsds = fnames[0] + '.tdm'

        print('FILES: ', fnames)

        with h5py.File(out_h5,'r') as h_det:
            assert 'm_range' in h_det
            assert 'm_range_rate' in h_det
            assert 'm_time' in h_det

        sim_data = ccsds_write.read_ccsds(out_ccsds)

        date_sim = sim_data['date']
        sort_sim = n.argsort(date_sim)
        date_sim = date_sim[sort_sim]

        r_sim = sim_data['range'][sort_sim]
        v_sim = sim_data['doppler_instantaneous'][sort_sim]

        lt_correction = n.round(r_sim/scipy.constants.c*1e6).astype(n.int64).astype('timedelta64[us]')

        date_sim_cor = date_sim + lt_correction

        t_sim = dpt.jd_to_unix(dpt.mjd_to_jd(dpt.npdt2mjd(date_sim_cor)))

        for ind in range(len(date_sim)):
            time_df = (dpt.npdt2mjd(date_sim_cor[ind]) - dpt.npdt2mjd(date_obs[ind]))*3600.0*24.0
            assert time_df < 0.01

        assert len(r_obs) == len(r_sim)

        dat = {
            't': t_sim,
            'r': r_sim*1e3,
            'v': v_sim*1e3,
        }

        cdat = correlator.correlate(
            data = dat,
            station = radar._rx[0],
            population = pop,
            metric = correlator.residual_distribution_metric,
            n_closest = 1,
            out_file = None,
            verbose = False,
            MPI_on = False,
        )

        self.assertLess(n.abs(cdat[0]['stat'][0]), 5.0)
        self.assertLess(n.abs(cdat[0]['stat'][1]), 50.0)
        self.assertLess(n.abs(cdat[0]['stat'][2]), 5.0)
        self.assertLess(n.abs(cdat[0]['stat'][3]), 50.0)

        nt.assert_array_less(n.abs(r_sim - r_obs), 1.0)

        os.remove(out_h5)
        print('removed "{}"'.format(out_h5))

        os.remove(out_ccsds)
        print('removed "{}"'.format(out_ccsds))

        sat_folder = os.sep.join(fnames[0].split(os.sep)[:-1])
        os.rmdir(sat_folder)
        print('removed "{}"'.format(sat_folder))

