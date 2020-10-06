import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import time

import unittest
import pathlib

import numpy as np
import numpy.testing as nt
import scipy
import h5py
from astropy.time import Time

import sorts

radar = sorts.radars.eiscat_uhf


class TestCorrelator(unittest.TestCase):

    def test_correlator(self):

        try:
            tle_pth = pathlib.Path(__file__).parent / 'data' / 'uhf_correlation' / 'tle-201801.txt'
            obs_pth = pathlib.Path(__file__).parent / 'data' / 'uhf_correlation' / 'det-000000.h5'
        except NameError:
            import os
            tle_pth = 'data' + os.path.sep + 'uhf_correlation' + os.path.sep + 'tle-201801.txt'
            obs_pth = 'data' + os.path.sep + 'uhf_correlation' + os.path.sep + 'det-000000.h5'

        # Each entry in the input `measurements` list must be a dictionary that contains the following fields:
        #   * 't': [numpy.ndarray] Times relative epoch in seconds
        #   * 'r': [numpy.ndarray] Two-way ranges in meters
        #   * 'v': [numpy.ndarray] Two-way range-rates in meters per second
        #   * 'epoch': [astropy.Time] epoch for measurements
        #   * 'tx': [sorts.TX] Pointer to the TX station
        #   * 'rx': [sorts.RX] Pointer to the RX station
        print('Loading EISCAT UHF monostatic measurements')
        with h5py.File(str(obs_pth),'r') as h_det:
            r = h_det['r'][()]*1e3 #km -> m, one way
            t = h_det['t'][()] #Unix seconds
            v = -h_det['v'][()] #Inverted definition of range rate, one way

            inds = np.argsort(t)
            t = t[inds]
            r = r[inds]
            v = v[inds]

            t = Time(t, format='unix', scale='utc')
            epoch = t[0]
            t = (t - epoch).sec

            dat = {
                'r': r*2,
                't': t,
                'v': v*2,
                'epoch': epoch,
                'tx': radar.tx[0],
                'rx': radar.rx[0],
            }

        print('Loading TLE population')
        tles = [
            '1 43075U 17083F   18004.50601262  .00000010  00000-0 -26700-5 0  9995',
            '2 43075  86.5273 171.5023 0000810  83.8359 276.2943 14.58677444  1820',
        ]
        pop = sorts.population.tle_catalog(tles, cartesian=False)

        #correlate requires output in ECEF 
        pop.out_frame = 'ITRS'

        print('Correlating data and population')
        indecies, metric, cdat = sorts.correlate(
            measurements = [dat],
            population = pop,
            n_closest = 1,
        )


        assert int(indecies[0]) == 43075

        r_ref = cdat[0][0]['r_ref']
        v_ref = cdat[0][0]['v_ref']

        self.assertLess(n.abs(np.mean(r_ref - r)), 1e3)
        self.assertLess(n.abs(np.std(r_ref - r)), 500)
        self.assertLess(n.abs(np.mean(v_ref - v)), 10.0)
        self.assertLess(n.abs(np.std(v_ref - v)), 5.0)

    