import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import time

import unittest
import numpy as n
import numpy.testing as nt
import scipy
import h5py

import correlator

import population_library
import radar_library as rlib

class TestCorrelator(unittest.TestCase):

    def test_correlator(self):


        radar = rlib.eiscat_uhf()

        measurement_folder = './data/uhf_test_data/events'
        tle_file = './data/uhf_test_data/tle-201801.txt'
        measurement_file = measurement_folder + '/det-000000.h5'

        with h5py.File(measurement_file,'r') as h_det:
            r = h_det['r'].value*1e3
            t = h_det['t'].value
            v = -h_det['v'].value

            dat = {
                'r': r,
                't': t,
                'v': v,
            }

        pop = population_library.tle_snapshot(tle_file, sgp4_propagation=True)

        pop.filter('oid', lambda oid: n.abs(oid - 43075) < 50)

        cdat = correlator.correlate(
            data = dat,
            station = radar._rx[0],
            population = pop,
            metric = correlator.residual_distribution_metric,
            n_closest = 2,
            out_file = None,
            verbose = False,
            MPI_on = False,
        )

        assert int(cdat[0]['sat_id']) == 43075
        self.assertLess(n.abs(cdat[0]['stat'][0]), 1e3)
        self.assertLess(n.abs(cdat[0]['stat'][1]), 500)
        self.assertLess(n.abs(cdat[0]['stat'][2]), 10.0)
        self.assertLess(n.abs(cdat[0]['stat'][3]), 5.0)

    