import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import time

import unittest
import numpy as n
import numpy.testing as nt
import scipy
import h5py

import population_filter

from population import Population
import radar_library as rlib

from test_simulate_functions import mock_radar

class TestPopulationFilter(unittest.TestCase):

    def test_filter_objects(self):
        pop = Population(
            name='two objects',
            extra_columns = ['d'],
            space_object_uses = [True],
        )
        pop.allocate(3)
        pop.objs[0] = (0,
            8000.0,
            0.01,
            0.0,
            0.0,
            0.0,
            0.0,
            57125.7729,
            10.0,
        )
        pop.objs[1] = (0,
            8000.0,
            0.01,
            90.0,
            0.0,
            0.0,
            0.0,
            57125.7729,
            10.0,
        )
        pop.objs[2] = (0,
            8000.0,
            0.01,
            70.0,
            0.0,
            0.0,
            0.0,
            57125.7729,
            10.0,
        )


        radar = mock_radar()

        detectable, n_rx_dets, peak_snrs = population_filter.filter_objects(radar, pop, ofname=None, prop_time=24.0)

        nt.assert_array_almost_equal(detectable, n.array([0, 1, 1], dtype=detectable.dtype))


