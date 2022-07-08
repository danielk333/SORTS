import sys
import os

import unittest
import numpy as np
import numpy.testing as nt

import h5py
import scipy.constants

import sorts
from sorts.radar import controllers as ctrl
from sorts.radar import scans


# class TestTracker(unittest.TestCase):

#     def test_init(self):
#         radar = sorts.radars.mock

#         ecefs = radar.tx[0].ecef.copy().reshape(3,1)*2
#         t = np.array([2.0])
#         rc = ctrl.Tracker(radar, t, ecefs, t0=0.0, dwell=0.1, return_copy=False)


#     def test_call(self):
#         radar = sorts.radars.mock

#         ecefs = radar.tx[0].ecef.copy().reshape(3,1)*2
#         t = np.array([2.0])
#         rc = ctrl.Tracker(radar, t, ecefs, t0=0.0, dwell=0.1, return_copy=False)

#         rc.point(rc.radar, np.array([1,0,0]))
        
#         tt = [1.0,2.0,3.0]
#         for ind, mrad in zip(range(len(tt)), rc(tt)):
#             radar, meta = mrad
#             if ind == 1:
#                 for st in radar.tx + radar.rx:
#                     assert st.enabled
#                 nt.assert_almost_equal(st.beam.pointing, np.array([0,0,1]), decimal=2)

#             else:
#                 for st in radar.tx + radar.rx:
#                     assert not st.enabled
