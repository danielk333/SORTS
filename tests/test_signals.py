import unittest
import numpy as np
import numpy.testing as nt

import h5py
import scipy.constants

import sorts

class TestSignals(unittest.TestCase):
    def test_hard_target_snr_scaling(self):
        wavelength = 1.287 # m
        separator = wavelength/(np.pi*np.sqrt(3.0)) # = 0.236


        # test regime #1 : 
        diameter_from = 0.147
        diameter_to = 0.183

        scale = sorts.signals.hard_target_snr_scaling(diameter_from, diameter_to, wavelength)
        scale_th = (diameter_to/diameter_from)**6.0
        nt.assert_almost_equal(scale, scale_th)


        # test regime #2 : 
        diameter_from = 0.143
        diameter_to = 0.247 

        scale = sorts.signals.hard_target_snr_scaling(diameter_from, diameter_to, wavelength)
        scale_th = ((wavelength**2.0*diameter_to)/(3.0*np.pi**2.0*diameter_from**3.0))**2.0
        nt.assert_almost_equal(scale, scale_th)


        # test regime #3 : 
        diameter_from = 0.546
        diameter_to = 0.139

        scale = sorts.signals.hard_target_snr_scaling(diameter_from, diameter_to, wavelength)
        scale_th = ((3.0*np.pi**2.0*diameter_to**3.0)/(wavelength**2.0*diameter_from))**2.0
        nt.assert_almost_equal(scale, scale_th)


        # test regime #4 : 
        diameter_from = 0.546
        diameter_to = 0.785

        scale = sorts.signals.hard_target_snr_scaling(diameter_from, diameter_to, wavelength)
        scale_th = (diameter_to/diameter_from)**6.0
        nt.assert_almost_equal(scale, scale_th)




    def test_hard_target_rcs(self):
        pass

    def test_hard_target_snr(self):
        pass

    def test_hard_target_diameter(self):
        pass

    def test_incoherent_snr(self):
        pass

    def test_doppler_spread_hard_target_snr(self):
        pass
