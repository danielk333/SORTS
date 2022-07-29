import unittest
import numpy as np
import numpy.testing as nt

import h5py
import scipy.constants

import sorts

class TestSignals(unittest.TestCase):
    def test_hard_target_snr_scaling(self):
        wavelength = 1.287 # m
        separator = wavelength/(np.pi*7.11**(1.0/4.0)) # = 0.236


        # test regime #1 : 
        diameter_from = separator*0.1
        diameter_to = separator*0.1

        scale = sorts.signals.hard_target_snr_scaling(diameter_from, diameter_to, wavelength)
        scale_th = (diameter_to/diameter_from)**6.0
        nt.assert_almost_equal(scale, scale_th)


        # test regime #2 : 
        diameter_from = separator*0.1
        diameter_to = separator*10.0

        scale = sorts.signals.hard_target_snr_scaling(diameter_from, diameter_to, wavelength)
        scale_th = ((wavelength**2.0*diameter_to)/(3.0*np.pi**2.0*diameter_from**3.0))**2.0
        nt.assert_almost_equal(scale, scale_th)


        # test regime #3 : 
        diameter_from = separator*10.0
        diameter_to = separator*0.1

        scale = sorts.signals.hard_target_snr_scaling(diameter_from, diameter_to, wavelength)
        scale_th = ((3.0*np.pi**2.0*diameter_to**3.0)/(wavelength**2.0*diameter_from))**2.0
        nt.assert_almost_equal(scale, scale_th)


        # test regime #4 : 
        diameter_from = separator*10.0
        diameter_to = separator*10.0

        scale = sorts.signals.hard_target_snr_scaling(diameter_from, diameter_to, wavelength)
        scale_th = (diameter_to/diameter_from)**6.0
        nt.assert_almost_equal(scale, scale_th)



    def test_hard_target_rcs(self):
        """ Tests signals.py hard_target_rcs implementation

        Tests the two regimes (Rayleigh/Optical) of the hard target Radar Cross-Section function
        """
        wavelength = 1.287 # m
        separator = wavelength/(np.pi*7.11**(1.0/4.0)) # = 0.236

        diameters = np.ndarray((3,), dtype=float)
        diameters[0] = 10*separator     # test regime #1 (Optical)
        diameters[1] = separator        # test regime intersection
        diameters[2] = 0.1*separator    # test regime #2 (Rayleigh)

        optical_rcs = np.pi*diameters**2.0/4.0
        rayleigh_rcs = np.pi*diameters**2.0*7.11/4.0*(np.pi*diameters/wavelength)**4

        rcs_th = np.array([*optical_rcs[0:2], rayleigh_rcs[2]])
        rcs = sorts.signals.hard_target_rcs(wavelength, diameters)

        nt.assert_almost_equal(rcs, rcs_th, decimal=6)

        # test boundary condition
        nt.assert_almost_equal(rcs[1], optical_rcs[1])
        nt.assert_almost_equal(rcs[1], rayleigh_rcs[1])



    def test_hard_target_snr(self):
        wavelength = 1.287 # m
        separator = wavelength/(np.pi*np.sqrt(3.0))

        # antenna properties
        gain_tx = 15.0
        gain_rx = 12.5
        power_tx = 10.4e3

        # distance from target
        range_rx_m = 508e3
        range_tx_m = 1200e3

        # compute SNR for multiple diameters
        diameters = np.ndarray((3,), dtype=float)
        diameters[0] = separator* 0.5 # test regime #1 (Optical)
        diameters[1] = separator* 1.0 # test regime intersection
        diameters[2] = separator* 5.0 # test regime #2 (Rayleigh)

        snr = np.ndarray((3,), dtype=float)
        for i in range(len(diameters)):
            snr[i] = sorts.signals.hard_target_snr(
                gain_tx, 
                gain_rx, 
                wavelength, 
                power_tx, 
                range_tx_m, 
                range_rx_m, 
                bandwidth=10,
                rx_noise_temp=150.0,
                radar_albedo=1.0,
                diameter=diameters[i],
            )
    
        # theoretical SNR        
        snr_th = np.array([1.45192324e-04, 9.29230871e-03, 2.32307718e-01])

        # test equality
        nt.assert_almost_equal(snr, snr_th, decimal=6)



    def test_hard_target_diameter(self):
        wavelength = 1.287 # m
        separator = wavelength/(np.pi*np.sqrt(3.0))

        # antenna properties
        gain_tx = 15.0
        gain_rx = 12.5
        power_tx = 10.4e3

        # distance from target
        range_rx_m = 508e3
        range_tx_m = 1200e3

        diameter_th = np.ndarray((3,), dtype=float)
        diameter_th[0] = separator* 0.5 # test regime #1 (Optical)
        diameter_th[1] = separator* 1.0 # test regime intersection
        diameter_th[2] = separator* 5.0 # test regime #2 (Rayleigh)

        # theoretical SNRs for the given diameters
        snr_th = np.array([1.45192324e-04, 9.29230871e-03, 2.32307718e-01])

        diameter = np.ndarray((3,), dtype=float)
        for i in range(len(diameter)):
            diameter[i] = sorts.signals.hard_target_diameter(
                gain_tx, 
                gain_rx,
                wavelength,
                power_tx,
                range_tx_m, 
                range_rx_m,
                snr_th[i], 
                bandwidth=10,
                rx_noise_temp=150.0,
                radar_albedo=1.0)

        nt.assert_almost_equal(diameter, diameter_th, decimal=6)



    def test_incoherent_snr(self):
        # antenna properties
        gain_tx = 15.0
        gain_rx = 12.5
        signal_power = 10.5e3

        # distance from target
        range_rx_m = 508e3
        range_tx_m = 1200e3

        epsilon = 0.05
        bandwidth = 10.0
        incoherent_integration_time = 3600.0

        # theoretical SNR
        snr_th = np.array([1.45192324e-04, 9.29230871e-03, 2.32307718e-01])

        for i in range(len(snr_th)):
            noise_power = signal_power/snr_th[i]

            # compute theoretical values
            minimal_observation_time_th = ((signal_power + noise_power)**2.0)/(epsilon**2.0*signal_power**2.0*bandwidth)
            snr_incoh_th = snr_th[i]*np.sqrt(int(incoherent_integration_time * bandwidth))
            
            # compute using sorts implementation :
            snr, snr_incoh, minimal_observation_time = sorts.signals.incoherent_snr(
                signal_power, 
                noise_power, 
                epsilon=epsilon, 
                bandwidth=bandwidth, 
                incoherent_integration_time=incoherent_integration_time,
            )

            nt.assert_almost_equal(snr, snr_th[i], decimal=6)
            nt.assert_almost_equal(snr_incoh, snr_incoh_th, decimal=6)
            nt.assert_almost_equal(minimal_observation_time, minimal_observation_time_th, decimal=6)



    def test_doppler_spread_hard_target_snr(self):
        wavelength = 1.287 # m
        separator = wavelength/(np.pi*np.sqrt(3.0))

        # antenna properties
        gain_tx = 15.0
        gain_rx = 12.5
        power_tx = 10.4e3

        # distance from target
        range_rx_m = 508e3
        range_tx_m = 1200e3

        # other parameters
        bandwidth = 10.0
        rx_noise_temp = 150.0
        radar_albedo = 1.0

        spin_period = 2.5
        t_obs = 1000.0
        duty_cycle = 0.45


        # compute SNR for multiple diameters
        diameters = np.ndarray((3,), dtype=float)
        diameters[0] = separator* 0.5 # test regime #1 (Optical)
        diameters[1] = separator* 1.0 # test regime intersection
        diameters[2] = separator* 5.0 # test regime #2 (Rayleigh)

        # theoretical SNR        
        snr_th = np.array([1.45192324e-04, 9.29230871e-03, 2.32307718e-01])


        for i in range(len(diameters)):
            # compute the bandwidth of the doppler shifted RADAR echo
            doppler_bandwidth = 4.0*np.pi*diameters[i]/(wavelength*spin_period)
            detection_bandwidth = np.max([doppler_bandwidth, bandwidth*duty_cycle, 1.0/t_obs])
            base_int_bandwidth = np.max([doppler_bandwidth, 1.0/t_obs])

            rx_noise = scipy.constants.k*rx_noise_temp*bandwidth
            signal_power = snr_th[i]*rx_noise

            # compute coherent SNR
            coh_noise_power = scipy.constants.k * rx_noise_temp * detection_bandwidth/duty_cycle
            snr_coh_th = signal_power/coh_noise_power

            # compute incoherent SNR
            incoh_noise_power = scipy.constants.k * rx_noise_temp * base_int_bandwidth/duty_cycle
            snr_incoh_th = signal_power/incoh_noise_power*np.sqrt(int(t_obs * base_int_bandwidth))


            # compute SNR using sorts
            snr_coh, snr_incoh = sorts.signals.doppler_spread_hard_target_snr(
                t_obs, 
                spin_period, 
                gain_tx, 
                gain_rx,
                wavelength,
                power_tx,
                range_tx_m, 
                range_rx_m,
                duty_cycle=duty_cycle,
                diameter=diameters[i], 
                bandwidth=bandwidth,
                rx_noise_temp=rx_noise_temp,
                radar_albedo=radar_albedo,
            )


            # test equality between sorts and theoretical values
            nt.assert_almost_equal(snr_incoh, snr_incoh_th, decimal=6)
            nt.assert_almost_equal(snr_coh, snr_coh_th, decimal=6)

