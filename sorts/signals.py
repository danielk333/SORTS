#!/usr/bin/env python

'''This module is used to define the radar network configuration.

'''
import numpy as np
import scipy.constants


def hard_target_snr(gain_tx, gain_rx,
                    wavelength_m, power_tx,
                    range_tx_m, range_rx_m,
                    diameter_m=0.01, bandwidth=10,
                    rx_noise_temp=150.0):
    '''
    Determine the signal-to-noise ratio (energy-to-noise) ratio for a hard target.
    Assume a smooth transition between Rayleigh and optical scattering. 
    Ignore Mie regime and use either optical or Rayleigh scatter.

    :param float/numpy.ndarray gain_tx: transmit antenna gain, linear
    :param float/numpy.ndarray gain_rx: receiver antenna gain, linear
    :param float wavelength_m: radar wavelength (meters)
    :param float power_tx: transmit power (W)
    :param float/numpy.ndarray range_tx_m: range from transmitter to target (meters)
    :param float/numpy.ndarray range_rx_m: range from target to receiver (meters)
    :param float diameter_m: object diameter (meters)
    :param float bandwidth: effective receiver noise bandwidth for incoherent integration (tx_len*n_ipp/sample_rate)
    :param float rx_noise_temp: receiver noise temperature (K)
    :return: signal-to-noise ratio
    :rtype: float/numpy.ndarray


    **Reference:** Markkanen et.al., 1999
    
    '''

    is_rayleigh = diameter_m < wavelength_m/(np.pi*np.sqrt(3.0))
    is_optical = diameter_m >= wavelength_m/(np.pi*np.sqrt(3.0))
    rayleigh_power = (9.0*power_tx*(((gain_tx*gain_rx)*(np.pi**2.0)*(diameter_m**6.0))/(256.0*(wavelength_m**2.0)*(range_rx_m**2.0*range_tx_m**2.0))))
    optical_power = (power_tx*(((gain_tx*gain_rx)*(wavelength_m**2.0)*(diameter_m**2.0)))/(256.0*(np.pi**2)*(range_rx_m**2.0*range_tx_m**2.0)))
    rx_noise = scipy.constants.k*rx_noise_temp*bandwidth
    snr = ((is_rayleigh)*rayleigh_power + (is_optical)*optical_power)/rx_noise
    return snr

