#!/usr/bin/env python

"""This module is used to define the radar network configuration.

"""
import numpy as np
import scipy.constants


def hard_target_snr_scaling(diameter_from, diameter_to, wavelength):
    """Determine the scaling constant `c` to convert the SNR generated by a
    `diameter_from` target to a `diameter_to` target. I.e. by
    `c SNR(diameter_from) = SNR(diameter_to)`.

    :param float diameter_from: object diameter of snr to convert from (meters)
    :param float diameter_to: object diameter of snr to convert to (meters)
    :param float wavelength: radar wavelength (meters)
    """

    sep_const = wavelength / (np.pi * np.sqrt(3.0))
    if diameter_from < sep_const and diameter_to < sep_const:
        scale = diameter_to**6 / diameter_from**6
    elif diameter_from < sep_const and diameter_to >= sep_const:
        scale = (wavelength**4 * diameter_to**2) / (9 * np.pi**4 * diameter_from**6)
    elif diameter_from >= sep_const and diameter_to < sep_const:
        scale = (9 * np.pi**4 * diameter_to**6) / (wavelength**4 * diameter_from**2)
    else:
        scale = diameter_to**6 / diameter_from**6

    return scale


def hard_target_rcs(wavelength, diameter):
    """Determine the radar cross section for a hard target.
    Assume a smooth transition between Rayleigh and optical scattering.
    Ignore Mie regime and use either optical or Rayleigh scatter.

    :param float wavelength: radar wavelength (meters)
    :param float/numpy.ndarray diameter: diameter in meters of the objects.
    """
    is_rayleigh = diameter < wavelength / (np.pi * np.sqrt(3.0))
    is_optical = diameter >= wavelength / (np.pi * np.sqrt(3.0))
    optical_rcs = np.pi * diameter**2.0 / 4.0
    rayleigh_rcs = diameter**6.0 * 9.0 * np.pi**5 / (4.0 * wavelength**4)
    rcs = is_rayleigh * rayleigh_rcs + is_optical * optical_rcs
    return rcs


def hard_target_snr(
    gain_tx,
    gain_rx,
    wavelength,
    power_tx,
    range_tx_m,
    range_rx_m,
    diameter=0.01,
    bandwidth=10,
    rx_noise_temp=150.0,
    radar_albedo=1.0,
):
    """
    Determine the signal-to-noise ratio (energy-to-noise) ratio for a hard target.
    Assume a smooth transition between Rayleigh and optical scattering.
    Ignore Mie regime and use either optical or Rayleigh scatter.

    :param float/numpy.ndarray gain_tx: transmit antenna gain, linear
    :param float/numpy.ndarray gain_rx: receiver antenna gain, linear
    :param float wavelength: radar wavelength (meters)
    :param float power_tx: transmit power (W)
    :param float/numpy.ndarray range_tx_m: range from transmitter to target (meters)
    :param float/numpy.ndarray range_rx_m: range from target to receiver (meters)
    :param float diameter: object diameter (meters)
    :param float bandwidth: effective receiver noise bandwidth
    :param float rx_noise_temp: receiver noise temperature (K)
    :return: signal-to-noise ratio
    :rtype: float/numpy.ndarray


    **Reference:** Markkanen et.al., 1999

    """

    is_rayleigh = diameter < wavelength / (np.pi * np.sqrt(3.0))
    is_optical = diameter >= wavelength / (np.pi * np.sqrt(3.0))
    rayleigh_power = (
        9.0
        * power_tx
        * (
            ((gain_tx * gain_rx) * (np.pi**2.0) * (diameter**6.0))
            / (256.0 * (wavelength**2.0) * (range_rx_m**2.0 * range_tx_m**2.0))
        )
    )
    optical_power = (
        power_tx
        * (((gain_tx * gain_rx) * (wavelength**2.0) * (diameter**2.0)))
        / (256.0 * (np.pi**2) * (range_rx_m**2.0 * range_tx_m**2.0))
    )
    rx_noise = scipy.constants.k * rx_noise_temp * bandwidth
    snr = ((is_rayleigh) * rayleigh_power + (is_optical) * optical_power) * radar_albedo / rx_noise
    return snr


def hard_target_diameter(
    gain_tx,
    gain_rx,
    wavelength,
    power_tx,
    range_tx_m,
    range_rx_m,
    snr,
    bandwidth=10,
    rx_noise_temp=150.0,
    radar_albedo=1.0,
):
    """
    Determine the diamtere for a hard target based on the signal-to-noise ratio (energy-to-noise).
    Assume a smooth transition between Rayleigh and optical scattering.
    Ignore Mie regime and use either optical or Rayleigh scatter.

    :param float/numpy.ndarray gain_tx: transmit antenna gain, linear
    :param float/numpy.ndarray gain_rx: receiver antenna gain, linear
    :param float wavelength: radar wavelength (meters)
    :param float power_tx: transmit power (W)
    :param float/numpy.ndarray range_tx_m: range from transmitter to target (meters)
    :param float/numpy.ndarray range_rx_m: range from target to receiver (meters)
    :param float snr: object signal to noise ratio (1)
    :param float bandwidth: effective receiver noise bandwidth
    :param float rx_noise_temp: receiver noise temperature (K)
    :return: diameter (meters)
    :rtype: float/numpy.ndarray


    **Reference:** Markkanen et.al., 1999

    """

    rx_noise = scipy.constants.k * rx_noise_temp * bandwidth
    power = snr * rx_noise / radar_albedo

    diameter_rayleigh = (
        256.0
        * (wavelength**2.0)
        * (range_rx_m**2.0 * range_tx_m**2.0)
        * power
        / (9.0 * power_tx * (gain_tx * gain_rx) * (np.pi**2.0))
    ) ** (1.0 / 6.0)
    diameter_optical = np.sqrt(
        256.0
        * (np.pi**2)
        * (range_rx_m**2.0 * range_tx_m**2.0)
        * power
        / (power_tx * (gain_tx * gain_rx) * (wavelength**2.0))
    )

    separatrix = wavelength / (np.pi * np.sqrt(3.0))

    is_rayleigh = np.logical_and(diameter_rayleigh < separatrix, diameter_optical < separatrix)
    is_optical = np.logical_and(diameter_rayleigh >= separatrix, diameter_optical >= separatrix)

    diameter = (is_rayleigh) * diameter_rayleigh + (is_optical) * diameter_optical

    return diameter


def incoherent_snr(p_s, p_n, epsilon=0.05, B=10.0, t_incoh=3600.0):
    """Calculate the incoherent SNR based on ????

    TODO: Finish docstring
    TODO: generalize theory??
    TODO: Juha knows
    """
    snr = p_s / p_n
    t_epsilon = ((p_s + p_n) ** 2.0) / (epsilon**2.0 * p_s**2.0 * B)

    K = t_incoh * B
    delta_pn = p_n / np.sqrt(K)
    snr_incoh = p_s / delta_pn

    return snr, snr_incoh, t_epsilon


def doppler_spread_hard_target_snr(
    t_obs,
    spin_period,
    gain_tx,
    gain_rx,
    wavelength,
    power_tx,
    range_tx_m,
    range_rx_m,
    duty_cycle=0.25,
    diameter=10.0,
    bandwidth=10,
    rx_noise_temp=150.0,
    radar_albedo=0.1,
):
    """
    t_obs = observation duration

    #TODO: Double check the "bandwidth" parameter to see that it is actually
    # defined and used correctly

    returns:
    snr - signal to noise ratio using coherent integration, when doing object discovery with a
          limited coherent integration duration and no incoherent integration
    snr_incoh - the signal to noise ratio using incoherent integration, when using a priori
                orbital elements to assist in coherent integration and incoherent integration.
                coherent integration length is determined by t_obs (seconds)
    """

    doppler_bandwidth = 4 * np.pi * diameter / (wavelength * spin_period)

    # for serendipitous discovery
    detection_bandwidth = np.max([doppler_bandwidth, bandwidth * duty_cycle, 1.0 / t_obs])

    # for detection with a periori know orbit
    # the bandwidth cannot be smaller than permitted by the observation duration.
    base_int_bandwidth = np.max([doppler_bandwidth, (1.0 / t_obs)])

    # standard one segment rx noise
    rx_noise = scipy.constants.k * rx_noise_temp * bandwidth

    # effective noise power when using just coherent integration
    p_n0 = scipy.constants.k * rx_noise_temp * detection_bandwidth / duty_cycle

    # effective noise power when doing incoherent integration and using a good
    # a priori orbital elements
    p_n1 = scipy.constants.k * rx_noise_temp * base_int_bandwidth / duty_cycle

    h_snr = hard_target_snr(
        gain_tx=gain_tx,
        gain_rx=gain_rx,
        wavelength=wavelength,
        power_tx=power_tx,
        range_tx_m=range_tx_m,
        range_rx_m=range_rx_m,
        diameter=diameter,
        bandwidth=bandwidth,
        rx_noise_temp=rx_noise_temp,
        radar_albedo=radar_albedo,
    )
    p_s = h_snr * rx_noise

    snr, snr_incoh, te = incoherent_snr(p_s, p_n1, B=base_int_bandwidth, t_incoh=t_obs)

    snr_coh = p_s / p_n0
    return snr_coh, snr_incoh
