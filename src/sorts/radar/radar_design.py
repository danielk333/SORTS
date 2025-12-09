import numpy as np
import scipy.constants
import pyant

from ..signals import rayleigh_separatrix_range, hard_target_diameter

T_REF = 290  # K


def estimate_radar_parameters(
    frequency,
    antenna_num,
    antenna_spacing_lambda,
    antenna_efficiency,
    antenna_input_power,
    thermal_load,
    noise_figure_db,
    amplifier_gain_db,
    insertion_loss_db,
    duty_cycle=0.1,
    computation_power_draw_scaling=2,
    t_sky=30,
    reference_snr_db=10.0,
    reference_ranges=1000e3,
    coherent_integration_time=0.02,
    bandwidth_limit_ratio=10,
):
    """
    todo

    ref: kildal book, other books
    """
    lam = scipy.constants.c / frequency

    tx_power = antenna_num * antenna_input_power

    thermal_production = antenna_num * thermal_load * duty_cycle
    radar_power_draw = antenna_num * (thermal_load + antenna_input_power) * duty_cycle
    total_power_draw = computation_power_draw_scaling * radar_power_draw
    # avg antenna package size
    size = antenna_spacing_lambda * lam

    # approx the circle radius of all antenna package area
    # A_tot = size ** 2 * antenna_num
    # r = np.sqrt(A_tot/np.pi) = np.sqrt(antenna_num / np.pi) * size
    radius = np.sqrt(antenna_num / np.pi) * size

    # just based on the aperture
    directivity = 4 * (np.pi * radius) ** 2 / lam**2

    # the peak gain, this is an approximation in the main lobe since
    # the gain will no longer be "directive gain" when divided by
    # the antenna_efficiency as the antenna gain would also modify
    # the side lobes and such
    peak_gain = antenna_efficiency * directivity

    beam = pyant.models.Gaussian(
        peak_gain=peak_gain,
    )
    param = pyant.models.GaussianParams(
        pointing=np.array([0, 0, 1], dtype=np.float64),
        normal_pointing=np.array([0, 0, 1], dtype=np.float64),
        frequency=frequency,
        radius=radius,
        beam_width_scaling=1.22,
    )
    half_power_angle = np.degrees(param.beam_width_scaling * lam / (radius * 2))

    # the radar equation will give us the power induced in the system after antenna reception, now
    # to turn this into a SNR, we can assume that this power stays constant and instead just modify
    # the noise power accordingly as a function of all the effects of the system

    # assume we have a noise of T_0 at the digitisers
    # doing mean over the different antenna signals will keep T_0 constant but increase signal
    # strength in a coherent fashion, this is covered by the gain function (usually done trough the
    # "array factor"), hence we can keep T_0 constant trough the lossless combiner.

    # this noise is pretty much only the LNA from the system side and the sky noise from the
    # external side, the noise is additive so the following equation should suffice

    # The noise factor is the ratio of input SNR to output SNR, i.e. the noise introduced by the
    # component. The noise figure is the decibel equivalent version of the noise factor.
    #
    # F = 1 + T_noise / T_ref
    # F = SNR_in / SNR_out
    amplifier_gain = 10 ** (amplifier_gain_db / 10)
    noise_figure = 10 ** (noise_figure_db / 10)
    t_noise = amplifier_gain * t_sky + (noise_figure - 1) * T_REF
    t_noise = t_noise / bandwidth_limit_ratio

    # Then coherent integration will happen, and here, whatever our sampling frequency is gets
    # removed by the fact that we are coherently integrating, and all that matters is the coherent
    # integration time and the ratio of bandwidth reduction by any supposed filtering
    rayleigh_separatrix = rayleigh_separatrix_range(
        beam.peak_gain,
        beam.peak_gain,
        lam,
        tx_power,
        10 ** (reference_snr_db / 10),
        coherent_integration_time=coherent_integration_time,
        effective_noise_temperature=t_noise,
        radar_albedo=1.0,
    )
    effective_bandwidth = 1 / coherent_integration_time
    minimum_diameter = [
        hard_target_diameter(
            beam.peak_gain,
            beam.peak_gain,
            lam,
            tx_power,
            reference_range,
            reference_range,
            10 ** (reference_snr_db / 10),
            bandwidth=effective_bandwidth,
            rx_noise_temp=t_noise,
            radar_albedo=1.0,
        )
        for reference_range in reference_ranges
    ]
    data = dict(
        rayleigh_separatrix=rayleigh_separatrix,
        minimum_diameter=minimum_diameter,
        t_noise=t_noise,
        tx_power=tx_power,
        thermal_production=thermal_production,
        radar_power_draw=radar_power_draw,
        total_power_draw=total_power_draw,
        beam=beam,
        beam_parameters=param,
        half_power_angle=half_power_angle,
        effective_bandwidth=effective_bandwidth,
    )
    return data
