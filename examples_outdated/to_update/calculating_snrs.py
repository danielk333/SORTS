#!/usr/bin/env python

"""
Calculating SNRs
================================

"""
import numpy as np
import sorts

radar = sorts.get_radar("eiscat3d", "stage1-array")

k0 = np.array([0, 0, 1])
radar.tx[0].beam.point(k0)
radar.rx[0].beam.point(k0)

snr_coh, snr_incoh = sorts.signals.doppler_spread_hard_target_snr(
    3600.0,
    spin_period=500.0,
    gain_tx=radar.tx[0].beam.gain(k0),
    gain_rx=radar.rx[0].beam.gain(k0),
    wavelength=radar.tx[0].wavelength[0],
    power_tx=radar.tx[0].power,
    range_tx_m=300000e3,
    range_rx_m=300000e3,
    duty_cycle=0.25,
    bandwidth=10,
    rx_noise_temp=150.0,
    diameter=150.0,
    radar_albedo=0.1,
)

print(f"Spin period : {500.0:.2f} s")
print(f"Radar albedo: {0.1:.2f}")
print(f"Receiver noise temperature: {150.0:.2f} K")

print(f"Coherent   SNR [bandwidth   = 10 Hz ]: {np.log10(snr_coh)*10:.2f}")
print(f"Incoherent SNR [observation = 3600 s]: {np.log10(snr_incoh)*10:.2f}")
