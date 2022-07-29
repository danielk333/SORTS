#!/usr/bin/env python

'''
================
Calculating SNRs
================

Showcases the ``sorts.signals`` module used for the computation of 
hard target incoherent and coherent Signal-to-noise ratio.
'''

import numpy as np
import matplotlib.pyplot as plt
import sorts

# intitializes the radar
radar = sorts.radars.eiscat3d

# point stations towards local vertical
k0 = np.array([0,0,1])
radar.tx[0].beam.point(k0)
radar.rx[0].beam.point(k0)

# compute incoherent and coherent SNR
snr_coh, snr_incoh = sorts.signals.doppler_spread_hard_target_snr(
    3600.0, 
    gain_tx = radar.tx[0].beam.gain(k0),
    gain_rx = radar.rx[0].beam.gain(k0),
    wavelength = radar.tx[0].wavelength,
    power_tx = radar.tx[0].power,
    range_tx_m = 300000e3, 
    range_rx_m = 300000e3,
    duty_cycle=0.25,
    bandwidth=10,
    rx_noise_temp=150.0,
    diameter=150.0,
    spin_period=500.0,
    radar_albedo=0.1,
)

# print computation results
print(f'Spin period : {500.0:.2f} s')
print(f'Radar albedo: {0.1:.2f}')
print(f'Receiver noise temperature: {150.0:.2f} K')

print(f'Coherent   SNR [bandwidth   = 10 Hz ]: {np.log10(snr_coh)*10:.2f}')
print(f'Incoherent SNR [observation = 3600 s]: {np.log10(snr_incoh)*10:.2f}')