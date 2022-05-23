#!/usr/bin/env python

'''
Noise equivalent diameter
=========================

'''
import numpy as np
import matplotlib.pyplot as plt

import sorts
from sorts import radars

cmap_name = 'vibrant'
color_cycle = sorts.plotting.colors.get_cycle(cmap_name)

ranges = 10**(np.linspace(2, 5, 1000))
ranges = ranges*1e3  # km -> m
snr = 1.0

data = zip(
    [
        radars.eiscat_uhf.tx[0], 
        radars.eiscat_esr.tx[0],
        radars.eiscat_esr.tx[1],
    ],
    [
        radars.eiscat_uhf.rx[0], 
        radars.eiscat_esr.rx[0],
        radars.eiscat_esr.rx[1],
    ],
    [   
        'EISCAT UHF', 
        'EISCAT Svalbard 32m radar',
        'EISCAT Svalbard 42m radar',
    ],
)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.set_prop_cycle(color_cycle)

for tx, rx, label in data:

    G0_tx = tx.beam.gain(tx.beam.pointing)
    G0_rx = rx.beam.gain(rx.beam.pointing)
    
    diameters = sorts.signals.hard_target_diameter(
        G0_tx,
        G0_rx,
        tx.beam.wavelength,
        tx.power,
        ranges, 
        ranges,
        snr, 
        bandwidth = tx.coh_int_bandwidth,
        rx_noise_temp = rx.noise,
        radar_albedo = 1.0,
    )

    ax.loglog(ranges*1e-3, diameters*1e2, label=label)

ax.set_xlabel('Range [km]')
ax.set_ylabel('System noise equivalent diameter [cm]')
ax.set_title('Radar sensitivity comparison')
ax.legend()

plt.show()
