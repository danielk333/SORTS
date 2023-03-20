#!/usr/bin/env python

'''
HPBW calculation
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

data = zip(
    [
        radars.eiscat_uhf.tx[0], 
        radars.eiscat_esr.tx[0],
        radars.eiscat_esr.tx[1],
    ],
    [   
        'EISCAT UHF', 
        'EISCAT Svalbard 32m radar',
        'EISCAT Svalbard 42m radar',
    ],
)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.set_prop_cycle(color_cycle)

el = np.linspace(90, 80, 1000)

for tx, label in data:

    tx.beam.sph_point(azimuth=0, elevation=90)
    G0_tx = tx.beam.sph_gain(azimuth=0, elevation=el)

    G0_max = G0_tx[0]
    minind = np.argmin(np.abs(G0_tx - G0_max*0.5))
    
    ax.plot(el, 10*np.log10(G0_tx), label=label)
    ax.plot(el[minind], 10*np.log10(G0_tx[minind]), 'or')

    print(label, (90 - el[minind])*2, ' deg')

ax.set_xlabel('Elevation [deg]')
ax.set_ylabel('Tx gain [dB]')
ax.set_title('Radar beam comparison')
ax.legend()

plt.show()
