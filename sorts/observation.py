#!/usr/bin/env python

'''

'''

import numpy as np
import pyorb
import pyant

#Local import
from .signals import hard_target_snr



def calculate_passes_snr(passes, controller, diameter):
    for txi in range(len(passes)):
        for rxi in range(len(passes[txi])):
            for ps in passes[txi][rxi]:
                rgen = controller(ps.t)
                snr = np.empty((len(ps.t),), dtype=np.float64)
                enus = ps.enu
                ranges = ps.range()

                for ti, radar in enumerate(rgen):
                    if radar.tx[txi].enabled and radar.rx[rxi].enabled:
                        snr[ti] = hard_target_snr(
                            radar.tx[txi].beam.gain(enus[0][:,ti]),
                            radar.rx[rxi].beam.gain(enus[1][:,ti]),
                            radar.rx[rxi].wavelength,
                            radar.tx[txi].power,
                            ranges[0][ti],
                            ranges[1][ti],
                            diameter_m=diameter,
                            bandwidth=radar.tx[txi].coh_int_bandwidth,
                            rx_noise_temp=radar.rx[rxi].noise,
                        )
                    else:
                        snr[ti] = np.nan

                ps.snr = snr