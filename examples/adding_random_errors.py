#!/usr/bin/env python

'''
====================
Adding random errors
====================

This short example showcases the use of the :class:`LinearizedCoded<sorts.radar.measurement_errors.linearized_coded.LinearizedCoded>` and :class:`LinearizedIonosphericCoded
<sorts.radar.measurement_errors.linearized_coded.LinearizedIonosphericCoded>` classes to add Ionospheric and random errors to our range and 
range rate estimates of radar measurements using coded pulses. 
'''
import pathlib

import numpy as np
import matplotlib.pyplot as plt

import sorts

eiscat3d = sorts.radars.eiscat3d

try:
    pth = pathlib.Path(__file__).parent / 'data'
except NameError:
    pth = pathlib.Path('.').parent / 'data'

print(f'Caching error calculation data to: {pth}')

# Range errors
err = sorts.measurement_errors.LinearizedCoded(eiscat3d.tx[0], seed=123, cache_folder=pth)

num = 1000
n_bins = 20

# generate 1000 range values and generate random SNR values
ranges = np.linspace(300e3, 350e3, num=num)[::-1]

# generate 1000 SNR measurement values of mean 10.0 and variance 5.0 
snrs = np.random.randn(num)*5.0 + 10**1.0
snrs[snrs < 0.1] = 0.1 # cutoff at 0.1

# compute perturbated range estimates due to SNR fluctuations
perturbed_ranges = err.range(ranges, snrs)

# plot results
fig, axes = plt.subplots(2,2)
axes[0,0].plot(range(num), ranges, n_bins)
axes[0,0].set_ylabel("$N$ [$-$]")
axes[0,0].set_xlabel("$r$ [$m$]")

axes[0,1].hist(10*np.log10(snrs), n_bins)
axes[0,1].set_ylabel("$N$ [$-$]")
axes[0,1].set_xlabel("$SNR$ [$dB$]")

axes[1,0].plot(range(num), perturbed_ranges, n_bins)
axes[1,0].set_ylabel("$N$ [$-$]")
axes[1,0].set_xlabel("$r_{est}$ [$m$]")

axes[1,1].hist(ranges - perturbed_ranges, n_bins)
axes[1,1].set_ylabel("$N$ [$-$]")
axes[1,1].set_xlabel("$r-r_{est}$ [$m$]")

# Range rates and range errors (ionospheric + code errors)
print(f'Caching error calculation data to: {pth}')

err = sorts.measurement_errors.LinearizedCodedIonospheric(eiscat3d.tx[0], seed=123, cache_folder=pth)

num2 = 1000 
ranges = [300e3, 1000e3, 2000e3]

# generate 1000 range eate values
range_rates = np.linspace(0, 10e3, num=num2)

# generate 1000 snr values
snrs_db = np.linspace(14,50,num=num2)
snrs = 10.0**(snrs_db*0.1)

# get standard deviation on range rate estimates due to ionospheric errors
v_std = err.range_rate_std(snrs)
fig, axes = plt.subplots(1,2)

# plot range standard deviation due to ionospheric errors 
for j in range(len(ranges)):
    range_ = np.zeros_like(snrs)
    range_[:] = ranges[j]
    r_std = err.range_std(range_, snrs)
    axes[0].plot(snrs_db, r_std, label=f'Range: {ranges[j]*1e-3:.1f} km')

# plot results
axes[0].legend()
axes[0].set_xlabel("$r$ [$m$]")
axes[0].set_ylabel("$\\sigma_r$ [$dB$]")

axes[1].plot(snrs_db, v_std)
axes[1].set_xlabel("$SNR$ [$dB$]")
axes[1].set_ylabel("$\\sigma_V$ [$m/s$]")
plt.show()