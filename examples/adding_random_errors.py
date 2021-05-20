#!/usr/bin/env python

'''
Adding random errors
================================

'''
import pathlib

import numpy as np
import matplotlib.pyplot as plt

import sorts.errors as errors
import sorts
eiscat3d = sorts.radars.eiscat3d


try:
    pth = pathlib.Path(__file__).parent / 'data'
except NameError:
    pth = pathlib.Path('.').parent / 'data'



print(f'Caching error calculation data to: {pth}')
err = errors.LinearizedCoded(eiscat3d.tx[0], seed=123, cache_folder=pth)

num = 1000

ranges = np.linspace(300e3, 350e3, num=num)[::-1]
snrs = np.random.randn(num)*5 + 10**1.0
snrs[snrs < 0.1] = 0.1

perturbed_ranges = err.range(ranges, snrs)

fig, axes = plt.subplots(2,2)
axes[0,0].plot(range(num), ranges, 100)
axes[0,1].hist(10*np.log10(snrs), 100)
axes[1,0].plot(range(num), perturbed_ranges, 100)
axes[1,1].hist(ranges - perturbed_ranges, 100)


print(f'Caching error calculation data to: {pth}')


eiscat3d.tx[0].bandwidth = 10000.0
err = errors.LinearizedCodedIonospheric(eiscat3d.tx[0], seed=123, cache_folder=pth)

num2 = 400

ranges = [300e3, 1000e3, 2000e3]

range_rates = np.linspace(0, 10e3, num=num2)
snrs_db = np.linspace(14,50,num=num2)
snrs = 10.0**(snrs_db*0.1)

v_std = err.range_rate_std(snrs)

fig, axes = plt.subplots(1,2)

for j in range(len(ranges)):
    range_ = np.zeros_like(snrs)
    range_[:] = ranges[j]
    r_std = err.range_std(range_, snrs)
    axes[0].plot(snrs_db, r_std, label=f'Range: {ranges[j]*1e-3:.1f} km')

axes[0].legend()
axes[1].plot(snrs_db, v_std)


plt.draw()