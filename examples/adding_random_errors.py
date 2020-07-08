#!/usr/bin/env python

'''

'''
import pathlib

import numpy as np
import matplotlib.pyplot as plt

import sorts.errors as errors
from sorts.radar import eiscat3d

pth = pathlib.Path(__file__).parent / 'data'

print(f'Caching error calculation data to: {pth}')

err = errors.LinearizedCoded(eiscat3d.tx[0], seed=123, cache_folder=pth, diameter=0.1)

my_ranges = np.linspace(300e3, 400e3, num=1000)
my_snrs = np.random.randn(1000)*10 + 10**3.0

perturbed_ranges = err.range(my_ranges, my_snrs)

fig, axes = plt.subplots(2,2)
axes[0,0].hist(my_ranges, 100)
axes[0,1].hist(my_snrs, 100)
axes[1,0].hist(perturbed_ranges, 100)
axes[1,1].hist(my_ranges - perturbed_ranges, 100)

plt.show()