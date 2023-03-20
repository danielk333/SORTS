#!/usr/bin/env python

'''
Loading the MASTER population
================================

'''
import configparser
import pathlib

import matplotlib.pyplot as plt

import sorts

path = pathlib.Path('/home/danielk/data/master_2009/celn_20090501_00.sim')

pop = sorts.population.master_catalog(path)
pop_factor = sorts.population.master_catalog_factor(pop, treshhold = 0.1)

sorts.plotting.kepler_scatter(
    pop.get_states(named=False),
    title =  "Orbit distribution of Master catalog",
    axis_labels = 'earth-orbit',
)

print(f'Master: {len(pop)} | Master - factor (diameter > 0.1 m): {len(pop_factor)}')

plt.show()
