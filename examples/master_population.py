#!/usr/bin/env python

'''
Loading the MASTER population
================================

'''
import configparser
import pathlib

import matplotlib.pyplot as plt

from sorts import plotting
from sorts.population import master_catalog, master_catalog_factor


try:
    base_pth = pathlib.Path(__file__).parents[1].resolve()
except NameError:
    base_pth = pathlib.Path('.').parents[1].resolve()

config = configparser.ConfigParser(interpolation=None)
config.read([base_pth / 'example_config.conf'])
master_path = pathlib.Path(config.get('master_population.py', 'master_catalog'))

if not master_path.is_absolute():
    master_path = base_pth / master_path.relative_to('.')

pop = master_catalog(master_path)
pop_factor = master_catalog_factor(pop, treshhold = 0.1)

plotting.orbits(
    pop.get_states(named=False),
    title =  "Orbit distribution of Master catalog",
    axis_labels = 'earth-orbit',
)

print(f'Master: {len(pop)} | Master - factor (diameter > 0.1 m): {len(pop_factor)}')

plt.show()
