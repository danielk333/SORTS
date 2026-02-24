#!/usr/bin/env python

'''

'''
import matplotlib.pyplot as plt

from sorts import plotting
from sorts.population import master_catalog, master_catalog_factor

#THIS NEEDS TO BE REPLACED WITH PATH TO _YOUR_ FILE
path = '/homes/tom/Documents/NORCE/AO9884_RPPD/master/celn_20090501_00.sim'

pop = master_catalog(path)
pop_factor = master_catalog_factor(pop, treshhold = 0.1)

plotting.orbits(
    pop.get_states(named=False),
    title =  "Orbit distribution of Master catalog",
    axis_labels = 'earth-orbit',
)

print(f'Master: {len(pop)} | Master - factor (diameter > 0.1 m): {len(pop_factor)}')

plt.show()
