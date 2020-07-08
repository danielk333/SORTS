#!/usr/bin/env python

'''

'''
import matplotlib.pyplot as plt

from sorts import plotting
from sorts.population import master_catalog, master_catalog_factor

#THIS NEEDS TO BE REPLACED WITH PATH TO _YOUR_ FILE
path = '/home/danielk/IRF/IRF_GITLAB/SORTSpp/master/celn_20090501_00.sim'

pop = master_catalog(path)
pop_factor = master_catalog_factor(pop, treshhold = 0.1)

plotting.orbits(
    pop.get_all_orbits(order_angs=True),
    title =  "Orbit distribution of Population",
)

print(f'Master: {len(pop)} | Master - factor (diameter > 0.1 m): {len(pop_factor)}')

plt.show()
