#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:29:10 2022

@author: thomas
"""

import numpy as np
from sorts import plotting
import matplotlib.pyplot as plt

import pyorb
from sorts.radar.system import instances

eiscat3d = instances.eiscat3d

orb = pyorb.Orbit(
    M0 = pyorb.M_earth,
    direct_update=True,
    auto_update=True,
    degrees = True,
    a=7000e3,
    e=0.1,
    i=90,
    omega=0,
    Omega=20,
)

fig = plt.figure(figsize=(15,15))

ax = fig.add_subplot(111, projection='3d')
plotting.general.grid_earth(ax)

for tx in eiscat3d.tx:
    ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
    
for rx in eiscat3d.rx:
    ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')

plotting.kepler_orbit(orb, ax=ax)

plt.show()