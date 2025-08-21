#!/usr/bin/env python

"""
Starting example
=========================

"""
import numpy as np
import pyorb
import sorts

radar = sorts.get_radar("eiscat3d", "stage1-array")

prop = sorts.propagator.SGP4(
    settings=dict(
        out_frame="ITRS",
    ),
)

orb = pyorb.Orbit(
    M0=pyorb.M_earth,
    direct_update=True,
    auto_update=True,
    degrees=True,
    a=7200e3,
    e=0.05,
    i=75,
    omega=0,
    Omega=79,
    anom=72,
    epoch=53005.0,
)
print(orb)

t = np.arange(0, 3600 * 24 * 1, 30.0)

states = prop.propagate(t, orb.cartesian[:, 0], orb.epoch)

passes = radar.find_passes(t, states)

for txi in range(len(radar.tx)):
    for rxi in range(len(radar.rx)):
        for ps in passes[txi][rxi]:
            print(ps)
