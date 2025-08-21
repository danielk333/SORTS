#!/usr/bin/env python

"""
SGP4 propagator usage
======================================
"""

import numpy as np
import matplotlib.pyplot as plt
import pyorb
from astropy.time import Time

from sorts.propagator import SGP4
from sorts import frames

prop = SGP4(
    settings=dict(
        out_frame="ITRS",
    ),
)

print(prop)

orb = pyorb.Orbit(
    M0=pyorb.M_earth,
    direct_update=True,
    auto_update=True,
    degrees=True,
    a=7000e3,
    e=0,
    i=69,
    omega=0,
    Omega=0,
    anom=0,
)
print(orb)

t = np.linspace(0, 3600 * 24.0, num=5000)
mjd0 = 53005
times = Time(mjd0 + t / (3600 * 24.0), format="mjd", scale="utc")

# we can propagate and get ITRS out
states_itrs = prop.propagate(t, orb.cartesian[:, 0], epoch=mjd0, A=1.0, C_R=1.0, C_D=1.0)

# or we can set out_frame to TEME which will cause no transformation to be applied after propagation
prop.set(out_frame="TEME")
states_teme = prop.propagate(t, orb.cartesian[:, 0], epoch=mjd0, A=1.0, C_R=1.0, C_D=1.0)

# We can also apply conversion transforms ourself
states_itrs_teme = frames.convert(
    times,
    states_itrs,
    in_frame="ITRS",
    out_frame="TEME",
)
states_teme_itrs = frames.convert(
    times,
    states_teme,
    in_frame="TEME",
    out_frame="ITRS",
)


fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(221, projection="3d")
ax.plot(states_itrs[0, :], states_itrs[1, :], states_itrs[2, :], "-b")
ax.set_title("In: TEME, out: ITRS, convert: None")

ax = fig.add_subplot(222, projection="3d")
ax.plot(states_teme[0, :], states_teme[1, :], states_teme[2, :], "-b")
ax.set_title("In: TEME, out: TEME, convert: None")

ax = fig.add_subplot(223, projection="3d")
ax.plot(states_itrs_teme[0, :], states_itrs_teme[1, :], states_itrs_teme[2, :], "-b")
ax.set_title("In: TEME, out: ITRS, convert: ITRS->TEME")

ax = fig.add_subplot(224, projection="3d")
ax.plot(states_teme_itrs[0, :], states_teme_itrs[1, :], states_teme_itrs[2, :], "-b")
ax.set_title("In: TEME, out: TEME, convert: TEME->ITRS")

plt.show()
