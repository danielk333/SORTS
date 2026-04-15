#!/usr/bin/env python

"""
SGP4 propagator usage
======================================
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

from sorts.propagator import Sgp4, Sgp4Settings
from sorts.space_object import SpaceObject

prop = Sgp4(settings=Sgp4Settings())

spobj = SpaceObject.from_kepler(
    semi_major_axis=7000e3,
    eccentricity=0,
    inclination=69,
    argument_of_periapsis=0,
    longitude_of_ascending_node=0,
    mean_anomaly=0,
    epoch=Time("2025-12-8T00:00:00", format="isot", scale="utc"),
    frame="TEME",
    properties={},
    degrees=True,
)
print(spobj.orbit)

t = np.linspace(0, 3600 * 24.0, num=1000)

# we can propagate and get ITRS out
prop.settings.out_frame = "ITRS"
states_itrs = prop.propagate(spobj, t)

# or we can set out_frame to TEME which will cause no
# transformation to be applied after propagation
prop.settings.out_frame = "TEME"
states_gcrs = prop.propagate(spobj, t)


# We can also use TLE input
tle_line1 = "1 27421U 02021A   02124.48976499 -.00021470  00000-0 -89879-2 0    20"
tle_line2 = "2 27421  98.7490 199.5121 0001333 133.9522 226.1918 14.26113993    62"

prop.settings.out_frame = "ITRS"
tle_itrs = prop.propagate_tle(tle_line1, tle_line2, t)

prop.settings.out_frame = "TEME"
tle_gcrs = prop.propagate_tle(tle_line1, tle_line2, t)


fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(221, projection="3d")
ax.plot(states_itrs[0, :], states_itrs[1, :], states_itrs[2, :], "-b")
ax.set_title("In: TEME state, out: ITRS")
ax.axis("equal")

ax = fig.add_subplot(222, projection="3d")
ax.plot(states_gcrs[0, :], states_gcrs[1, :], states_gcrs[2, :], "-b")
ax.set_title("In: TEME state, out: GCRS")
ax.axis("equal")

ax = fig.add_subplot(223, projection="3d")
ax.plot(tle_itrs[0, :], tle_itrs[1, :], tle_itrs[2, :], "-b")
ax.set_title("In: TLE, out: ITRS")
ax.axis("equal")

ax = fig.add_subplot(224, projection="3d")
ax.plot(tle_gcrs[0, :], tle_gcrs[1, :], tle_gcrs[2, :], "-b")
ax.set_title("In: TLE, out: GCRS")
ax.axis("equal")

plt.show()
