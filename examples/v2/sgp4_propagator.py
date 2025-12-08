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
print(spobj.state)

t = np.linspace(0, 3600 * 24.0, num=1000)

# we can propagate and get ITRS out
prop.settings.out_frame = "ITRS"
states_itrs = prop.propagate(spobj, t)

# or we can set out_frame to TEME which will cause no
# transformation to be applied after propagation
prop.settings.out_frame = "TEME"
states_gcrs = prop.propagate(spobj, t)


# or we can propagate the epoch of the object
new_spobj = prop.propagate_to_new_epoch(spobj, 3600.0)
print(f"{spobj=}\n{spobj.state}")
print(f"{new_spobj=}\n{new_spobj.state}")


fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(121, projection="3d")
ax.plot(states_itrs[0, :], states_itrs[1, :], states_itrs[2, :], "-b")
ax.set_title("In: TEME, out: ITRS")
ax.axis("equal")

ax = fig.add_subplot(122, projection="3d")
ax.plot(states_gcrs[0, :], states_gcrs[1, :], states_gcrs[2, :], "-b")
ax.set_title("In: TEME, out: GCRS")
ax.axis("equal")

plt.show()
