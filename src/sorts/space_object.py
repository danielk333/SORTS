#!/usr/bin/env python

"""Defines a space object. Encapsulates orbital elements, propagation and related methods."""
from copy import deepcopy
from typing import Any, Self
from dataclasses import dataclass, fields

import numpy as np
from pyorb import Orbit, M_earth
from astropy.time import Time, TimeDelta
from sorts.types import Frames, NDArray_N


@dataclass
class SpaceObject:
    """Encapsulates a object in space which has a state, at an epoch and in a frame, and some
    properties.

    The state of the object is stored in a `pyorb.Orbit` instance.
    This instance contains direct transformations between the Cartesian and Kepler states.

    To propagate this object in time and get states, supply it to a propagator.
    """

    state: Orbit
    frame: Frames
    epoch: Time
    properties: dict[str, Any]
    object_id: int = 0

    def copy(self) -> Self:
        kwargs = {key: deepcopy(getattr(self, key)) for key in self.keys}
        return self.__class__(**kwargs)

    @property
    def keys(self) -> list[str]:
        return [key.name for key in fields(self)]

    @classmethod
    def from_kepler(
        cls,
        semi_major_axis: float,
        eccentricity: float,
        inclination: float,
        argument_of_periapsis: float,
        longitude_of_ascending_node: float,
        mean_anomaly: float,
        epoch: Time,
        frame: Frames,
        properties: dict[str, Any],
        center_mass: float = M_earth,
        object_id: int = 0,
        degrees: bool = True,
    ):
        state = Orbit(
            M0=center_mass,
            degrees=degrees,
            type="mean",
            auto_update=True,
            direct_update=True,
            num=1,
            m=0.0,
        )
        state._kep[0, 0] = semi_major_axis
        state._kep[1, 0] = eccentricity
        state._kep[2, 0] = inclination
        state._kep[3, 0] = argument_of_periapsis
        state._kep[4, 0] = longitude_of_ascending_node
        state._kep[5, 0] = mean_anomaly
        state.calculate_cartesian()

        return cls(
            state=state,
            frame=frame,
            epoch=epoch,
            properties=properties,
            object_id=object_id,
        )

    def __repr__(self):
        return f"SpaceObject(oid={self.object_id} @ {self.epoch.iso})"

    @property
    def d(self) -> float:
        if "d" in self.properties:
            diam = self.properties["d"]
        elif "diam" in self.properties:
            diam = self.properties["diam"]
        elif "r" in self.properties:
            diam = self.properties["r"] * 2
        elif "A" in self.properties:
            diam = np.sqrt(self.properties["A"] / np.pi) * 2
        else:
            raise AttributeError(
                "Space object does not have a diameter parameter or any way to calculate one"
            )
        return diam

    def to_relative_time(self, times: Time | TimeDelta | NDArray_N) -> NDArray_N:
        if isinstance(times, TimeDelta):
            tv = times.sec
        elif isinstance(times, Time):
            tv = (times - self.epoch).sec
        else:
            tv = times  # assume input is in seconds

        if not isinstance(tv, np.ndarray):
            tv = np.array([tv])
        return tv

    def __str__(self):
        p = "\nSpace object {}: {}:\n".format(self.oid, repr(self.epoch))
        p += str(self.state) + "\n"
        p += "Parameters: " + ", ".join([f"{key}={val}" for key, val in self.parameters.items()])
        return p
