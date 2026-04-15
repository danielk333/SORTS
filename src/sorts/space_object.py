"""Defines a space object. Encapsulates orbital elements, propagation and related methods."""

import typing as t
from copy import deepcopy
from dataclasses import dataclass, fields
import numpy as np
from pyorb import Orbit, M_earth
from astropy.time import Time
from sorts.types import Frames


@dataclass
class SpaceObject:
    """Encapsulates a object in space which has an orbit, at an epoch and in a frame, and some
    properties.

    The orbit of the object is a `pyorb.Orbit` instance,
    it contains direct transformations between the Cartesian and Kepler states.

    To propagate this object in time and get states, supply it to a propagator.
    """

    orbit: Orbit
    frame: Frames
    epoch: Time
    properties: dict[str, t.Any]
    object_id: int = 0

    def copy(self) -> t.Self:
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
        properties: dict[str, t.Any],
        center_mass: float = M_earth,
        object_id: int = 0,
        degrees: bool = True,
    ):
        orbit = Orbit(
            M0=center_mass,
            degrees=degrees,
            type="mean",
            auto_update=True,
            direct_update=True,
            num=1,
            m=0.0,
        )
        orbit._kep[0, 0] = semi_major_axis
        orbit._kep[1, 0] = eccentricity
        orbit._kep[2, 0] = inclination
        orbit._kep[3, 0] = argument_of_periapsis
        orbit._kep[4, 0] = longitude_of_ascending_node
        orbit._kep[5, 0] = mean_anomaly
        orbit.calculate_cartesian()

        return cls(
            orbit=orbit,
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

    def __str__(self):
        orb_str = str(self.orbit)
        orb_str = "".join([f"  {row}\n" for row in orb_str.split("\n")])
        p = f"\nSpace object {self.object_id}: {repr(self.epoch)}:\n"
        p += orb_str
        p += "Parameters:\n" + "\n".join([f"  {key}={val}" for key, val in self.properties.items()])
        return p
