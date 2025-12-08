#!/usr/bin/env python

"""Defines a space object. Encapsulates orbital elements, propagation and related methods."""
from typing import Any, Literal
from dataclasses import dataclass

import numpy as np
from pyorb import Orbit, M_earth
from astropy.time import Time, TimeDelta
from sorts.types import EcefStates, Frames
from sorts.propagator import Propagator
import spacecoords.celestial as cel


@dataclass
class SpaceObject:
    """Encapsulates a object in space which has a state, at an epoch and in a frame, and some
    properties.

    The state of the object is stored in a `pyorb.Orbit` instance. This instance contains direct transformations between
    the Cartesian and Kepler states.

    To propagate this object in time, simply supply it to a propagator.
    """

    state: Orbit
    frame: Frames
    epoch: Time
    properties: dict[str, Any]
    object_id: int = 0

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
    ):
        state = Orbit(
            M0=center_mass,
            degrees=True,
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
        return f"SpaceObject(oid={self.oid})"

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

    def propagate_new_epoch(self, dt: TimeDelta, propagator: Propagator):
        """Propagate and change the epoch of this space object if the state is a `pyorb.Orbit`."""

        if "in_frame" in self.propagator.settings and "out_frame" in self.propagator.settings:
            out_frame = self.propagator.settings["out_frame"]
            self.propagator.set(out_frame=self.propagator.settings["in_frame"])

            state = self.get_state(np.array([dt], dtype=np.float64))

            self.propagator.set(out_frame=out_frame)
        else:
            state = self.get_state(np.array([dt], dtype=np.float64))

        self.epoch = self.epoch + TimeDelta(dt, format="sec")

        x, y, z, vx, vy, vz = state.flatten()

        self.update(
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
        )

    def update(self, **kwargs):
        """If a `pyorb.Orbit` is present, updates the orbital elements and Cartesian state vector of the space object.

        Can update any of the related state parameters, all others will automatically update.

        Cannot update Keplerian and Cartesian elements simultaneously.

        :param float a: Semi-major axis in km
        :param float e: Eccentricity
        :param float i: Inclination in degrees
        :param float aop/omega: Argument of perigee in degrees
        :param float raan/Omega: Right ascension of the ascending node in degrees
        :param float mu0/anom: Mean anomaly in degrees
        :param float x: X position in km
        :param float y: Y position in km
        :param float z: Z position in km
        :param float vx: X-direction velocity in km/s
        :param float vy: Y-direction velocity in km/s
        :param float vz: Z-direction velocity in km/s
        """
        if not isinstance(self.state, pyorb.Orbit):
            raise ValueError(f"Cannot update non-Orbit state ({type(self.state)})")

        if "aop" in kwargs:
            kwargs["omega"] = kwargs.pop("aop")
        if "raan" in kwargs:
            kwargs["Omega"] = kwargs.pop("raan")
        if "mu0" in kwargs:
            kwargs["anom"] = kwargs.pop("mu0")

        for key in kwargs:
            if key not in pyorb.Orbit.UPDATE_KW:
                self.parameters[key] = kwargs[key]

        self.orbit.update(**kwargs)

    def __str__(self):
        p = "\nSpace object {}: {}:\n".format(self.oid, repr(self.epoch))
        p += str(self.state) + "\n"
        p += f"Parameters: " + ", ".join([f"{key}={val}" for key, val in self.parameters.items()])
        return p

    def get_position(self, t):
        """Gets position at specified times using propagator instance.

        :param float/list/numpy.ndarray t: Time relative epoch in seconds.

        :return: Array of positions as a function of time.
        :rtype: numpy.ndarray of size (3,len(t))
        """
        ecefs = self.get_state(t)
        return ecefs[:3, :]

    def get_velocity(self, t):
        """Gets velocity at specified times using propagator instance.

        :param float/list/numpy.ndarray t: Time relative epoch in seconds.

        :return: Array of positions as a function of time.
        :rtype: numpy.ndarray of size (3,len(t))
        """
        ecefs = self.get_state(t)
        return ecefs[3:, :]

    def get_state(self, t) -> EcefStates:
        """Gets ECEF state at specified times using propagator instance.

        :param int/float/list/numpy.ndarray/astropy.time.Time/astropy.time.TimeDelta t: Time relative epoch in seconds.

        :return: Array of state (position and velocity) as a function of time.
        :rtype: numpy.ndarray of size (6,len(t))
        """
        kw = {}
        kw.update(self.propagator_args)
        kw.update(self.parameters)

        ret = self.propagator.propagate(t=t, state0=self.state, epoch=self.epoch, **kw)

        # if propagator returns something non-standard, just return that
        # Otherwise, ensures a 2d-object is returned
        if isinstance(ret, np.ndarray):
            if len(ret.shape) == 1:
                ret = ret.reshape((6, 1))

        return ret
