#!/usr/bin/env python

"""Defines grid sampled populations over some parameters."""
import numpy as np

from .population import Population
from ..types import NDArray_N


def orbit_grid(
    semi_major_axis_samples: NDArray_N,
    eccentricity_samples: NDArray_N,
    inclination_samples: NDArray_N,
    argument_of_periapsis_samples: NDArray_N,
    longitude_of_ascending_node_samples: NDArray_N,
    mean_anomaly_samples: NDArray_N,
    diameter_samples: NDArray_N,
    mjd0: NDArray_N | float = 53005.0,
    propagator=None,
    propagator_options={},
    propagator_args={},
):
    pop = Population(
        fields=[
            "oid",
            "a",
            "e",
            "i",
            "aop",
            "raan",
            "mu0",
            "mjd0",
            "d",
            "C_D",
        ],
        dtypes=["int"] + ["float64"] * 9,
        space_object_fields=["d", "C_D"],
        state_fields=["a", "e", "i", "aop", "raan", "mu0"],
        epoch_field={"field": "mjd0", "format": "mjd", "scale": "utc"},
        propagator=propagator,
        propagator_options=propagator_options,
        propagator_args=propagator_args,
    )
    samples = [
        semi_major_axis_samples,
        eccentricity_samples,
        inclination_samples,
        argument_of_periapsis_samples,
        longitude_of_ascending_node_samples,
        mean_anomaly_samples,
        diameter_samples,
    ]
    grids = [x.flatten() for x in np.meshgrid(*samples)]
    size = grids[0].size

    pop.allocate(size)
    pop.data["oid"] = np.arange(size)
    pop.data["a"] = grids[0]
    pop.data["e"] = grids[1]
    pop.data["i"] = grids[2]
    pop.data["aop"] = grids[3]
    pop.data["raan"] = grids[4]
    pop.data["mu0"] = grids[5]
    pop.data["mjd0"] = mjd0
    pop.data["d"] = grids[6]
    pop.data["C_D"] = 2.3

    return pop
