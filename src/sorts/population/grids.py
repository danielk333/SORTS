#!/usr/bin/env python

"""Defines grid sampled populations over some parameters."""
from typing import Any
import numpy as np
from astropy.time import Time
from .population import Population
from sorts.types import NDArray_N, Frames


def orbit_grid(
    semi_major_axis_samples: NDArray_N,
    eccentricity_samples: NDArray_N,
    inclination_samples: NDArray_N,
    argument_of_periapsis_samples: NDArray_N,
    longitude_of_ascending_node_samples: NDArray_N,
    mean_anomaly_samples: NDArray_N,
    diameter_samples: NDArray_N,
    frame: Frames = "GCRS",
    epoch_mjd: NDArray_N | float = 53005.0,
    additional_parameters: dict[str, Any] = None,
    degrees: bool = True,
):
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
    parameters = {"d": grids[-1]}
    if additional_parameters is not None:
        parameters.update(additional_parameters)

    pop = Population(
        states=np.stack(grids[:-1]),
        epochs=Time(
            np.full((size,), epoch_mjd, dtype=np.float64),
            format="mjd",
            scale="utc",
        ),
        frame=frame,
        parameters=parameters,
        object_ids=np.arange(size),
        state_format="kepler",
        anomly_type="mean",
        dtypes={"id": np.int64},
        default_dtype=np.float64,
        epoch_format="mjd",
        epoch_scale="utc",
        degrees=degrees,
    )
    return pop
