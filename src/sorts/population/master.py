#!/usr/bin/env python

"""Defines function for reading the MASTER catalog files."""

from pathlib import Path
import numpy as np

from .population import Population
from sorts.types import NDArray_N, Frames


def master_catalog(
    input_file: str | Path,
    epoch_mjd: float = 53005.0,
    frame: Frames = "TEME",
) -> Population:
    """Return the master catalog specified in the input file as a population instance. The catalog only contains the master sampling objects and not an actual realization of the population using the factor.

    **Note:** MASTER catalog files are usually in TEME frame

    The format of the input master files is:

        0. ID
        1. Factor
        2. Mass [kg]
        3. Diameter [m]
        4. m/A [kg/m2]
        5. a [km]
        6. e
        7. i [deg]
        8. RAAN [deg]
        9. AoP [deg]
        10. M [deg]
    """
    # TODO: confirm if MASTER catalog files are actually usually in TEME frame, then update the docstring

    master_raw = np.genfromtxt(input_file)
    states = np.empty((6, len(master_raw)), dtype=np.float64)
    states[0, :] = master_raw[:, 5] * 1e3  # semi-major axis in km
    states[1, :] = master_raw[:, 6]  # eccentricity
    states[2, :] = master_raw[:, 7]  # inclination in deg
    states[3, :] = master_raw[:, 9]  # argument of periapsis in deg
    states[4, :] = master_raw[:, 8]  # longitude of the ascending node in deg
    states[5, :] = master_raw[:, 10]  # mean anomaly in deg

    parameters = dict(
        area_to_mass=master_raw[:, 4],
        mass=master_raw[:, 2],
        diameter=master_raw[:, 3],
        area=master_raw[:, 2] / master_raw[:, 4],
        factor=master_raw[:, 1],
        master_id=master_raw[:, 0],
    )

    master = Population(
        states=states,
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
        degrees=True,
    )
    return master


def master_catalog_factor(
    master_base: Population,
    copy: bool = True,
    treshhold: float = 0.01,
    seed: int | None = None,
) -> Population:
    """Returns a random realization of the master population specified by the input file/population. In other words, each sampling object in the catalog is sampled a "factor" number of times with random mean anomalies to create the population.

    Args:
        master_base: A master catalog consisting only of sampling objects.
            This catalog will be modified and the pointer to it returned.
        treshhold: Diameter limit in meters below witch sampling objects are not included.
            Can be `None` to skip filtering.
        seed: Random number generator seed given to `numpy.random.seed` to
            allow for consisted generation of a random realization of the population.
            If seed is `None` a random seed from high-entropy data is used.
        copy: Modify the given `master_base` instance or return a modified copy.

    """
    if seed is not None:
        st0 = np.random.get_state()
        np.random.seed(seed=seed)

    raise NotImplementedError()
    if copy:
        master = master_base.copy()
    else:
        master = master_base

    if treshhold is not None:
        master.filter("diameter", lambda d: d >= treshhold)

    factors = np.round(master.data["factor"]).astype(np.int64)
    new_len = int(np.sum(factors)),

    i = 0
    for row in master.data:
        f_int = int(np.round(row[13]))
        if f_int >= 1:
            ip = i + f_int
            for coli, head in enumerate(master.fields):
                full_objs[i:ip, coli] = row[head]

            full_objs[i:ip, 0] = np.array(range(i, ip), dtype=np.float64)
            full_objs[i:ip, 6] = np.random.rand(f_int) * 360.0
            i = ip

    master.allocate(full_objs.shape[0])

    master[:, :] = full_objs

    if seed is not None:
        np.random.seed(None)
        np.random.set_state(st0)

    return master
