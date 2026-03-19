#!/usr/bin/env python

"""TLE list loading to population"""

from pathlib import Path

import numpy as np
from astropy.time import Time
import pyorb

from sorts.propagator import pysgp4
from .population import Population


def tle_catalog(
    tles: str | Path | list[tuple[str, str]],
):
    """Reads a TLE-snapshot file and converts the TLE's to orbits in a TEME frame and creates a population file.
    A snapshot generally contains several TLE's for the same object thus will this population also contain duplicate objects.
    The BSTAR parameter is saved in field BSTAR `BSTAR`.

    *Numerical propagator assumptions:*
    To propagate with a numerical propagator one needs to make assumptions.
       * Density is $5\\cdot 10^3 \\;\\frac{kg}{m^3}$.
       * Object is a sphere
       * Drag coefficient is 2.3.

    Can take path to the input TLE snapshot file. Or the TLE-set can be given directly as a list
    of two lines that can be unpacked in a loop, e.g. `[(tle1_l1, tle1_l2), (tle2_l1, tle2_l2)]`.
    """
    if isinstance(tles, str) or isinstance(tles, Path):
        # first character in a line is line number (1 or 2), so just ignore everything else
        tle = {"1": [], "2": []}
        for line in open(tles):
            num = line[0]
            if num not in ["1", "2"]:
                continue
            tle[num].append(line.rstrip("\n"))

        if len(tle["1"]) != len(tle["2"]):
            raise Exception("Not even number of lines [not TLE compatible]")

        tles = list(zip(tle["1"], tle["2"]))

    tle_size = len(tles)

    sgp4_settings = pysgp4.Sgp4Settings(
        out_frame="TEME",
    )
    prop = pysgp4.Sgp4(sgp4_settings)

    states = np.empty((6, tle_size), dtype=np.float64)
    parameters = dict(
        bstar=np.empty((tle_size,), dtype=np.float64),
        line1=np.empty((tle_size,), dtype="S70"),
        line2=np.empty((tle_size,), dtype="S70"),
    )

    satnum = np.empty((tle_size,), dtype=np.int64)
    dtypes = {"line1": "S70", "line2": "S70", "id": np.int64}
    jd1 = np.empty((tle_size,), dtype=np.float64)
    jd2 = np.empty((tle_size,), dtype=np.float64)
    for line_id, lines in enumerate(tles):
        line1, line2 = lines
        parameters["line1"][line_id] = line1
        parameters["line2"][line_id] = line2

        params = pysgp4.get_TLE_parameters(line1, line2)
        jd1[line_id] = params["jdsatepoch"]
        jd2[line_id] = params["jdsatepochF"]

        satnum[line_id] = params["satnum"]

        # TODO: is this to convert to SI?
        bstar = params["bstar"] / (prop.radiusearthkm * 1000.0)
        parameters["bstar"][line_id] = bstar

        state_TEME = prop.propagate_tle(line1, line2, np.array([0.0]))
        states[:, line_id] = state_TEME[:, 0]

    jd_epochs = Time(jd1, jd2, format="jd", scale="utc")

    pop = Population(
        states=states,
        epochs=jd_epochs,
        frame="TEME",
        parameters=parameters,
        object_ids=satnum,
        state_format="cartesian",
        anomly_type="mean",
        dtypes=dtypes,
        default_dtype=np.float64,
        epoch_format="jd",
        epoch_scale="utc",
        degrees=True,
    )
    return pop
