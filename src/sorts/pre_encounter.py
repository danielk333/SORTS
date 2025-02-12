#!/usr/bin/env python

"""Module for using REBOUND to calculate pre-encounter states of objects that make close-encounters/collisions with Earth.


"""

import sys
import os
import time
import glob

from tqdm import tqdm
import numpy as np
import scipy
import h5py
from astropy.time import Time, TimeDelta

import pyorb

from .propagator import Rebound


def distance_termination(dAU):
    def distance_termination_method(self, t, step_index, massive_states, particle_states):
        e_state = massive_states[:3, step_index, self._earth_ind]
        d_earth = (
            np.linalg.norm(particle_states[:3, step_index, :] - e_state[:, None], axis=0) / pyorb.AU
        )
        return np.all(d_earth > dAU)

    return distance_termination_method


def propagate_pre_encounter(
    state,
    epoch,
    in_frame,
    out_frame,
    termination_check,
    kernel,
    dt=10.0,
    max_t=10 * 24 * 3600.0,
    settings=None,
):
    """Propagates a state from the states backwards in time until the termination_check is true.

    #TODO: implement no saving of intermediate states for large particle runs
    """
    t = -np.arange(0, max_t, dt, dtype=np.float64)

    class TerminatedRebound(Rebound):
        pass

    TerminatedRebound.termination_check = termination_check

    reb_settings = dict(
        in_frame=in_frame,
        out_frame=out_frame,
        time_step=dt,  # s
        termination_check=True,
        save_massive_states=True,
    )
    if settings is not None:
        settings.update(reb_settings)
    else:
        settings = reb_settings

    prop = TerminatedRebound(
        kernel=kernel,
        settings=settings,
    )

    states, massive_states = prop.propagate(t, state, epoch)

    t = t[: states.shape[1]]

    return states, massive_states, t
