"""
Functions for core functionalities of this subpackage

- Intended to be imported as a whole module when consuming
"""

# TODO: this module can be moved to top level as helper funcs of sorts pkg?

from dataclasses import dataclass
import typing as t
import pickle
from pathlib import Path
import numpy as np
import numpy.typing as npt
from astropy.time import Time
from sorts.types import Datetime64_us, EcefStates, Float64_as_sec, Datetime_Like
from sorts.utils import to_datetime64_us
from sorts.radar import Station
from sorts.space_object import SpaceObject
from .types import Passage, SpaceObjectJacobianTuple
from sorts.interpolation import Interpolator
from sorts.propagator import Propagator
from sorts.population import Population


@dataclass
class InterpolatedPropagation:
    times: npt.NDArray[Datetime64_us]
    states: EcefStates
    interpolator: Interpolator

    @classmethod
    def from_space_objects(
        cls,
        space_objects: t.Sequence[SpaceObject] | Population,
        propagator: Propagator,
        interpolator_class: t.Type[Interpolator],
        start_time: Time,
        end_time: Time,
        time_step: float,
    ) -> list[t.Self]:
        dt = (end_time - start_time).sec
        prop_interps = []
        for spobj in space_objects:
            t0 = (start_time - spobj.epoch).sec
            t_obj = np.arange(t0, t0 + dt, time_step, dtype=np.float64)

            itrs_states = propagator.propagate(spobj, t_obj)
            interp = interpolator_class(itrs_states, t_obj)
            pint = cls(
                times=spobj.epoch.datetime64 + t_obj,
                states=itrs_states,
                interpolator=interp,
            )
            prop_interps.append(pint)
        return prop_interps


def find_simultaneous_passages(
    dt: npt.NDArray[Float64_as_sec],
    space_object: SpaceObject,
    states: EcefStates,
    tx_station: Station,
    rx_stations: t.Sequence[Station],
    epoch: Datetime_Like,
    fov_kw=None,
) -> list[Passage]:
    """
    Finds all passages that are simultaneously inside all tx, rx stations' FOV.
    """
    # NOTE: based on the `find_passes` func in `src/sorts/passes.py`

    epoch = to_datetime64_us(epoch)

    passages: list[Passage] = []
    if fov_kw is None:
        fov_kw = {}

    enu = []
    check = np.full((len(dt),), True, dtype=bool)
    for station in [tx_station, *rx_stations]:
        enu_st = station.enu(states)
        enu.append(enu_st)

        check_st = station.field_of_view(states, **fov_kw)
        check = np.logical_and(check, check_st)

    inds = np.where(check)[0]

    if len(inds) == 0:
        return passages

    dind = np.diff(inds)
    splits = np.where(dind > 1)[0]

    splits = np.insert(splits, 0, -1)
    splits = np.insert(splits, len(splits), len(inds) - 1)
    splits += 1
    for si in range(len(splits) - 1):
        ps_inds = inds[splits[si] : splits[si + 1]]
        if len(ps_inds) == 0:
            continue

        start_time: Datetime64_us = t.cast(
            np.timedelta64, (dt[ps_inds[0]] * 1e6).astype("timedelta64[us]")
        ) + np.datetime64(epoch)

        end_time: Datetime64_us = t.cast(
            np.timedelta64, (dt[ps_inds[-1]] * 1e6).astype("timedelta64[us]")
        ) + np.datetime64(epoch)

        time_range = (start_time, end_time)
        passages.append(
            Passage(
                space_object=space_object,
                tx_station=tx_station,
                rx_stations=list(rx_stations),
                epoch=epoch,
                time_range=time_range,
            )
        )

    return passages


def find_passages(
    dt: npt.NDArray[Float64_as_sec],
    space_object: SpaceObject,
    states: EcefStates,
    tx_station: Station,
    rx_station: Station,
    epoch: Datetime_Like,
    fov_kw=None,
) -> list[Passage]:
    """
    Finds all passages that are simultaneously inside a tx-rx station pair's FOV.
    """
    passages = find_simultaneous_passages(
        dt=dt,
        space_object=space_object,
        states=states,
        tx_station=tx_station,
        rx_stations=[rx_station],
        epoch=epoch,
        fov_kw=fov_kw,
    )
    return passages


def duplicate_and_perturbate_space_objects(
    spobjs: t.Sequence[SpaceObject],
    propagator: Propagator,
    interpolator_class: t.Type[Interpolator],
    start_time: Time,
    end_time: Time,
    time_step: float,
    pert_val: tuple[float, float, float, float, float, float] = (
        1e-3, 1e-3, 1e-3, 1e-5, 1e-5, 1e-5  # fmt: skip
    ),
) -> list[tuple[SpaceObjectJacobianTuple, InterpolatedPropagation]]:
    # duplicate list items
    spobjs_jacobian_tuples = [
        (spobj, spobj.copy(), spobj.copy(), spobj.copy(), spobj.copy(), spobj.copy(), spobj.copy())
        for spobj in spobjs
    ]

    # perturbate
    for spobjs_jacobian_tuple in spobjs_jacobian_tuples:
        for idx, spobj in enumerate(spobjs_jacobian_tuple):
            # the original spobj are left intact
            if idx != 0:
                spobj.state._cart[idx - 1, 0] += pert_val[idx - 1]

    prop_interps = InterpolatedPropagation.from_space_objects(
        space_objects=spobjs,
        propagator=propagator,
        interpolator_class=interpolator_class,
        start_time=start_time,
        end_time=end_time,
        time_step=time_step,
    )
    return list(zip(spobjs_jacobian_tuples, prop_interps))


def ensure_directory_exist(dpath: str | Path):
    """Check if the directory exist and create it if not"""

    dpath = Path(dpath)

    if not dpath.exists():
        dpath.mkdir(parents=True)
    assert dpath.exists()
    assert dpath.is_dir()


def safe_pickle(obj, fpath: str | Path):
    """
    Use pickle to save an object to the specified file path, with a few extra steps to make the write operation safer:
    - The output directory will be created if not exists
    - We write to an tmp file first then rename that file, as a simple way to reduce risk of corrupted files
    """

    fpath = Path(fpath)

    ensure_directory_exist(fpath.parent)

    fpath_tmp = fpath.with_suffix(fpath.suffix + ".tmp")
    with open(fpath_tmp, "wb") as f:
        pickle.dump(obj, f)
    fpath_tmp.rename(fpath)
