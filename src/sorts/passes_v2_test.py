import typing as t, os, pickle
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from astropy.time import Time
import pyorb
import sorts
from sorts.propagator import SGP4
from .passes_v2 import find_simultaneous_passes
from .radar.radars.composite_key import RadarStationCompositeKey
from . import signals


def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()


def find_simultaneous_matches_radar_passes_example_test():
    passes_pickle_fpath = (
        Path(os.path.dirname(os.path.abspath(__file__)))
        / ".."
        / ".."
        / "examples"
        / "radar_passes__passes.pickle"
    )
    with open(passes_pickle_fpath, "rb") as f:
        radar_passes__passes = pickle.load(f)

    eiscat3d = sorts.get_radar("eiscat3d", "stage1-array")
    epoch = Time(53005.0, format="mjd", scale="utc")
    radar_station_composite_keys: list[RadarStationCompositeKey] = [
        ("eiscat3d", "stage1-array", "tx", "0"),
        ("eiscat3d", "stage1-array", "rx", "1"),
    ]

    Prop_cls = SGP4
    Prop_opts = dict(
        settings=dict(
            out_frame="ITRS",
        ),
    )
    prop = Prop_cls(**Prop_opts)

    orb = pyorb.Orbit(
        M0=pyorb.M_earth,
        direct_update=True,
        auto_update=True,
        degrees=True,
        a=7200e3,
        e=0.05,
        i=75,
        omega=0,
        Omega=79,
        anom=72,
        epoch=epoch.to_value("mjd"),
    )

    dt_arr = sorts.equidistant_sampling(
        orbit=orb,
        start_t=0,
        end_t=3600 * 24 * 1,
        max_dpos=1e3,
    )

    states = prop.propagate(dt_arr, orb.cartesian[:, 0], orb.epoch, A=1.0, C_R=1.0, C_D=1.0)

    result = find_simultaneous_passes(
        dt_arr=dt_arr,
        states=states,
        stations=[eiscat3d.tx[0], eiscat3d.rx[1]],
        radar_station_composite_keys=radar_station_composite_keys,
        epoch=t.cast(datetime, epoch.to_datetime(timezone=timezone.utc)),
    )

    target_passes = radar_passes__passes[0][1]
    # assert the deltatime ndarray are equal:
    # - ndarray from target is converted from `float64` of seconds to "timedelta64[us]"
    # - ndarray from result is converted from "datetime64[us]" to "timedelta64[us]"
    assert all(
        [
            np.array_equal(
                (target_pass.t * 1e6).astype("timedelta64[us]"),
                candidate_pass.get_deltatime_ndarray(
                    t.cast(datetime, epoch.to_datetime(timezone=timezone.utc))
                ),
            )
            for target_pass, candidate_pass in zip(target_passes, result)
        ]
    )

    # assert the states ndarray are equal
    assert all(
        [
            np.array_equal(target_pass.enu, candidate_pass.enu)
            for target_pass, candidate_pass in zip(target_passes, result)
        ]
    )

    result_snrs = [
        signals.calculate_snr(
            pass_obj=pass_obj, tx=eiscat3d.tx[0], rx=eiscat3d.rx[0], diameter=0.05
        )
        for pass_obj in result
    ]

    # assert the snr ndarray are equal
    target_snrs = [
        ps.calculate_snr(eiscat3d.tx[0], eiscat3d.rx[0], diameter=0.05) for ps in target_passes
    ]
    assert all(
        [
            np.array_equal(target_snr, candidate_snr)
            for target_snr, candidate_snr in zip(target_snrs, result_snrs)
        ]
    )

    return
