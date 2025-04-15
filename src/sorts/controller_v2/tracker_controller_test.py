import typing as t
from datetime import datetime, timezone, timedelta
import numpy as np
from astropy.time import Time
import pyorb
import sorts
from sorts.propagator import SGP4
from .. import scheduler_v2 as schr
from .tracker_controller import TrackerController
from ..passes import Pass


def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()


def tracker_controller_smoke_test():
    """based on `examples/radar_passes.py`"""

    eiscat3d = sorts.get_radar("eiscat3d", "stage1-array")
    epoch = Time(53005.0, format="mjd", scale="utc")

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
        epoch=53005.0,
    )

    dt_arr = sorts.equidistant_sampling(
        orbit=orb,
        start_t=0,
        end_t=3600 * 24 * 1,
        max_dpos=1e3,
    )

    states = prop.propagate(dt_arr, orb.cartesian[:, 0], orb.epoch, A=1.0, C_R=1.0, C_D=1.0)

    passes = eiscat3d.find_passes_v2(
        dt_arr=dt_arr,
        states=states,
        radar_composite_key=("eiscat3d", "stage1-array"),
        epoch=t.cast(datetime, epoch.to_datetime(timezone=timezone.utc)),
    )
    pass_obj = passes[0][1][0]

    controller = TrackerController(pass_obj=pass_obj)

    stt_tstmp = datetime.now(timezone.utc)
    end_tstmp = stt_tstmp + timedelta(hours=24)
    result = controller.generate(stt_tstmp, end_tstmp)

    for k in result:
        assert isinstance(result[k], schr.Schedule)

    return
