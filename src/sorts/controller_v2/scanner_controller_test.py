import typing as t, itertools
from datetime import datetime, timezone
import numpy as np
from astropy.time import Time, TimeDelta
import sorts
from sorts.propagator import SGP4
from .. import scheduler_v2 as schr
from .scanner_controller import ScannerController
from sorts.radar.scans import Fence
from ..passes_v2 import Pass


def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()


def scanner_controller_smoke_test():
    """based on `examples/simulate_scanning_v2.py`"""

    eiscat3d = sorts.get_radar("eiscat3d", "stage1-array")
    stt_tstmp = Time(53005.0, format="mjd", scale="utc")
    end_tstmp = stt_tstmp + TimeDelta(600.0, format="sec")
    scan = Fence(azimuth=90, num=40, dwell=0.1, min_elevation=30)

    Prop_cls = SGP4
    Prop_opts = dict(
        settings=dict(
            out_frame="ITRF",
        ),
    )

    objs = [
        sorts.SpaceObject(
            Prop_cls,
            propagator_options=Prop_opts,
            a=7200e3,
            e=0.02,
            i=75,
            raan=86,
            aop=0,
            mu0=60,
            epoch=stt_tstmp,
            parameters=dict(
                d=0.1,
            ),
        ),
    ]

    dt_arr_list = [
        sorts.equidistant_sampling(
            orbit=obj.state,
            start_t=0,
            end_t=t.cast(np.float64, (end_tstmp - stt_tstmp).to_value("s")).item(),
            max_dpos=1e3,
        )
        for obj in objs
    ]

    states_list = [obj.get_state(dt_arr) for dt_arr, obj in zip(dt_arr_list, objs)]

    passes = eiscat3d.find_passes_v2(
        dt_arr=np.concat(dt_arr_list),  # flatten the list by 1 level,
        states=np.concat(states_list),  # flatten the list by 1 level,
        radar_composite_key=("eiscat3d", "stage1-array"),
        epoch=t.cast(datetime, stt_tstmp.to_datetime(timezone=timezone.utc)),
    )

    # data = scheduler.observe_passes(passes[ind], space_object=objs[ind], snr_limit=False)

    controller = ScannerController(scan=scan)

    # result = controller.generate(
    #     t.cast(datetime, epoch.to_datetime(timezone=timezone.utc))
    #     t.cast(datetime, epoch.to_datetime(timezone=timezone.utc))
    # )

    # for k in result:
    #     assert isinstance(result[k], schr.Schedule)
    for pass_obj in passes[0][1]:
        assert isinstance(pass_obj, Pass)

    return
