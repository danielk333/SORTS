from datetime import datetime, timezone, timedelta
import numpy as np
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


def test_tracker_controller():
    """just a smoke test for now"""

    eiscat3d = sorts.get_radar("eiscat3d", "stage1-array")

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

    t = sorts.equidistant_sampling(
        orbit=orb,
        start_t=0,
        end_t=3600 * 24 * 1,
        max_dpos=1e3,
    )

    states = prop.propagate(t, orb.cartesian[:, 0], orb.epoch, A=1.0, C_R=1.0, C_D=1.0)

    passes = eiscat3d.find_passes(t, states)

    controller = TrackerController(pass_obj=passes[0][1][0])

    stt_tstmp = datetime.now(timezone.utc)
    end_tstmp = stt_tstmp + timedelta(hours=24)
    result = controller.generate(stt_tstmp, end_tstmp)

    assert isinstance(result, schr.Schedule)
    return


def test_enu_to_azelr():
    pass
