from datetime import datetime, timezone, timedelta
import numpy as np
from .. import scheduler_v2 as schr
from .tracker_controller import TrackerController
from ..passes import Pass


def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()


def test_tracker_controller():
    """just a smoke test for now"""

    npoint = 1000
    passes = [
        Pass(
            t=np.arange(
                np.datetime64("2025-03-01"),
                np.datetime64("2025-03-02"),
                np.timedelta64(1, "m"),  # 1 min
                dtype="datetime64[us]",
            ),
            enu=np.random.uniform(
                size=(3, npoint)
            ),  # (3,n) input matrix of positions in the ENU-convention
            inds=np.arange(0, npoint),  # TODO: tmp val, revisit later
        )
    ]
    controller = TrackerController(passes=passes)

    stt_tstmp = datetime.now(timezone.utc)
    end_tstmp = stt_tstmp + timedelta(hours=24)
    result = controller.generate(stt_tstmp, end_tstmp)

    assert isinstance(result, schr.Schedule)
    return
