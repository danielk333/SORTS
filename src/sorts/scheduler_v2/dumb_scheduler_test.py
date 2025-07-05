from datetime import datetime
from sorts.schedule_v2 import Schedule
from sorts.scheduler_v2.dumb_scheduler import DumbScheduler
from sorts.controller_v2 import RandomUniformScansController


def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()


def test_DumpScheduler():
    """just a smoke test for now"""

    stt_tstmp = datetime.fromisoformat("2025-01-01 00:00:00")
    end_tstmp = datetime.fromisoformat("2025-01-02 00:00:00")
    result = DumbScheduler(
        controllers=(
            RandomUniformScansController(exp_num=0, min_elevation_deg=75, npoints=10),
            RandomUniformScansController(exp_num=1, min_elevation_deg=5, npoints=2),
        )
    ).generate_schedule(stt_tstmp, end_tstmp)

    assert isinstance(result, Schedule)
    return
