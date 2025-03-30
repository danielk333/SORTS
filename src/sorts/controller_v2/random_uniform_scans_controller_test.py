from datetime import datetime, timezone, timedelta
from .. import scheduler_v2 as schr
from .random_uniform_scans_controller import RandomUniformScansController


def setup_function():
    # a workaround to avoid pytest printings and test printings interleaved in the same line
    # https://github.com/pytest-dev/pytest/issues/8574#issuecomment-1806404215
    print()


def test_random_uniform_scans_controller():
    """just a smoke test for now"""

    controller = RandomUniformScansController(npoints=10)

    stt_tstmp = datetime.now(timezone.utc)
    end_tstmp = stt_tstmp + timedelta(hours=24)
    result = controller.generate(stt_tstmp, end_tstmp)

    assert isinstance(result, schr.Schedule)
    return
